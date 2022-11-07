from models import *
from reinforce import *
import torch
import wandb
import os
from trainer import train
import sys
import torch.multiprocessing as mp
import random
#import faulthandler
#faulthandler.enable()


hyperparameter_defaults = dict(
    lr_speaker=0.0003,
    lr_listener=0.0003,
    speaker_ent_target=1.0,
    ps_weight=0.1,
    speaker_ent_bonus=0.0,
    listener_ent_bonus=0.03,
    speaker_lambda=0.3,
    normalise_rewards=False,
    epochs=40,
    pl_weight=0.01,
    ce_weight=0.001,
    train=True,
    save=False,
    load=False,
    speaker_model='trained_models/speaker_0.pt',
    listener_model='trained_models/listener_0.pt',
    identifier=True,
    speaker_consistency=False,
    listener_consistency=False,
    l_c=0.25,
    v_weight=0.01,
    mode='none',
    feature_extraction=True,
    head_copy=True,
)


def generate_speaker(name):
    speaker_conf = {'model': SpeakerNet(identifier=config.identifier, mode=config.mode,
                                        feature_extraction=config.feature_extraction, head_copy=config.head_copy),
                    'device': device,
                    'lr': config.lr_speaker,
                    'speaker': True, 'listener': False,
                    'lambda': torch.Tensor([config.speaker_lambda]),
                    'ent_target': torch.Tensor([config.speaker_ent_target]),
                    'ps_weight': torch.Tensor([config.ps_weight]), 'ent_bonus': torch.Tensor([config.speaker_ent_bonus]),
                    'normalise_rewards': config.normalise_rewards,
                    'batch_size': 32, 'name': name,
                    'speaker_consistency': config.speaker_consistency,
                    'listener_consistency': False, 'l_c': torch.Tensor([config.l_c]),
                    'v_weight': torch.Tensor([config.v_weight]), 'mode': config.mode,
                    'feature_extraction': config.feature_extraction, 'head_copy': config.head_copy}
    return PolicyGradient(speaker_conf)


def generate_listener(name):
    listener_conf = {'model': ListenerNet(identifier=config.identifier, mode=config.mode,
                                          feature_extraction=config.feature_extraction, head_copy=config.head_copy),
                     'device': device,
                     'lr': config.lr_listener,
                     'speaker': False, 'listener': True,
                     'pl_weight': torch.Tensor([config.pl_weight]),
                     'ent_bonus': torch.Tensor([config.listener_ent_bonus]),
                     'normalise_rewards': config.normalise_rewards, 'lr_no_m': config.lr_listener,
                     'batch_size': 32, 'ce_weight': torch.Tensor([config.ce_weight]),
                     'name': name, 'listener_consistency': config.listener_consistency, 'speaker_consistency': False,
                     'l_c': torch.Tensor([config.l_c]), 'v_weight': torch.Tensor([config.v_weight]),
                     'mode': config.mode,
                     'feature_extraction': config.feature_extraction, 'head_copy': config.head_copy}
    return PolicyGradient(listener_conf)


def launch_multi(speaker_list, listener_list, group, evaluate=False):
    """
    A very messy function, but if I need the output of train i currently use star map. If I don't need the output
    I use Process
    :param speaker_list:
    :param listener_list:
    :param group:
    :param evaluate:
    :return:
    """
    if evaluate:
        train_model = False
        epochs = 1
        save = False
    else:
        train_model = config.train
        epochs = config.epochs
        save = config.save
    processes = []
    print([s.config['name'] for s in speaker_list], [l.config['name'] for l in listener_list])
    to_args = []
    for i, (s, l) in enumerate(zip(speaker_list, listener_list)):
        s.move_cuda('cuda:%s' % i)
        l.move_cuda('cuda:%s' % i)
        if not evaluate:
            id1 = torch.eye(20)[int(s.config['name'][1])].repeat(32, 1)
            id2 = torch.eye(20)[int(l.config['name'][1])].repeat(32, 1)
            s.new_pairing(id2)
            l.new_pairing(id1)
        to_args.append((s, l, epochs, 'cuda:%s' % i, index, train_model, save, group))
    if evaluate:
        with mp.Pool(processes=4) as pool:
            results = pool.starmap(train, to_args)
        return results
    else:
        processes = []
        for i in to_args:
            p = mp.Process(target=train, args=i)
            p.start()
            processes.append(p)
        for p in processes:
            p.join()



def launch_single(speaker_list, listener_list, group):
    print([s.config['name'] for s in speakers], [l.config['name'] for l in listeners])
    for i, (s, l) in enumerate(zip(speaker_list, listener_list)):
        s.move_cuda('cuda:%s' % i)
        l.move_cuda('cuda:%s' % i)
        id1 = torch.eye(20)[int(s.config['name'][1])].repeat(32, 1)
        id2 = torch.eye(20)[int(l.config['name'][1])].repeat(32, 1)
        s.new_pairing(id2)
        l.new_pairing(id1)
        train(s, l, config.epochs, 'cuda:%s' % i, index, config.train, config.save, group)


def sort_list(to_sort):
    to_sort_names = [s.config['name'] for s in to_sort]
    return [x for _, x in sorted(zip(to_sort_names, to_sort))]


def load_models(to_load):
    for i in to_load:
        i.policy.load_state_dict(torch.load('trained_models/%s.pt' % i.config['name']))
        # old pairing
        old_partner = torch.eye(20)[int(i.config['name'][1])].repeat(32, 1)
        i.new_pairing(old_partner)
    return to_load


def validate(speaker_list, listener_list, group, old_data, first=False):
    returned_output = []
    formatted_output = {}
    for i in range(len(speaker_list)):
        speaker_list = speaker_list[i:]+speaker_list[:i]
        out = launch_multi(speaker_list, listener_list, group, evaluate=True)
        returned_output.append(out)
    returned_output = [item for sublist in returned_output for item in sublist]
    if not first:
        for i, j in zip(returned_output, old_data):
            formatted_output[i['name'] + '_norm'] = torch.norm(i['policy'] - j['policy'], p = 1)
            formatted_output[i['name'] + '_reward'] = i['reward']
    else:
        for i in returned_output:
            formatted_output[i['name'] + '_norm'] = 0
            formatted_output[i['name'] + '_reward'] = i['reward']
    return returned_output, formatted_output


if __name__ == "__main__":
    debugging = False
    multiprocess = True
    pairs = 4

    if len(sys.argv) == 1:
        cuda_device = 'cuda:1'
        index = 0
    else:
        cuda_device = sys.argv[1]
        index = sys.argv[2]

    if debugging:
        os.environ['WANDB_MODE'] = "offline"
    else:
        os.environ['WANDB_MODE'] = "online"

    group = 'feature_extract_with_weight_copy'
    logger = wandb.init(project='em_comms', config=hyperparameter_defaults, name='high_level_tracker', group=group)
    config = wandb.config
    device = cuda_device
    output = None
    mp.set_start_method('spawn', force=True)
    speakers = [generate_speaker('s'+str(i)) for i in range(pairs)]
    listeners = [generate_listener('l'+str(i)) for i in range(pairs)]
    if config.load:
        speakers = load_models(speakers)
        listeners = load_models(listeners)
    if not multiprocess:
        launch_single(speakers, listeners, group=group)
    else:
        for i in range(2):
            output, formatted_output = validate(speakers, listeners, group, output, first=i == 0)
            logger.log(formatted_output)
            launch_multi(speakers[i:]+speakers[:i], listeners, group)
        output, formatted_output = validate(speakers, listeners, group, output, first=False)
        logger.log(formatted_output)
        print('Return to OG')
        speakers = sort_list(speakers)
        listeners = sort_list(listeners)
        launch_multi(speakers, listeners, group)
        output, formatted_output = validate(speakers, listeners, group, output, first=False)
        logger.log(formatted_output)
    logger.finish()
