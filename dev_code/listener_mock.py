from models import *
from reinforce import *
import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import wandb
import os


hyperparameter_defaults = dict(
    lr_speaker=0.0003,
    lr_listener=0.001,
    speaker_ent_target=1.0,
    ps_weight=0.1,
    speaker_ent_bonus=0.0,
    listener_ent_bonus=0.0,
    speaker_lambda=0.3,
    normalise_rewards=False,
    epochs=17,
    pl_weight=0.01,
    ce_weight=0.001,
    listener=False,
)

debugging = False

if debugging:
    os.environ['WANDB_MODE'] = "offline"
else:
    os.environ['WANDB_MODE'] = "online"

wandb.init(project='reinforce_mnist', config=hyperparameter_defaults)
config = wandb.config
device = "cuda:1"


listener_conf = {'model': ListenerMock(), 'device': device, 'lr': config.lr_listener,
                 'speaker': False, 'listener': config.listener,
                 'pl_weight': torch.Tensor([config.pl_weight]).to(device),
                 'ent_bonus': torch.Tensor([config.listener_ent_bonus]).to(device),
                 'normalise_rewards': config.normalise_rewards, 'lr_no_m': config.lr_listener,
                 'batch_size': 32, 'ce_weight': torch.Tensor([config.ce_weight]).to(device),
                 'baseline': False}

listener = PolicyGradient(listener_conf)


def reward(t1, t2, answer):
    """
    :param t1: Label t1, label of the first image
    :param t2: Label of the 2nd image
    :param answer: the answer given by the agent
    :return: the reward which the agent will observe
    I've set this so that it is only for the 1st agent
    Changing this to be -1, 1 seems to result in more consistent performance
    """
    rewards = -torch.ones(t1.shape).to(device)
    indexes = torch.where(answer == (t1+t2))[0]
    rewards[indexes] = 1.0
    return rewards


def combine(dict1, dict2, rew):
    """
    Purpose of this function is to combine the two loss outputs and to handle any clashes
    First remove None values, second change name clashes, third combine

    :param dict1: speaker output
    :param dict2: listener output
    :return: combined dictionary
    """
    dict1 = {k: v for k, v in dict1.items() if v is not None}
    dict2 = {k: v for k, v in dict2.items() if v is not None}
    dict1['loss_speaker'] = dict1.pop('loss')
    dict2['loss_listener'] = dict2.pop('loss')
    dict1['ent_speaker'] = dict1.pop('ent')
    dict2['ent_listener'] = dict2.pop('ent')
    dict2.update(dict1)
    dict2['reward'] = rew
    return dict2


def train(epochs=1):
    listener.reset()
    to_onehot = lambda x: F.one_hot(x, num_classes=20).type(torch.float32)
    for _ in range(epochs):
        for i in range(1875): #number of size 32 batches in MNIST
            number_1 = torch.randint(0, 9, (32,)).to(device)
            number_2 = torch.randint(0, 9, (32,)).to(device)
            # reset trained_models
            answer = listener.forward([to_onehot(number_1), to_onehot(number_2)])
            r = reward(number_1, number_2, answer)
            listener.add_to_buffer(answer, r)
            out = listener.train()
            out['reward'] = r.mean()
            wandb.log(out)
            if i % 50 == 0:
                print("%s: %s" % (i, r.mean()))
        print(_, r.mean())


if __name__ == "__main__":
    train(config.epochs)
