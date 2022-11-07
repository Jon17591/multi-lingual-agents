from models import *
from reinforce import *
import torch
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import wandb
import os


hyperparameter_defaults = dict(
    lr_speaker=0.0003,
    lr_listener=0.0003,
    speaker_ent_target=1.0,
    ps_weight=0.1,
    speaker_ent_bonus=0.00,
    listener_ent_bonus=0.03,
    speaker_lambda=0.3,
    normalise_rewards=True,
    epochs=17,
    pl_weight=0.01,
    ce_weight=0.001,
    speaker=False,
)

debugging = False

if debugging:
    os.environ['WANDB_MODE'] = "offline"
else:
    os.environ['WANDB_MODE'] = "online"

wandb.init(project='reinforce_mnist', config=hyperparameter_defaults)
config = wandb.config
device = "cuda:1"
speaker_conf = {'model': SpeakerNet(), 'device': device, 'lr': config.lr_speaker,
                'speaker': config.speaker, 'listener': False,
                'lambda': torch.Tensor([config.speaker_lambda]).to(device),
                'ent_target': torch.Tensor([config.speaker_ent_target]).to(device),
                'ps_weight': torch.Tensor([config.ps_weight]).to(device), 'ent_bonus': torch.Tensor([config.speaker_ent_bonus]).to(device),
                'normalise_rewards': config.normalise_rewards,
                'batch_size': 32, 'baseline': True}
speaker = PolicyGradient(speaker_conf)


def reward(t1, answer):
    """
    :param t1: Label t1, label of the first image
    :param t2: Label of the 2nd image
    :param answer: the answer given by the agent
    :return: the reward which the agent will observe
    I've set this so that it is only for the 1st agent
    """
    rewards = -torch.ones(t1.shape).to(device)
    indexes = torch.where(answer == (t1))[0]
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


transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

training_data = datasets.MNIST(root='./data',
                               train=True,
                               download=True,
                               transform=transforms)


def train(epochs=1):
    train_dataloader_agent_0 = DataLoader(training_data, batch_size=32, shuffle=True)
    speaker.reset()
    for _ in range(epochs):
        for batch_idx, (data1, target1) in enumerate(
                train_dataloader_agent_0):
            data1, target1 = data1.to(device), target1.to(device)
            answer = speaker.forward([data1]) #returns action as an integar
            r = reward(target1, answer)
            speaker.add_to_buffer(answer, r)
            out = speaker.train()
            out['reward'] = r.mean()
            wandb.log(out)
            if batch_idx % 50 == 0:
                print("%s: %s" % (batch_idx, r.mean()))
        print(_, r.mean())


if __name__ == "__main__":
    train(config.epochs)
    wandb.finish()
