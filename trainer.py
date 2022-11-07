import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import wandb


def train(s, l, epochs, device, index, train, save, group):
    """
    :param s: speaker network
    :param l: listener network
    :param epochs: number of episodes to run for
    :param device: cuda to go to
    :return: none
    """
    if train:
        run = wandb.init(project='em_comms', name=str(float(s.config['l_c'])), group=group)
    else:
        logger = Logger()
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
    train_dataloader_agent_0 = DataLoader(dataset, batch_size=32, shuffle=True)
    train_dataloader_agent_1 = DataLoader(dataset, batch_size=32, shuffle=True)
    s.reset()
    l.reset()
    id1 = torch.eye(20)[int(s.config['name'][1])].to(device).repeat(32, 1)
    id2 = torch.eye(20)[int(l.config['name'][1])].to(device).repeat(32, 1)
    for epoch in range(epochs):
        count = torch.zeros((10, 20))
        for batch_idx, ((data1, target1), (data2, target2)) in enumerate(
                zip(train_dataloader_agent_0, train_dataloader_agent_1)):
            data1, target1 = data1.to(device), target1.to(device)
            data2, target2 = data2.to(device), target2.to(device)
            # reset trained_models
            message, s_value = s.forward([data1, id2])
            count[target1, message] += 1
            message = torch.eye(20)[message.detach()].to(device)
            answer, l_value = l.forward([data2, message, id1])
            r = reward(target1, target2, answer)
            if train:
                s.add_to_buffer(message, r)
                l.add_to_buffer(answer, r)
                loss_speaker = s.train()
                loss_listener = l.train()
                out = combine(loss_speaker, loss_listener, {s.config['name']+'_iter': float(s.iterations),
                                                            s.config['name']: float(r.mean().item()),
                                                            l.config['name']: float(r.mean().item()),
                                                            l.config['name']+'_iter': float(l.iterations)})
                run.log(out)
            else:
                out = {'reward': r.mean()}
                logger.add(out)
            if batch_idx % 50 == 0:
                pass
                #print(batch_idx, r.mean(), s_value.mean(), l_value.mean())

    if save:
        s.save_model(index)
        l.save_model(index)
    if train:
        run.finish()
        return r.mean()
    else:
        logger.process() # average over epoch
        logger.logs.update({'policy': count/count.sum(), 'name': s.config['name'] + '_' + l.config['name']})
        return logger.logs


def reward(t1, t2, answer):
    """
    :param t1: Label t1, label of the first image
    :param t2: Label of the 2nd image
    :param answer: the answer given by the agent
    :return: the reward which the agent will observe
    I've set this so that it is only for the 1st agent
    """
    rewards = -torch.ones_like(t1).type(torch.float32)
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
    dict1 = {k: float(v.item()) for k, v in dict1.items() if v is not None}
    dict2 = {k: float(v.item()) for k, v in dict2.items() if v is not None}
    dict2.update(dict1)
    dict2.update(rew)
    return dict2


class Logger:

    def __init__(self):
        self.logs = {}

    def add(self, new_data):
        if self.logs:
            for i in self.logs.keys():
                self.logs[i].append(new_data[i])
        else:
            self.logs = {i: [j] for i, j in new_data.items()}

    def process(self):
        self.logs = {i: sum(j)/len(j) for i, j in self.logs.items()}
