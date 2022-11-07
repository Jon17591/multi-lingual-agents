import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerNet(nn.Module):
    """
    """
    def __init__(self, identifier=False, mode='q_critic', feature_extraction=False, head_copy=False):
        super(SpeakerNet, self).__init__()
        self.identifier = identifier
        self.mode = mode
        self.feature_extraction = feature_extraction
        self.head_copy = head_copy
        if self.feature_extraction:
            self.outputs = 4
        else:
            self.outputs = 1
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        self.heads = nn.ModuleList([LanguageHead(self.identifier, agent='speaker', mode=self.mode) for _ in
                                    range(self.outputs)])
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x, i=None, index=None):
        """
        :param x:
        :param i:
        :param index:
        :return:
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)  # change to flatten
        outputs = []
        for num, head in enumerate(self.heads):
            if num == 0:
                outputs.append(head((x, None, i)))
            else:
                outputs.append(head((x.detach(), None, i)))
        if self.feature_extraction:
            return outputs[index]
        else:
            return outputs[0]

    def copy_trained_head(self):
        """
        to copy heads
        :return:
        """
        for num, head in enumerate(self.heads[1:]):
            head.load_state_dict(self.heads[0].state_dict())


class ListenerNet(nn.Module):
    def __init__(self, identifier=False, mode='q_critic', feature_extraction=False, head_copy=False):
        super(ListenerNet, self).__init__()
        self.identifier = identifier
        self.mode = mode
        self.feature_extraction = feature_extraction
        self.head_copy = head_copy
        if self.feature_extraction:
            self.outputs = 4
        else:
            self.outputs = 1
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        self.heads = nn.ModuleList([LanguageHead(self.identifier, agent='listener', mode=self.mode) for _ in
                                    range(self.outputs)])
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x, m, i=None, index=None):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)  # change to flatten
        outputs = []
        for num, head in enumerate(self.heads):
            if num == 0:
                outputs.append(head((x, m, i)))
            else:
                outputs.append(head((x.detach(), m, i)))
        if self.feature_extraction:
            return outputs[index]
        else:
            return outputs[0]

    def copy_trained_head(self):
        """
        to copy heads
        :return:
        """
        for num, head in enumerate(self.heads[1:]):
            head.load_state_dict(self.heads[0].state_dict())


class LanguageHead(nn.Module):
    """
    Language head per model. Speaker and listener have slightly different structures.
    """

    def __init__(self, identifier, agent='speaker', mode='q_critic'):
        super(LanguageHead, self).__init__()
        self.identifier = identifier
        self.mode = mode
        self.agent = agent
        self.fc1 = nn.Linear(self.calculate_input(), 1024)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        self.fc2 = nn.Linear(1024, 1024)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        self.logits = nn.Linear(1024, 19)  # this should be 19
        nn.init.kaiming_uniform_(self.logits.weight, nonlinearity='relu')
        if self.mode == 'q_critic':
            self.values = nn.Linear(1024, 19)  # this should be 19
        elif self.mode == 'baseline':
            self.values = nn.Linear(1024, 1)  # this should be 19
        else:
            self.values = nn.Linear(1024, 1)  # this should be 19

    def forward(self, input_args):
        x, m, i = input_args
        if self.identifier:
            if self.agent == 'listener':
                x = torch.cat((x, m, i), dim=1)  # concat message
            else: # speaker
                x = torch.cat((x, i), dim=1)
        else:
            if self.agent == 'listener':
                x = torch.cat((x, m), dim=1)  # concat message
            else: # speaker
                x = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        l = F.relu(self.logits(x))
        v = self.values(x)
        return F.log_softmax(l, dim=1), v

    def calculate_input(self):
        """
        This calculates the input size dependent on the model configurations
        :return: integar
        """
        input_size = (self.identifier * 20) + 1024 + ((self.agent == 'listener') * 20)
        return input_size


