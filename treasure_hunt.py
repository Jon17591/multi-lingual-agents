import gym
from gym import spaces
import matplotlib.pyplot as plt
import torch
import numpy as np


class TreasureHunt(gym.Env):
    """
    Custom environment defined according to Eccles.
    """
    def __init__(self):
        super(TreasureHunt, self).__init__()
        self.action_space = spaces.MultiDiscrete([5, 5])
        self.observation_space = spaces.Box(low=0.0, high=255.0, shape=(3, 18, 24), dtype=np.float32)
        self.grid = None
        self.agent_loc = None
        self.treasure_location = None
        self.action_dict = {0: torch.Tensor([0, 0]),
                            1: torch.Tensor([1, 0]),
                            2: torch.Tensor([0, 1]),
                            3: torch.Tensor([-1, 0]),
                            4: torch.Tensor([0, -1])}

    def step(self, action):
        r = 0
        done = False
        info = {}
        self.update_agent_location(actions=action)
        if torch.all(torch.eq(self.agent_loc[0], self.treasure_location)):
            r = 1
            done = True
        return self.grid.numpy(), r, done, info

    def reset(self):
        self.grid = torch.Tensor(3, 18, 24)
        self.grid.fill_(128)  # everything is now green
        self.grid[:, [1,-2], 1:-1] = 0
        tunnels = torch.Tensor(4)
        tunnels[0] = torch.randint(1, 23, (1,))
        for i in range(1, tunnels.shape[0]):
            done = False
            while not done:
                number = torch.randint(1, 23, (1,))
                if torch.all((tunnels[:i] - number).abs() > 3):
                    tunnels[i] = number
                    done = True
        self.grid[:, 1:-3, tunnels.long()] = 0.0
        treasure = torch.randint(0, 4, (1,))
        self.grid[:, -4, tunnels[treasure].long()] = torch.Tensor([[255.0], [255.0], [0.0]])
        self.treasure_location = torch.Tensor([tunnels[treasure].long(), 14])
        self.update_agent_location()
        return self.grid.numpy()

    def update_agent_location(self, actions = None):
        if actions is None:
            x = torch.randint(1, 23, (2,))
            y = torch.Tensor([1,16])
            self.grid[:, y.long(), x.long()] = torch.Tensor([[255.0], [0.0], [0.0]])
            self.agent_loc = torch.stack((x.float(), y.float()), dim=1)
        else:
            # Remove red referring to old location
            self.grid[:, self.agent_loc[:, 1].long(), self.agent_loc[:, 0].long()] = torch.Tensor([[0.0], [0.0], [0.0]])
            # Compute the new location according to the action
            translation = torch.stack([self.action_dict[c] for c in actions], dim=0)
            test_loc = self.agent_loc + translation
            # Test the new location is not invalid. If valid, change location.
            for num, i in enumerate(self.grid[:, test_loc[:, 1].long(), test_loc[:, 0].long()].chunk(2, dim=1)):
                if not torch.all(torch.eq(i, torch.Tensor([[128.0], [128.0], [128.0]]))):
                    self.agent_loc[num] = test_loc[num]
                else:
                    print('Error')
            self.grid[:, self.agent_loc[:, 1].long(), self.agent_loc[:, 0].long()] = torch.Tensor([[255.0], [0.0], [0.0]])


def plotter(grid):
    plt.imshow(grid.transpose(0, 2).transpose(0, 1))
    plt.xlim(0, 23)
    plt.grid()
    plt.xticks(range(0, 24))
    plt.yticks(range(0, 18))
    plt.show()


if __name__ == "__main__":
    # this seems to work
    env = TreasureHunt()
    env.reset()
    plotter(env.grid)
    print('stay still')
    env.step([0, 0])
    plotter(env.grid)
    print('right')
    env.step([1, 1])
    plotter(env.grid)
    print('up')
    env.step([2, 2])
    plotter(env.grid)
    print('left')
    env.step([3, 3])
    plotter(env.grid)
    print('down')
    env.step([4, 4])
    plotter(env.grid)#hing is now green