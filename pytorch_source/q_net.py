from torch import nn
from torch.nn import functional as F
import pfrl

class ParkingAgent(nn.Module):
    def __init__(self, num_actions, observation_space):
        """Initializes an instance of ParkingAgent
        Arguments:
            num_actions (int): number of actions.
            observation_space (int): length of observation_space.
        """
        super(ParkingAgent, self).__init__()

        self.fc1 = nn.Linear(observation_space, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = self.fc4(x)

        return pfrl.action_value.DiscreteActionValue(out)
