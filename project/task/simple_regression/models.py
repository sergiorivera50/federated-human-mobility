from torch import nn

from project.types.common import NetGen
from project.utils.utils import lazy_config_wrapper


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.main = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.main(x)


# Simple wrapper to match the NetGenerator Interface
get_net: NetGen = lazy_config_wrapper(Net)
