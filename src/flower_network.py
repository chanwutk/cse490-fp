import torch.nn as nn
import torch.nn.functional as F


class FlowerNetwork(nn.Module):
    def __init__(self, save=True):
        super(FlowerNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 7, 1)
        self.conv2 = nn.Conv2d(10, 10, 7, 1)
        self.conv3 = nn.Conv2d(10, 20, 5, 1)
        self.conv4 = nn.Conv2d(20, 20, 5, 1)
        self.conv5 = nn.Conv2d(20, 40, 5, 1)
        self.fc1 = nn.Linear(40 * 30 * 30, 500)
        self.fc2 = nn.Linear(500, 5)
        self.save = save

    def forward(self, x):
        x_before = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x) + x_before)
        x = F.max_pool2d(x, 2, 2)

        x_before = x
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x) + x_before)
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 40 * 30 * 30)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
