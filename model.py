import torch.nn as nn
import torch.nn.functional as F


class LeNetRevised(nn.Module):
    def __init__(self):
        super(LeNetRevised, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(5, 5))
        self.bn1 = nn.BatchNorm2d(num_features=6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.fc1 = nn.Linear(20944, 120)
        self.fc2 = nn.Linear(120, 1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # print(out.size())
        out = F.max_pool2d(out, kernel_size=2)
        # print(out.size())
        out = F.relu(self.bn2(self.conv2(out)))
        # print(out.size())
        out = F.max_pool2d(out, kernel_size=2)
        # print(out.size())
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = F.relu(self.fc1(out))
        # print(out.size())
        out = self.fc2(out)
        return out
