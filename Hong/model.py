import torch
import torch.nn as nn


class Basic_CNN(nn.Module):
    def __init__(self, im_size, nf, num_classes):
        super(Basic_CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=nf),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(nf, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(nf * 2, nf * 4, 3, 1, 1),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Linear(int(im_size / 8) * int(im_size / 8) * (nf * 4), num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x
