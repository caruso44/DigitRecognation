import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride = 1),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, stride = 2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride = 2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(800, 256),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 10)
        )
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
