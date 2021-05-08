#%%
import torch.nn as nn
import torch.nn.functional as F
import torch
import ipdb
import torchvision
#%%
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(12, 6, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(6)
        self.fc1 = nn.Linear(507**2*6, 120)
        self.fc2 = nn.Linear(120, 2)
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.pool1(x)

        x = self.bn2(self.conv2(x))
        x = self.relu(x)

        x = self.bn3(self.conv3(x))
        x = self.relu(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MyNet1(nn.Module):
    def __init__(self):
        super(MyNet1, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(((1024-(3-1))//2)**2*6, 120)
        self.fc2 = nn.Linear(120, 2)
    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = torch.sigmoid(self.fc2(x))
        return x

#%%
if __name__ =="__main__":
    # %%
    model = MyNet()

    # input = torch.autograd.Variable(torch.randn(1, 1, 1024, 1024))
    input = torch.randn(3, 1, 1024, 1024)

    # torchvision.models.resnext50_32x4d()
    # model = torchvision.models.resnet18(pretrained=True)
    # model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=7, stride=2, padding=3,
    #                            bias=False)
    # model.fc = nn.Linear(model.fc.in_features, 2)
    # input = torch.randn(3, 1, 1000, 1000)
    # ipdb.set_trace()
    o = model(input)
