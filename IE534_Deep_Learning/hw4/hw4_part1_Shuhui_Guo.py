import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import pandas as pd
from torch.optim import lr_scheduler


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_planes, out_channels = out_planes,
                                kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(in_channels = out_planes, out_channels = out_planes,
                                kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels = in_planes, out_channels = out_planes,
                            kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return out

class ResNet(nn.Module):
    def __init__(self, block):
        super(ResNet, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32,
                                kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout2d(0.2)
        self.layer1 = self._make_layer(block, 32, 2, stride=1)
        self.layer2 = self._make_layer(block, 64, 4, stride=2)
        self.layer3 = self._make_layer(block, 128, 4, stride=2)
        self.layer4 = self._make_layer(block, 256, 2, stride=2)
        self.linear = nn.Linear(1024, 100)

    def _make_layer(self, block, out_planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, out_planes, stride))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.max_pool2d(out, kernel_size = 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


#data preprocessing and augmentation
#data tranformation and augmentation for training dataset
trans_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441),
                                        (0.267, 0.256, 0.276))])
train_data = torchvision.datasets.CIFAR100(root='../data',
                                            train=True, transform=trans_train, download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)

#corresponding test data transformation
trans_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441),
                                        (0.267, 0.256, 0.276))])
test_data = torchvision.datasets.CIFAR100(root='../data',
                                            train=False, transform=trans_test, download=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=200,shuffle=False)


ResNet = ResNet(BasicBlock)
num_echos = 30
train_accu_epoch = []
train_loss = []
test_accu_epoch = []
batch_size_train = 100
batch_size_test = 200
use_cuda = torch.cuda.is_available()
if use_cuda:
    ResNet.cuda()
    ResNet = torch.nn.DataParallel(ResNet, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
optimizer = optim.Adam(ResNet.parameters())
criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.15)


print("Model Training Starts:")

for epochs in range(num_echos):
    #training part
    print("Training Epochs: ",epochs)
    scheduler.step()
    ResNet.train()
    train_accu = []
    for index, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        data, target = Variable(data), Variable(target)
        output = ResNet(data)
        loss = criterion(output, target)
        loss.backward()
        if(epochs>5):
           for group in optimizer.param_groups:
              for p in group['params']:
                 state = optimizer.state[p]
                 if(state['step']>=1024):
                    state['step'] = 1000
        optimizer.step()
        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(target.data).sum())/float(batch_size_train))*100.0
        train_accu.append(accuracy)
    accu_train = np.mean(train_accu)
    print("Epoch Train Accu: ", accu_train)
    train_accu_epoch.append(accu_train)

    #testing part
    print("Testing Epochs: ",epochs)
    ResNet.eval()
    test_accu = []
    for index, (data, target) in enumerate(test_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = ResNet(data)
        loss = criterion(output, target)
        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(target.data).sum())/float(batch_size_test))*100.0
        test_accu.append(accuracy)
    accu_test = np.mean(test_accu)
    print("Epoch Test Accu: ", accu_test)
    test_accu_epoch.append(accu_test)


save = pd.DataFrame({"Train" : np.array(train_accu_epoch), "Test" : np.array(test_accu_epoch)})
save.to_csv("save.csv", index=False)
