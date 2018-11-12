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
#define the model we will use_cuda
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        #1
        self.c1 = nn.Conv2d(in_channels = 3, out_channels = 64,
                               kernel_size = 4, stride=1, padding=2)
        self.b1 = nn.BatchNorm2d(64)
        self.c2 = nn.Conv2d(in_channels = 64, out_channels = 64,
                               kernel_size = 4, stride=1, padding=2)
        self.d1 = nn.Dropout2d(p=0.5)
        #2
        self.c3 = nn.Conv2d(in_channels = 64, out_channels = 128,
                               kernel_size = 4, stride=1, padding=2)
        self.b2 = nn.BatchNorm2d(128)
        self.c4 = nn.Conv2d(in_channels = 128, out_channels = 128,
                               kernel_size = 4, stride=1, padding=2)
        self.d2 = nn.Dropout2d(p=0.5)
        #3
        self.c5 = nn.Conv2d(in_channels = 128, out_channels = 64,
                               kernel_size = 4, stride=1, padding=2)
        self.b3 = nn.BatchNorm2d(64)
        self.c6 = nn.Conv2d(in_channels = 64, out_channels = 64,
                               kernel_size = 3, stride=1, padding=0)
        self.d3 = nn.Dropout2d(p=0.5)
        #4
        self.c7 = nn.Conv2d(in_channels = 64, out_channels = 64,
                               kernel_size = 3, stride=1, padding=0)
        self.b4 = nn.BatchNorm2d(64)
        self.c8 = nn.Conv2d(in_channels = 64, out_channels = 64,
                               kernel_size = 3, stride=1, padding=0)
        self.b5 = nn.BatchNorm2d(64)
        self.d4 = nn.Dropout2d(p=0.5)
        #5
        self.fc1 = nn.Linear(1024, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 10)
    def forward(self, x):
        #1
        x = F.relu(self.b1(self.c1(x)))
        x = F.max_pool2d(F.relu(self.c2(x)),kernel_size = 2)
        x = self.d1(x)
        #2
        x = F.relu(self.b2(self.c3(x)))
        x = F.max_pool2d(F.relu(self.c4(x)),kernel_size = 2)
        x = self.d2(x)
        #3
        x = F.relu(self.b3(self.c5(x)))
        x = F.relu(self.c6(x))
        x = self.d3(x)
        #4
        x = F.relu(self.b4(self.c7(x)))
        x = F.relu(self.b5(self.c8(x)))
        x = self.d4(x)
        #5
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

#data preprocessing and augmentation
#data tranformation and augmentation for training dataset
trans_train = transforms.Compose([
transforms.RandomRotation(2),          transforms.RandomVerticalFlip(p=0.3), transforms.ToTensor(),
                    transforms.Normalize((0.49139968, 0.48215841, 0.44653091),(0.24703223, 0.24348513, 0.26158784))])
train_data = torchvision.datasets.CIFAR10(root='~/IE534',train=True,transform=trans_train,download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=50,shuffle=True)

#corresponding test data transformation
trans_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                                     (0.24703223, 0.24348513, 0.26158784))])
test_data = torchvision.datasets.CIFAR10(root='~/IE534',
                                             train=False,
                                             transform=trans_test,
                                             download=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100,shuffle=False)


model = Model()
num_echos = 50
train_accu_epoch = []
train_loss = []
test_accu_epoch = []
batch_size_train = 50
batch_size_test = 100
use_cuda = torch.cuda.is_available()
if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
optimizer = optim.Adam(model.parameters())

print("Model Training Starts:")
for epochs in range(num_echos):
    #training part
    print("Training Epochs ",epochs)
    model.train()
    train_accu = []
    for index, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        optimizer.zero_grad()
        loss = F.nll_loss(output, target)
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
        #if(index%100 == 1):
        #    print("Step:", index, " Training Accuracy: ",accuracy)
    accu_train = np.mean(train_accu)
    print("Epoch Train Accu: ", accu_train)
    train_accu_epoch.append(accu_train)
    #testing part
    print("Testing Epochs ",epochs)
    model.eval()
    test_accu = []
    for index, (data, target) in enumerate(test_loader):
        data, target = Variable(data), Variable(target)
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(target.data).sum())/float(batch_size_test))*100.0
        test_accu.append(accuracy)
    accu_test = np.mean(test_accu)
    print("Epoch Test Accu: ", accu_test)
    test_accu_epoch.append(accu_test)

save = pd.DataFrame({"Train" : np.array(train_accu_epoch), "Test" : np.array(test_accu_epoch)})
save.to_csv("save.csv", index=False)
