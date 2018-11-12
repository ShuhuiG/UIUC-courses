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


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        # conv1
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 196,
                                kernel_size = 3, stride = 1, padding = 1)
        self.ln1 = nn.LayerNorm([196, 32, 32], elementwise_affine = False)
        self.leakyR1 = nn.LeakyReLU()
        # conv2
        self.conv2 = nn.Conv2d(in_channels = 196, out_channels = 196,
                                kernel_size = 3, stride = 2, padding = 1)
        self.ln2 = nn.LayerNorm([196, 16, 16], elementwise_affine = False)
        self.leakyR2 = nn.LeakyReLU()
        # conv3
        self.conv3 = nn.Conv2d(in_channels = 196, out_channels = 196,
                                kernel_size = 3, stride = 1, padding = 1)
        self.ln3 = nn.LayerNorm([196, 16, 16], elementwise_affine = False)
        self.leakyR3 = nn.LeakyReLU()
        # conv4
        self.conv4 = nn.Conv2d(in_channels = 196, out_channels = 196,
                                kernel_size = 3, stride = 2, padding = 1)
        self.ln4 = nn.LayerNorm([196, 8, 8], elementwise_affine = False)
        self.leakyR4 = nn.LeakyReLU()
        # conv5
        self.conv5 = nn.Conv2d(in_channels = 196, out_channels = 196,
                                kernel_size = 3, stride = 1, padding = 1)
        self.ln5 = nn.LayerNorm([196, 8, 8], elementwise_affine = False)
        self.leakyR5 = nn.LeakyReLU()
        # conv6
        self.conv6 = nn.Conv2d(in_channels = 196, out_channels = 196,
                                kernel_size = 3, stride = 1, padding = 1)
        self.ln6 = nn.LayerNorm([196, 8, 8], elementwise_affine = False)
        self.leakyR6 = nn.LeakyReLU()
        # conv7
        self.conv7 = nn.Conv2d(in_channels = 196, out_channels = 196,
                                kernel_size = 3, stride = 1, padding = 1)
        self.ln7 = nn.LayerNorm([196, 8, 8], elementwise_affine = False)
        self.leakyR7 = nn.LeakyReLU()
        # conv8
        self.conv8 = nn.Conv2d(in_channels = 196, out_channels = 196,
                                kernel_size = 3, stride = 2, padding = 1)
        self.ln8 = nn.LayerNorm([196, 4, 4], elementwise_affine = False)
        self.leakyR8 = nn.LeakyReLU()

        self.fc1 = nn.Linear(196, 1)
        self.fc10 = nn.Linear(196, 10)

    def forward(self, x):
        x = self.leakyR1(self.ln1(self.conv1(x)))
        x = self.leakyR2(self.ln2(self.conv2(x)))
        x = self.leakyR3(self.ln3(self.conv3(x)))
        x = self.leakyR4(self.ln4(self.conv4(x)))
        x = self.leakyR5(self.ln5(self.conv5(x)))
        x = self.leakyR6(self.ln6(self.conv6(x)))
        x = self.leakyR7(self.ln7(self.conv7(x)))
        x = self.leakyR8(self.ln8(self.conv8(x)))
        x = F.max_pool2d(x, kernel_size = 4)
        x = x.view(x.size(0), -1)
        out1 = self.fc1(x)
        out10 = self.fc10(x)
        return (out1, out10)


transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

batch_size_train = 128
batch_size_test = 128

trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,
                                            shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                        download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test,
                                            shuffle=False, num_workers=8)

model =  discriminator()
use_cuda = torch.cuda.is_available()
if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
num_echos = 100
train_accu_epoch = []
test_accu_epoch = []
learning_rate = 0.0001
print("Model Training Starts:")

for epochs in range(num_echos):
    if(epochs==50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/10.0
    if(epochs==75):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/100.0
    #training part
    print("Training Epochs: ",epochs)
    model.train()
    train_accu = []
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
        if(Y_train_batch.shape[0] < batch_size_train):
            continue
        X_train_batch = Variable(X_train_batch).cuda()
        Y_train_batch = Variable(Y_train_batch).cuda()
        _, output = model(X_train_batch)
        loss = criterion(output, Y_train_batch)
        optimizer.zero_grad()
        loss.backward()
        if(epochs>6):
           for group in optimizer.param_groups:
              for p in group['params']:
                 state = optimizer.state[p]
                 if('step' in state and state['step']>=1024):
                    state['step'] = 1000
        optimizer.step()
        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(Y_train_batch.data).sum())/float(batch_size_train))*100.0
        train_accu.append(accuracy)
    accu_train = np.mean(train_accu)
    print("Epoch Train Accu: ", accu_train)
    train_accu_epoch.append(accu_train)

    #testing part
    print("Testing Epochs: ",epochs)
    model.eval()
    test_accu = []
    for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
        if(Y_test_batch.shape[0] < batch_size_test):
            continue
        X_test_batch = Variable(X_test_batch).cuda()
        Y_test_batch = Variable(Y_test_batch).cuda()
        _, output = model(X_test_batch)
        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(Y_test_batch.data).sum())/float(batch_size_test))*100.0
        test_accu.append(accuracy)
    accu_test = np.mean(test_accu)
    print("Epoch Test Accu: ", accu_test)
    test_accu_epoch.append(accu_test)

torch.save(model,'cifar10.model')
save = pd.DataFrame({"Train" : np.array(train_accu_epoch), "Test" : np.array(test_accu_epoch)})
save.to_csv("part1.csv", index=False)
