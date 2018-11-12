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


#data preprocessing and augmentation
#data tranformation and augmentation for training dataset
trans_train = transforms.Compose([
                    transforms.Resize(size=(224, 224)),
                    transforms.RandomCrop(224, padding=4),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441),
                                        (0.267, 0.256, 0.276))])
train_data = torchvision.datasets.CIFAR100(root='../data',
                                            train=True, transform=trans_train, download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True)

#corresponding test data transformation
trans_test = transforms.Compose([
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441),
                                        (0.267, 0.256, 0.276))])
test_data = torchvision.datasets.CIFAR100(root='../data',
                                            train=False, transform=trans_test, download=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100,shuffle=False)


ResNet = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
ResNet.load_state_dict(torch.load("../data/resnet18-5c106cde.pth"))
num_ftrs = ResNet.fc.in_features
ResNet.fc = nn.Linear(num_ftrs, 100)


num_echos = 20
train_accu_epoch = []
train_loss = []
test_accu_epoch = []
batch_size_train = 50
batch_size_test = 100
optimizer = optim.Adam(ResNet.parameters())
criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.2)

use_cuda = torch.cuda.is_available()
if use_cuda:
    ResNet.cuda()
    ResNet = torch.nn.DataParallel(ResNet, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


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
save.to_csv("save2.csv", index=False)
