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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
import time
import argparse
import datetime
import logging

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

batch_size = 100
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
testloader = enumerate(testloader)


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

    def forward(self, x):
        x = self.leakyR1(self.ln1(self.conv1(x)))
        x = self.leakyR2(self.ln2(self.conv2(x)))
        x = self.leakyR3(self.ln3(self.conv3(x)))
        x = self.leakyR4(self.ln4(self.conv4(x)))
        x = self.leakyR5(self.ln5(self.conv5(x)))
        x = self.leakyR6(self.ln6(self.conv6(x)))
        x = self.leakyR7(self.ln7(self.conv7(x)))
        x = self.conv8(x)
        x = F.max_pool2d(x, kernel_size = 4)
        x = x.view(-1, 196)
        return x


def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig

model =  discriminator()
use_cuda = torch.cuda.is_available()
if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

model = torch.load('../part1/cifar10.model')
model.cuda()
model.eval()

batch_idx, (X_batch, Y_batch) = testloader.__next__()
X_batch = Variable(X_batch,requires_grad=True).cuda()
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).cuda()
Y_batch = Variable(Y_batch).cuda()

X = X_batch.mean(dim=0)
X = X.repeat(batch_size,1,1,1)

Y = torch.arange(batch_size).type(torch.int64)
Y = Variable(Y).cuda()

lr = 0.1
weight_decay = 0.001
for i in range(200):
    output = model(X)

    loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).cuda(),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
    print(i,accuracy,-loss)

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

## save new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('../visualization/max_features_8layers.png', bbox_inches='tight')
plt.close(fig)
