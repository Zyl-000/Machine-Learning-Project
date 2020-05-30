import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.nn.functional as F

import torchvision.datasets as dset
import torchvision.transforms as T

from models import TwoLayerFC, ThreeLayerConvNet, AlexNet

import matplotlib.pyplot as plt

import numpy as np

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

NUM_TRAIN = 49000

# The torchvision.transforms package provides tools for preprocessing data
# and for performing data augmentation; here we set up a transform to
# preprocess the data by subtracting the mean RGB value and dividing by the
# standard deviation of each RGB value; we've hardcoded the mean and std.
transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

# We set up a Dataset object for each split (train / val / test); Datasets load
# training examples one at a time, so we wrap each Dataset in a DataLoader which
# iterates through the Dataset and forms minibatches. We divide the CIFAR-10
# training set into train and val sets by passing a Sampler object to the
# DataLoader telling how it should sample from the underlying Dataset.
cifar10_train = dset.CIFAR10('./datasets', train=True, download=True,
                             transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=64,
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

cifar10_val = dset.CIFAR10('./datasets', train=True, download=True,
                           transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=64,
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

cifar10_test = dset.CIFAR10('./datasets', train=False, download=True,
                            transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=64)

'''
for img, lab in cifar10_train:
    print(img)
    plt.show()
    img=img.numpy().transpose(1,2,0)
    #print(img)
    plt.title(str(classes[lab]))
    plt.imshow(img)
    plt.pause(1)
'''




USE_GPU = True

dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss
print_every = 100

print('using device:', device)


def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

def test_flatten():
    x = torch.arange(12).view(2, 1, 3, 2)
    print('Before flattening: ', x)
    print('After flattening: ', flatten(x))

#test_flatten()


def check_accuracy_part(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

def train_part(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            #print(y.shape)

            scores = model(x)
            #print(scores.shape)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy_part(loader_val, model)
                print()


'''
2-Layer-FullyConnected-Network
'''

hidden_layer_size = 4000
learning_rate = 1e-2

'''
first 2-Layer-FullyConnected-Network
'''

model = TwoLayerFC(3 * 32 * 32, hidden_layer_size, 10)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

train_part(model, optimizer)
check_accuracy_part(loader_test, model)
torch.save(model, 'Two_layer_FC_1.pkl')



'''
second 2-Layer-FullyConnected-Network
'''
model = TwoLayerFC(3 * 32 * 32, hidden_layer_size, 10)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

train_part(model, optimizer)
check_accuracy_part(loader_test, model)
torch.save(model, 'Two_layer_FC_2.pkl')




'''
3-Layer-ConvNet Training
'''

learning_rate = 3e-3
channel_1 = 32
channel_2 = 16

model = None
optimizer = None

'''
first 3-Layer-Conv-Net training
'''

model=ThreeLayerConvNet(3,channel_1,channel_2,10)
optimizer=optim.SGD(model.parameters(),lr=learning_rate)

train_part(model, optimizer)
check_accuracy_part(loader_test, model)
torch.save(model, 'three_layer_conv_1.pkl')



'''
second 3-Layer-Conv-Net training
'''

model=ThreeLayerConvNet(3,channel_1,channel_2,10)
optimizer=optim.SGD(model.parameters(),lr=learning_rate)

train_part(model, optimizer)
check_accuracy_part(loader_test, model)
torch.save(model, 'three_layer_conv_2.pkl')
print('ConvNet Trained !')



'''
first AlexNet Training
'''

model=AlexNet()
optimizer=optim.Adam(model.parameters())
train_part(model, optimizer, epochs=1)

check_accuracy_part(loader_test, model)
torch.save(model, 'AlexNet_1.pkl')


'''
second AlexNet Training
'''

model=AlexNet()
optimizer=optim.Adam(model.parameters())
train_part(model, optimizer, epochs=1)

check_accuracy_part(loader_test, model)
torch.save(model, 'AlexNet_2.pkl')
print('AlexNet Trained !')