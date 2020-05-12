# generate a model used for CIFAR-10 classification

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch import from_numpy
from torch.autograd import Variable
import numpy as np
import time
from models import TwoLayerFC, ThreeLayerConvNet, AlexNet, Net


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        pickle_dict = pickle.load(fo, encoding='latin1')
    return pickle_dict


''' prepare dataset '''
train_X = []
train_Y = []
test_X = []
test_Y = []
file_name = 'cifar-10-batches-py/data_batch_'
for x in range(1, 6):
    train_dict = unpickle(file_name + str(x))
    train_X.append(train_dict.get('data'))
    train_Y.append(train_dict.get('labels'))
test_dict = unpickle('cifar-10-batches-py/test_batch')
test_X.append(test_dict.get('data'))
test_Y.append(test_dict.get('labels'))
label_names = unpickle('cifar-10-batches-py/batches.meta').get('label_names')

train_X = np.array(train_X).reshape(50000, 3, 32, 32).astype(np.float32) / 255.0
train_Y = np.array(train_Y).reshape(50000)
test_X = np.array(test_X).reshape(10000, 3, 32, 32).astype(np.float32) / 255.0
test_Y = np.array(test_Y).reshape(10000)

trainset = Data.TensorDataset(from_numpy(train_X),
                              torch.tensor(train_Y, dtype=torch.long))
testset = Data.TensorDataset(from_numpy(test_X),
                             torch.tensor(test_Y, dtype=torch.long))

trainloader = Data.DataLoader(trainset, batch_size=100, shuffle=True)
testloader = Data.DataLoader(testset, batch_size=100, shuffle=False)

####################################
# show a image
# import cv2
# img = np.transpose(train_X[1], (1, 2, 0))
# cv2.imwrite('example.jpg', img)
# print(label_names[train_Y[1]])
###################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(0))


def train_process(net, optimizer, filename):
    criterion = nn.CrossEntropyLoss()
    net.to(device)
    net.train()
    running_loss = 0
    print('Training Start')
    for epoch in range(21):
        time_start = time.time()
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(
                device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % 50 == 49:
                print('%5d loss: %.3f' % (i + 1, running_loss / 50))
                running_loss = 0.0
        print('epoch %d cost %3f sec, epoch_loss = %.3f' %
              (epoch, time.time() - time_start, epoch_loss / 500))

    torch.save(net, 'models/' + filename + '.pkl')
    print('Finished Training')


def test_process(filename):
    print('Test Begin')
    correct = 0
    total = 0
    net = torch.load('models/' + filename + '.pkl')
    net.to(device)
    net.eval()
    with torch.no_grad():
        time_start = time.time()
        for data in testloader:
            image, label = data
            image, label = Variable(image).to(device), Variable(label).to(
                device)
            output = net(image)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    print('%s Accuracy : %.3f %% cost %3f sec' %
          (filename, 100.0 * correct / total, time.time() - time_start))


# 2-Layer-FullyConnected-Network

hidden_layer_size = 4000
learning_rate = 1e-2

# first 2-Layer-FullyConnected-Network

model = TwoLayerFC(3 * 32 * 32, hidden_layer_size, 10)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

train_process(model, optimizer, 'Two_layer_FC_1')
test_process('Two_layer_FC_1')

# second 2-Layer-FullyConnected-Network

model = TwoLayerFC(3 * 32 * 32, hidden_layer_size, 10)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

train_process(model, optimizer, 'Two_layer_FC_2')
test_process('Two_layer_FC_2')
print('FC Trained !')


# 3-Layer-ConvNet Training
learning_rate = 3e-3
channel_1 = 32
channel_2 = 16

# first 3-Layer-Conv-Net training

model = ThreeLayerConvNet(3, channel_1, channel_2, 10)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

train_process(model, optimizer, 'Three_layer_conv_1')
test_process('Three_layer_conv_1')

# second 3-Layer-Conv-Net training

model = ThreeLayerConvNet(3, channel_1, channel_2, 10)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

train_process(model, optimizer, 'Three_layer_conv_2')
test_process('Three_layer_conv_2')
print('ConvNet Trained !')


# AlexNet Training

# first AlexNet Training

model = AlexNet()
optimizer = optim.Adam(model.parameters())

train_process(model, optimizer, 'AlexNet_1')
test_process('AlexNet_1')

# second AlexNet Training

model = AlexNet()
optimizer = optim.Adam(model.parameters())

train_process(model, optimizer, 'AlexNet_2')
test_process('AlexNet_2')
print('AlexNet Trained !')
