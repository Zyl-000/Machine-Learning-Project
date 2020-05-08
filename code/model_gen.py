# generate a model used for CIFAR-10 classification

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch import from_numpy
from torch.autograd import Variable
import numpy as np
import time


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        pickle_dict = pickle.load(fo, encoding='latin1')
    return pickle_dict


NUM = 5

''' prepare dataset '''
train_X = []
train_Y = []
test_X = []
test_Y = []
file_name = 'cifar-10-batches-py/data_batch_'
for x in range(1, 6):
    train_dict = unpickle(file_name+str(x))
    train_X.append(train_dict.get('data'))
    train_Y.append(train_dict.get('labels'))
test_dict = unpickle('cifar-10-batches-py/test_batch')
test_X.append(test_dict.get('data'))
test_Y.append(test_dict.get('labels'))
label_names = unpickle('cifar-10-batches-py/batches.meta').get('label_names')

train_X = np.array(train_X).reshape(50000, 3, 32, 32).astype(np.float32)
train_Y = np.array(train_Y).reshape(50000)
test_X = np.array(test_X).reshape(10000, 3, 32, 32).astype(np.float32)
test_Y = np.array(test_Y).reshape(10000)

trainset = Data.TensorDataset(from_numpy(train_X), torch.tensor(train_Y, dtype=torch.long))
testset = Data.TensorDataset(from_numpy(test_X), torch.tensor(test_Y, dtype=torch.long))

trainloader = Data.DataLoader(trainset, batch_size=100, shuffle=True)
testloader = Data.DataLoader(testset, batch_size=100, shuffle=False)

####################################
# show a image
# import cv2
# img = np.transpose(train_X[1], (1, 2, 0))
# cv2.imwrite('example.jpg', img)
# print(label_names[train_Y[1]])
###################################


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 128, 1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.conv8 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, 1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.conv11 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 1, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()

        self.fc14 = nn.Linear(512*4*4, 1024)
        self.fc15 = nn.Linear(1024, 1024)
        self.fc16 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        # print(" x shape ",x.size())
        x = x.view(-1, 512*4*4)
        x = F.relu(self.fc14(x))
        x = F.relu(self.fc15(x))
        x = self.fc16(x)

        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(0))


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
net.to(device)
# '''
net.train()
running_loss = 0
print('Training Start')
for epoch in range(21):
    time_start = time.time()
    epoch_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        epoch_loss += loss.item()
        if i % 50 == 49:
            print('%5d loss: %.3f' % (i + 1, running_loss / 50))
            running_loss = 0.0
    print('epoch %d cost %3f sec, epoch_loss = %.3f' % (epoch, time.time()-time_start, epoch_loss/500))

torch.save(net, 'models/new_net'+str(NUM)+'.pkl')
print('Finished Training')
# '''
# net = torch.load('new_net5.pkl')

print('Test Begin')
correct = 0
total = 0
net.eval()
with torch.no_grad():
    time_start = time.time()
    for data in testloader:
        image, label = data
        image, label = Variable(image).to(device), Variable(label).to(device)
        output = net(image)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
print('Accuracy : %.3f %% cost %3f sec' % (100.0 * correct / total, time.time()-time_start))
