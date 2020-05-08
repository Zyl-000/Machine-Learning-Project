import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import time


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        pickle_dict = pickle.load(fo, encoding='latin1')
    return pickle_dict


filename = 'models/new_net'
img_number = 0
model_number = 5

''' load a image to attack '''
test_dict = unpickle('cifar-10-batches-py/test_batch')
test_X = test_dict.get('data')
test_Y = test_dict.get('labels')
label_names = unpickle('cifar-10-batches-py/batches.meta').get('label_names')
test_X = np.array(test_X).reshape(10000, 1, 3, 32, 32).astype(np.float32)
test_Y = np.array(test_Y).reshape(10000)

''' load model '''


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


def get_img():
    available_sample_set = set()
    correct_set = set()
    print('Begin Computing')
    for i in range(1, model_number+1):
        net = torch.load(filename+str(i)+'.pkl').to(device)
        net.eval()
        correct_set.clear()
        start_time = time.time()
        for j in range(0, 10000):
            img = torch.from_numpy(test_X[j])
            img = Variable(img).to(device)
            # print(img.size())
            cls_res = net(img).data.max(1)[1].cpu().numpy()[0]
            # print(cls_res)
            if cls_res == test_Y[j]:
                correct_set.add(j)
        if i == 1:
            available_sample_set = correct_set
        else:
            available_sample_set = available_sample_set & correct_set
        print('model %d correct number %d time %3f sec' % (i, len(correct_set), time.time()-start_time))
    return available_sample_set


ava_sample = get_img()
print(len(ava_sample))
sample_array = np.array(ava_sample)
np.save('avai_testsample.npy', sample_array)
