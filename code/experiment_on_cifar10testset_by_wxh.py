import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch import from_numpy
from torch.autograd import Variable
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
from models import TwoLayerFC, ThreeLayerConvNet, AlexNet, Net

USE_GPU = True

print(torch.cuda.is_available())

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
    print('cuda device count: %d' % (torch.cuda.device_count()))
    print('device name: %s' % (torch.cuda.get_device_name(0)))

else:
    device = torch.device('cpu')
print('using device:', device)



def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        pickle_dict = pickle.load(fo, encoding='latin1')
    return pickle_dict


'''
prepare dataset
'''
test_X = []
test_Y = []
file_name = 'cifar-10-batches-py/data_batch_'
test_dict = unpickle('cifar-10-batches-py/test_batch')
test_X.append(test_dict.get('data'))
test_Y.append(test_dict.get('labels'))
label_names = unpickle('cifar-10-batches-py/batches.meta').get('label_names')

test_X = np.array(test_X).reshape(10000, 3, 32, 32).astype(np.float32) / 255.0
test_Y = np.array(test_Y).reshape(10000)

testset = Data.TensorDataset(from_numpy(test_X),
                             torch.tensor(test_Y, dtype=torch.long))

testloader = Data.DataLoader(testset, batch_size=100, shuffle=False)

def test_process(filename):
    print('Test Begin')
    correct = 0
    total = 0
    net = torch.load('models/' + filename + '.pkl', map_location=device)
    net.to(device)
    net.eval()
    with torch.no_grad():
        time_start = time.time()
        for data in testloader:
            image, label = data
            image, label = Variable(image).to(device), Variable(label).to(
                device)
            output = net(image)
            _, predicted = output.max(1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    print('%s Accuracy : %.3f %% cost %3f sec' %
          (filename, 100.0 * correct / total, time.time() - time_start))

def test_compare(filename, ini1, ini2):
    print('Test Begin: %s' % (filename))
    sta1 = np.zeros(10, dtype='int32')
    sta2 = np.zeros(10, dtype='int32')
    correct1 = 0
    correct2 = 0
    total = 0
    pred_same = 0
    pred_right = 0
    net1 = torch.load('models/' + filename + '_1_' + ini1 + '.pkl', map_location=device)
    net2 = torch.load('models/' + filename + '_2_' + ini2 + '.pkl', map_location=device)
    net1.to(device)
    net2.to(device)
    net1.eval()
    net2.eval()
    with torch.no_grad():
        time_start = time.time()
        for idx, data in enumerate(testloader):
            image, label = data
            image, label = Variable(image).to(device), Variable(label).to(
                device)
            output1 = net1(image)
            output2 = net2(image)
            _1, pred1 = output1.max(1)
            _2, pred2 = output2.max(1)
            total += label.size(0)
            correct1 += (pred1 == label).sum().item()
            correct2 += (pred2 == label).sum().item()
            pred_same += (pred1 == pred2).sum()
            pred_right += ((pred1 == label) == (pred2 == label)).sum()


            for i in range(10):
                temp = (pred1 == label).int()
                temp[temp == 0] = 11
                temp[temp == 1] = i
                sta1[i] += (temp == pred1).sum()
                temp = (pred2 == label).int()
                temp[temp == 0] = 11
                temp[temp == 1] = i
                sta2[i] += (temp == pred2).sum()


    print('%s_1 Accuracy : %.3f %% cost %3f sec' %
          (filename, 100.0 * correct1 / total, time.time() - time_start))
    print('%s_2 Accuracy : %.3f %% cost %3f sec' %
          (filename, 100.0 * correct2 / total, time.time() - time_start))
    print('Same Prediction Percentage : %.3f %%' % (100.0 * pred_same / total))
    print('Same Correctness Percentage : %.3f %%' % (100.0 * pred_right / total))
    print('%s_1 Correct Prediction Number for each catagory' % (filename))
    print(sta1)
    print(sta1.sum())
    print('%s_2 Correct Prediction Number for each catagory' % (filename))
    print(sta2)
    print(sta2.sum())

    n_groups = 10
    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index, sta1, bar_width,
                    alpha=opacity, color='b',
                    tick_label=True,
                    label='%s_1_%s' % (filename, ini1))

    rects2 = ax.bar(index + bar_width, sta2, bar_width,
                    alpha=opacity, color='r',
                    tick_label=True,
                    label='%s_2_%s' % (filename, ini2))

    for rect in rects1+rects2:
        h=rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2, h, '%d' % int(h), ha='center', va='bottom')

    ax.set_xlabel('Catagory')
    ax.set_ylabel('Correct numbers')
    ax.set_title('Comparison on %s' % (filename))
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'))
    ax.legend()

    fig.tight_layout()
    plt.ylim(0, 1000)
    plt.show()

def cal_diff(filename, ini1, ini2):
    print('Test Begin: %s %s %s' % (filename, ini1, ini2))
    sta1 = np.zeros(10, dtype='int32')
    sta2 = np.zeros(10, dtype='int32')
    correct1 = 0
    correct2 = 0
    total = 0
    pred_same = 0
    pred_right = 0
    net1 = torch.load('models/' + filename + ini1 + '.pkl', map_location=device)
    net2 = torch.load('models/' + filename + ini2 + '.pkl', map_location=device)
    net1.to(device)
    net2.to(device)
    net1.eval()
    net2.eval()
    with torch.no_grad():
        time_start = time.time()
        for idx, data in enumerate(testloader):
            image, label = data
            image, label = Variable(image).to(device), Variable(label).to(
                device)
            output1 = net1(image)
            output2 = net2(image)
            _1, pred1 = output1.max(1)
            _2, pred2 = output2.max(1)
            total += label.size(0)
            correct1 += (pred1 == label).sum().item()
            correct2 += (pred2 == label).sum().item()
            pred_same += (pred1 == pred2).sum()
            pred_right += ((pred1 == label) == (pred2 == label)).sum()

            for i in range(10):
                temp = (pred1 == label).int()
                temp[temp == 0] = 11
                temp[temp == 1] = i
                sta1[i] += (temp == pred1).sum()
                temp = (pred2 == label).int()
                temp[temp == 0] = 11
                temp[temp == 1] = i
                sta2[i] += (temp == pred2).sum()

    alpha_1 = 0.5
    alpha_2 = 1.0
    p_diff_pred = pred_same / total
    diff = alpha_1 * p_diff_pred
    for i in range(10):
        diff += alpha_1 * 1.0 * abs(sta1[i] - sta2[i]) / min(sta1[i], sta2[i])

    print('%s%s Correct Prediction Number for each catagory' % (filename, ini1))
    print(sta1)
    print('%s%s Correct Prediction Number for each catagory' % (filename, ini2))
    print(sta2)

    print('Diff (%s%s, %s%s) = %f' % (filename, ini1, filename, ini2, diff))




test_compare('Two_layer_FC', 'kaiming_normal', 'kaiming_normal')
test_compare('Three_layer_conv', 'kaiming_normal', 'kaiming_normal')
test_compare('AlexNet', 'kaiming_normal', 'kaiming_normal')

#test_compare('Two_layer_FC', 'kaiming_uniform', 'kaiming_uniform')
#test_compare('Three_layer_conv', 'kaiming_uniform', 'kaiming_uniform')
#test_compare('AlexNet', 'xavier_normal', 'xavier_normal')

#cal_diff('Two_layer_FC', '_1_kaiming_normal', '_2_kaiming_normal')
#cal_diff('Three_layer_conv', '_1_kaiming_normal', '_2_kaiming_normal')
#cal_diff('AlexNet', '_1_kaiming_normal', '_2_kaiming_normal')

#cal_diff('Two_layer_FC', '_1_kaiming_uniform', '_2_kaiming_uniform')
#cal_diff('Three_layer_conv', '_1_kaiming_uniform', '_2_kaiming_uniform')
#cal_diff('AlexNet', '_1_xavier_normal', '_2_xavier_normal')

#cal_diff('Two_layer_FC', '_1_kaiming_normal', '_1_kaiming_uniform')
#cal_diff('Three_layer_conv', '_1_kaiming_normal', '_1_kaiming_uniform')
#cal_diff('AlexNet', '_1_kaiming_normal', '_1_xavier_normal')

#cal_diff('Two_layer_FC', '_2_kaiming_normal', '_2_kaiming_uniform')
#cal_diff('Three_layer_conv', '_2_kaiming_normal', '_2_kaiming_uniform')
#cal_diff('AlexNet', '_2_kaiming_normal', '_2_xavier_normal')

#cal_diff('Two_layer_FC', '_1_kaiming_normal', '_2_kaiming_uniform')
#cal_diff('Three_layer_conv', '_1_kaiming_normal', '_2_kaiming_uniform')
#cal_diff('AlexNet', '_1_kaiming_normal', '_2_xavier_normal')

#cal_diff('Two_layer_FC', '_2_kaiming_normal', '_1_kaiming_uniform')
#cal_diff('Three_layer_conv', '_2_kaiming_normal', '_1_kaiming_uniform')
#cal_diff('AlexNet', '_2_kaiming_normal', '_1_xavier_normal')