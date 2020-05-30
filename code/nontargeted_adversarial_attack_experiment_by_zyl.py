import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import numpy as np
from models import TwoLayerFC, ThreeLayerConvNet, AlexNet
import matplotlib.pyplot as plt


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        pickle_dict = pickle.load(fo, encoding='latin1')
    return pickle_dict


# load test images
test_dict = unpickle('cifar-10-batches-py/test_batch')
test_X = test_dict.get('data')
test_Y = test_dict.get('labels')
label_names = unpickle('cifar-10-batches-py/batches.meta').get('label_names')
test_X = np.array(test_X).reshape(10000, 1, 3, 32, 32).astype(np.float32) / 255.0
test_Y = np.array(test_Y).reshape(10000)

# load models
filename = 'models/AlexNet_'
model_0 = torch.load(filename+'1.pkl')
model_1 = torch.load(filename+'2.pkl')
loss = nn.CrossEntropyLoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(0))

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')


def show_result(img):
    model_0.to(device).eval()
    model_1.to(device).eval()
    img = torch.from_numpy(img).to(device)
    res_0 = F.softmax(model_0(img), dim=1).data
    print('Res_0:')
    print(res_0)
    res_0 = res_0.max(1)
    res_1 = F.softmax(model_1(img), dim=1).data
    print('Res_1:')
    print(res_1)
    res_1 = res_1.max(1)
    img = img.cpu().numpy().reshape(3, 32, 32).transpose(1, 2, 0)
    '''
    plt.suptitle('images')
    plt.subplot(1, 2, 1), plt.title('model_0 '+classes[res_0[1].cpu().numpy()[0]]+' Prob='+str(res_0[0].cpu().numpy()[0])[0:5])
    plt.imshow(img), plt.axis('off')
    plt.subplot(1, 2, 2), plt.title('model_1 '+classes[res_1[1].cpu().numpy()[0]]+' Prob='+str(res_1[0].cpu().numpy()[0])[0:5])
    plt.imshow(img), plt.axis('off')
    plt.show()
    '''
    plt.title('image')
    plt.title('model_0 '+classes[res_0[1].cpu().numpy()[0]]+' Prob='+str(res_0[0].cpu().numpy()[0])[0:5])
    plt.imshow(img), plt.axis('off')
    plt.show()
    plt.title('image')
    plt.title('model_1 '+classes[res_1[1].cpu().numpy()[0]]+' Prob='+str(res_1[0].cpu().numpy()[0])[0:5])
    plt.imshow(img), plt.axis('off')
    plt.show()


def nontargeted_attack(img, steps, step_lr, eps, lower_bound=0.2, model=model_0):
    model.to(device).eval()
    img = torch.from_numpy(img).to(device)
    label = model(img).data.max(1)[1]
    x, y = Variable(img, requires_grad=True).to(device), Variable(label).to(device)
    result = x
    adv = torch.zeros_like(x)
    flag = 0
    for step in range(steps):
        zero_gradients(x)
        out = model(x)
        prob = F.softmax(out, dim=1).data.max(1)[0].cpu().numpy()[0]
        if prob <= lower_bound:
            print(step, end='  ')
            print(prob)
            flag = 1
            break
        y.data = out.data.max(1)[1]
        _loss = loss(out, y)
        _loss.backward()
        normed_grad = step_lr * torch.sign(x.grad.data)
        step_adv = x.data + normed_grad
        adv = step_adv - img
        adv = torch.clamp(adv, -eps, eps)
        result = img + adv
        result = torch.clamp(result, 0.0, 1.0)
        x.data = result
    return result.cpu(), adv.cpu(), flag


def generate_sample(model, model_1, outputstring):
    print('generate_sample')
    model.to(device).eval()
    model_1.to(device).eval()
    classified_data = []
    classified_gen_sample = []
    for i in range(10):
        classified_data.append([])
        classified_gen_sample.append([])
    for i in range(test_X.shape[0]):
        img = test_X[i]
        print(i)
        img_t = torch.from_numpy(img).to(device)
        out = model(img_t)
        out_1 = model_1(img_t)
        temp_lb = out.data.max(1)[1].cpu().numpy()[0]
        temp_lb_1 = out_1.data.max(1)[1].cpu().numpy()[0]
        if temp_lb != test_Y[i] or temp_lb_1 != test_Y[i]:
            print('WRONG!!!')
            continue
        att_result, att_delta, flag = nontargeted_attack(img, steps=1000, step_lr=0.001, eps=0.1, lower_bound=0.2, model=model)
        if flag == 0:
            # print('wrong \n')
            continue
        classified_data[test_Y[i]].append(test_X[i])
        classified_gen_sample[test_Y[i]].append(att_result.detach().numpy())
        # print('\n')
        if i % 100 == 99:
            print('------- %d/all' % (i))
    return classified_data, classified_gen_sample


def test_adversarial_sample(model, origin_data, adv_sample, outputstring):
    print('test_sample ' + outputstring)
    model.to(device).eval()
    prob_list_0 = []
    prob_list_1 = []
    acc_list_0 = []
    acc_list_1 = []
    for i in range(10):
        # print('%d/10' % (i))
        aver_prob_0 = 0.0
        aver_prob_1 = 0.0
        acc_0 = 0.0
        acc_1 = 0.0
        num = len(origin_data[i])
        for j in range(num):
            img_0 = torch.from_numpy(origin_data[i][j]).to(device)
            out_0 = model(img_0)
            label_0 = out_0.data.max(1)[1].cpu().numpy()[0]
            if label_0 == i:
                acc_0 += 1
            aver_prob_0 += F.softmax(out_0, dim=1).data.max(1)[0].cpu().numpy()[0]
            img_1 = torch.from_numpy(adv_sample[i][j]).to(device)
            out_1 = model(img_1)
            label_1 = out_1.data.max(1)[1].cpu().numpy()[0]
            if label_1 == i:
                acc_1 += 1
            aver_prob_1 += F.softmax(out_1, dim=1).data.max(1)[0].cpu().numpy()[0]
        aver_prob_0, aver_prob_1 = aver_prob_0 / num, aver_prob_1 / num
        acc_0, acc_1 = acc_0 / num, acc_1 / num
        prob_list_0.append(aver_prob_0)
        prob_list_1.append(aver_prob_1)
        acc_list_0.append(acc_0)
        acc_list_1.append(acc_1)
    print(outputstring+' original')
    print(prob_list_0)
    print(acc_list_0)
    print(outputstring+' ad_sample')
    print(prob_list_1)
    print(acc_list_1)


# '''
ori_data, sample = generate_sample(model_0, model_1, 'model_0')
np.save('tl_ori_picture.npy', np.array(ori_data))
np.save('tl_sample.npy', np.array(sample))
# '''
'''
ori_data = np.load('tl_ori_picture.npy', allow_pickle=True)
sample = np.load('tl_sample.npy', allow_pickle=True)
print(ori_data.shape)
print(sample.shape)
'''
test_adversarial_sample(model_0, ori_data, sample, 'model_0')
test_adversarial_sample(model_1, ori_data, sample, 'model_1')
