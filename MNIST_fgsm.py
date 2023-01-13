#!/usr/bin/env python
# encoding: utf-8
"""
@author: hongwei zhang
@contact: zhanghwei@sjtu.edu.cn
@file: MNIST.py
@time: 2021/12/29 21:36
"""
import copy

# import pandas as pd
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import mnist

raw_dim = 28 * 28  # shape of the raw image
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
# for rate in range(50):
for rate in range(1):
    # compression_rate = (rate + 1) * 0.02
    compression_rate = min((rate + 10) * 0.1, 1)
    channel = int(compression_rate * raw_dim)

    lambda1 = 1 - compression_rate
    lambda2 = compression_rate

    class MLP(nn.Module):
        # coders
        def __init__(self):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(28 * 28, channel)
            self.fc2 = nn.Linear(channel, 28 * 28)

        def forward(self, x):
            # x = F.relu(self.fc1(x))
            # encoder
            x = self.fc1(x)

            # scale and quantize
            x = x.detach().cpu()
            x_max = torch.max(x)
            x_tmp = copy.deepcopy(torch.div(x, x_max))

            # quantize
            x_tmp = copy.deepcopy(torch.mul(x_tmp, 256))
            x_tmp = copy.deepcopy(x_tmp.clone().type(torch.int))
            x_tmp = copy.deepcopy(x_tmp.clone().type(torch.float32))
            x_tmp = copy.deepcopy(torch.div(x_tmp, 256))

            x = copy.deepcopy(torch.mul(x_tmp, x_max))

            # add noise
            x_np = x.detach().numpy()
            # x_np = x.numpy()
            out_square = np.square(x_np)
            aver = np.sum(out_square) / np.size(out_square)

            # snr = 3  # dB
            snr = 10  # dB
            aver_noise = aver / 10 ** (snr / 10)
            noise = np.random.random(size=x_np.shape) * np.sqrt(aver_noise)
            # noise = noise.to(device)

            x_np = x_np + noise
            x = torch.from_numpy(x_np)
            x = x.to(torch.float32)

            # decoder
            x=x.to(device)
            x = self.fc2(x)       
            return x


    class MLP_MNIST(nn.Module):
        # classifier
        def __init__(self):
            super(MLP_MNIST, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 500)
            self.fc2 = nn.Linear(500, 250)
            self.fc3 = nn.Linear(250, 125)
            # self.fc4 = nn.Linear(125, 10)
            self.fc4 = nn.Linear(125, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            return x


    mlp_encoder = MLP().to(device)
    mlp_mnist = MLP_MNIST().to(device)

    # load the MNIST classifier
    mlp_mnist.load_state_dict(torch.load('MLP_MNIST.pkl'))
    # load the MLP encoder
    mlp_encoder.load_state_dict(torch.load(('MLP_MNIST_encoder_combining_%f.pkl' % compression_rate)))


    # mlp_mnist_ini = copy.deepcopy(mlp_mnist)

    def data_transform(x):
        x = np.array(x, dtype='float32') / 255
        x = (x - 0.5) / 0.5
        # x = x / 0.5
        x = x.reshape((-1,))
        x = torch.from_numpy(x)
        return x


    def data_inv_transform(x):
        """
        :param x:
        :return:
        """
        recover_data = x * 0.5 + 0.5
        # recover_data = x * 0.5
        recover_data = recover_data * 255
        recover_data = recover_data.reshape((28, 28))
        recover_data = recover_data.detach().cpu().numpy()
        return recover_data

    def plot(out,i,prefix,epsilon):
        out=data_inv_transform(out)
        pil_img = Image.fromarray(np.uint8(out))
        pil_img.save("fgsm_attack/"+str(epsilon)+"_epsilon/"+prefix+"/mnist_"+prefix+"_%d_%f.jpg" % (i, compression_rate))

    def FGSM(img, label, epsilon, mlp_encoder, mlp_mnist,device,iterations = 10):
        mlp_encoder.eval()
        img = img.clone()     #取副本，不改动数据集
        img, label = img.to(device), label.to(device)
        img, label = Variable(img), Variable(label)
        # forward
        out = mlp_encoder(img)
        out_mnist = mlp_mnist(out)    
        _, init_pred = out_mnist.max(1)
        print("Origin class:",label.item())
        print("Prediction before attack:",init_pred.item())
        correct=0

        for t in range(iterations):
            img.requires_grad = True
            # forward
            out = mlp_encoder(img)
            out_mnist = mlp_mnist(out)    
            _, pred = out_mnist.max(1)
            # 计算是否分类错误，如果错误则攻击成功，停止迭代
            if pred.item() != label.item():
                correct=1
                break
            loss = criterion(out, label, img)
            mlp_encoder.zero_grad()
            mlp_mnist.zero_grad()
            loss.backward()
            data_grad = img.grad.data     #计算输入的导数
            sign_data_grad = data_grad.sign()  #符号函数
            img = img.detach()
            img += epsilon * sign_data_grad
            img = torch.clamp(img, 0, 1)

        print("Total attack times: %d"%t)
        print("Prediction after attack:",pred.item())
        return t,correct,init_pred,pred,img,out

    # load data
    trainset = mnist.MNIST('./dataset/mnist', train=True, transform=data_transform, download=True)
    testset = mnist.MNIST('./dataset/mnist', train=False, transform=data_transform, download=True)
    # use test data as attack dataset
    attackset = mnist.MNIST('./dataset/mnist_attack',train=False, transform=data_transform,download=True)
    train_data = DataLoader(trainset, batch_size=64, shuffle=True)
    test_data = DataLoader(testset, batch_size=128, shuffle=False)
    # attack dataloader - batch_size=1
    attack_data = DataLoader(attackset, batch_size=1, shuffle=False)

    print("total samples for attack:"+str(len(attack_data)))

    # loss function
    def criterion(x_in, y_in, raw_in):
        out_tmp1 = nn.CrossEntropyLoss()
        out_tmp2 = nn.MSELoss()
        z_in = mlp_mnist(x_in)
        mse_in = lambda2 * out_tmp2(x_in, raw_in)
        loss_channel = lambda1 * out_tmp1(z_in, y_in) + lambda2 * mse_in
        return loss_channel

    def attack(mlp_encoder,mlp_mnist,device,dataloader,epsilon):
        correct=0
        orig_examples=[]
        adv_examples=[]
        init_preds=[]
        final_preds=[]
        cnt=0
        attack_cnt=0
        recover_img=[]
        for data, target in dataloader:
            cnt+=1
            print("[Case] -",cnt)
            t,res,init_pred,final_pred,noise_img,out=FGSM(data,target,epsilon,mlp_encoder,mlp_mnist,device)
            attack_cnt+=t
            correct+=res
            plot(noise_img,cnt,"adversarial",epsilon)
            noise_img=data_inv_transform(noise_img)
            data=data_inv_transform(data)
            orig_examples.append(data)
            init_preds.append(init_pred.item())
            final_preds.append(final_pred.item())
            adv_examples.append(noise_img)
            recover_img.append(data_inv_transform(out))
            plot(out,cnt,"reverse",epsilon)
        
        final_acc=correct/float(len(dataloader))
        print("Epsilon: {}\tattack Accuracy = {}/{}={}".format(epsilon,correct,len(dataloader),final_acc))
        print("Average attack times: {}".format(attack_cnt))

        # saving adversarial images and other attack data
        np.savez("fgsm_attack/"+str(epsilon)+"_epsilon/data/adv_pred.npz",init_preds=np.array(init_preds),final_preds=np.array(final_preds))
        np.savez("fgsm_attack/"+str(epsilon)+"_epsilon/data/adv_example.npz",adv_examples=adv_examples)
        np.savez("fgsm_attack/"+str(epsilon)+"_epsilon/data/orig_example.npz",orig_examples=orig_examples)
        np.savez("fgsm_attack/"+str(epsilon)+"_epsilon/data/recover_example.npz",recover_examples=recover_img)

    epsilons = [0, .05, .1, .15, .2, .25, .3]
    epsilon = epsilons[0]
    # print(next(mlp_encoder.parameters()).is_cuda)
    print('Attacking Start')
    print('Under Compression Rate: ',compression_rate)
    print('FGSM: epsilon = ',epsilon)
    attack(mlp_encoder,mlp_mnist,device,attack_data,epsilon)
