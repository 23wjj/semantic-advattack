from matplotlib import pyplot as plt
import torch
import numpy as np
import torch
from torch import nn
from PIL import Image
import argparse


parser = argparse.ArgumentParser(description='fgsm', conflict_handler='resolve')
parser.add_argument('--epsilon', type=float, default=0.1, metavar='')

args = parser.parse_args()
epsilon=args.epsilon
input=np.load("fgsm_attack/"+str(epsilon)+"_epsilon/data/adv_pred.npz")
init_preds=input['init_preds'].tolist()
final_preds=input['final_preds'].tolist()
adv_examples=np.load("fgsm_attack/"+str(epsilon)+"_epsilon/data/adv_example.npz")['adv_examples']
orig_examples=np.load("fgsm_attack/"+str(epsilon)+"_epsilon/data/orig_example.npz")['orig_examples']
recover_img=np.load("fgsm_attack/"+str(epsilon)+"_epsilon/data/recover_example.npz")['recover_examples']

def class_aggr(epsilon):

    labels = ['0', '1', '2', '3', '4','5','6','7','8','9']
    target_number=[[],[],[],[],[],[],[],[],[],[]]

    for i in range(10):
        for j in range(10):
            cnt=0
            for k in range(len(init_preds)):
                if init_preds[k]==i and final_preds[k]==j:
                    cnt+=1
            target_number[i].append(cnt)
    print(target_number)

    x=[0,2,4,6,8,10,12,14,16,18]
    x=np.array(x)
    width = 0.15 # the width of the bars

    rects=[]
    fig, ax = plt.subplots()

    color=["bisque","darkorange","burlywood","y","olive","lightgreen","mediumaquamarine","forestgreen","lightcyan","c"]
    for i in range(10):
        rect=ax.bar(x-4.5*width+i*width,target_number[i],width,color=color[i],label='ground_truth: '+str(i))
        rects.append(rect)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Targets')
    ax.set_title('Target Adversarial Frequency under FGSM (epsilon='+str(epsilon)+')')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    fig.tight_layout()
    
    plt.savefig('./adv_target_'+str(epsilon)+'.png')

def plot_fig(epsilon):
    cnt = 0
    plt.figure(figsize=(20,6))
    examples=orig_examples[12:22]
    origin=init_preds[12:22]
    adver=final_preds[12:22]
    adv_examp=adv_examples[12:22]
    rec_examp=recover_img[12:22]
    ytitle=["original","adversarial","recovering"]
    for i in range(len(ytitle)):
        for j in range(len(examples)):
            print(str(i)+":"+str(j))
            cnt += 1
            plt.subplot(len(ytitle),len(examples),cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel(ytitle[i], fontsize=14)
            if i==0:
                ex=examples[j]
                orig=origin[j]
                adv=adver[j]
                plt.title("{} -> {}".format(orig, adv))
            elif i==1:
                ex=adv_examp[j]
            elif i==2:
                ex=rec_examp[j]
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.savefig("fgsm_res_"+str(epsilon)+".png")

plot_fig(epsilon)
class_aggr(epsilon)
