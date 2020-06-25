# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 17:35:40 2019
@Description: This code perform neural
network training with Standard Conv carnel.
Dataset used here is CIFAR-10 and architecture 
used is VGG. 
@author: ksouvik

"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import numpy as np
import time
import math

from VGG import VGG

from config_VGG import args
#######################Hyper parameters################
btsize = args.batch_size
wtdecay = args.weight_decay
numEpochs = args.epoch
numWarmup = 5 #number of warmup epochs
density_factor = 0.4 #This parameter is opposite to sparsity factor, for an FC
					 # network its 1.0.
zero_wt_frac = 1.0 - density_factor 
if density_factor == 0.1:
	nZero = 1
elif density_factor == 0.2:
	nZero = 2
elif density_factor == 0.3:
    nZero = 3
elif density_factor == 0.4:
	nZero = 4
else:
	nZero = 9
print ("using pSC:", nZero)
periodicity = 16
best_acc = 0.0
#######################End Hyper parameters$=##########


device = 'cuda' if (not args.no_cuda and torch.cuda.is_available()) else 'cpu'
print('using', device)

print ('====> Preparing data <====')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, \
        transform=transform_train)

test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, \
        transform=transform_test)


train_loader = torch.utils.data.DataLoader(train, batch_size = btsize, shuffle = True)

test_loader = torch.utils.data.DataLoader(test, batch_size = btsize, shuffle = False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def get_n_params(model):
	pp = 0
	for p in list(model.parameters()):
		nn = 1
		for s in list(p.size()):
			nn = nn*s
		pp += nn
	return pp

##################################################
#The add_noBiasWeightDecay function makes sure that the weight decay isn't applied to the bias and BN terms
##################################################
def add_noBiasWeightDecay(model, skip_list):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    assert len(list(model.parameters())) == (len(decay) + len(no_decay))
    return [{'params': no_decay, 'weight_decay' : 0.0}, {'params': decay}]


PATH = 'VGG16_full_1.0_20_0.1.pth'
exists = os.path.isfile(PATH)
if exists:
	print ('==> Loadinf pretrained model...')
	net = torch.load(PATH)
	net = net.to(device)

else:
	print('==> Building model...')
	net = VGG(args.archi_type, init_weights = True)
	net = net.to(device)

print("Total trainable parameters: ", get_n_params(net))



####################################################
## Following part of the code does take care of the pre-defined sparse 
## initialization of the weights. We first initialize the weights as normal
## then make the initial values of the zero-weights as 0.
####################################################
alpha = {}
k=1


for layer in net.modules():
    if isinstance(layer, nn.Conv2d):
    
        tmp1 = list(layer.weight.size())
        tmp2 = np.ones(tmp1)
        for i in range(tmp1[0]):
            cnt_inCh = i
            for j in range (tmp1[1]):
                cnt_inCh = cnt_inCh + 1  # We have added the term i to make the orientation different for each CONV3D 
                if (k!=1 and (cnt_inCh % 16 == 1 or cnt_inCh % 16 == 5 or cnt_inCh % 16 == 9 or cnt_inCh % 16 == 13) and tmp1[2] != 1):
                    tmp3 = np.array([1] * 2 + [0] * 1 + [1] * 2 + [0] * 4)
                elif (k!=1 and (cnt_inCh % 16 == 2 or cnt_inCh % 16 == 6 or cnt_inCh % 16 == 10 or cnt_inCh % 16 == 14) and tmp1[2] != 1):
                    tmp3 = np.array([0] * 1 + [1] * 2 + [0] * 1 + [1] * 2 + [0] * 3)
                elif (k!=1 and (cnt_inCh % 16 == 3 or cnt_inCh % 16 == 7 or cnt_inCh % 16 == 11 or cnt_inCh % 16 == 15) and tmp1[2] != 1):
                    tmp3 = np.array([0] * 3 + [1] * 2 + [0] * 1 + [1] * 2 + [0] * 1)
                elif (k!=1 and (cnt_inCh % 16 == 4 or cnt_inCh % 16 == 8 or cnt_inCh % 16 == 12) and tmp1[2] != 1):
                    tmp3 = np.array([0] * 4 + [1] * 2 + [0] * 1 + [1] * 2)
                elif (k!=1 and (cnt_inCh % 16 == 0) and tmp1[2] != 1):
                    tmp3 = np.array([1] * 9)
                elif (tmp1[2] == 1):
                    tmp3 = np.array([1] * 1)
                else:
                    tmp3 = np.array([1] * 9)
                tmp2[i,j,:,:] = np.reshape(tmp3,(tmp1[2], tmp1[3]))
        alpha[k] = torch.tensor((tmp2), dtype = torch.float).to(device)
        k = k + 1
fltr_cnt = k

i = 1

with torch.no_grad():
    for layer in net.modules():
        if (isinstance(layer, nn.Conv2d) and (i < fltr_cnt)):
            layer.weight *= alpha[i]
            i = i + 1

####################################################

loss_func = nn.CrossEntropyLoss()

if args.no_bias_decay:
	print ('using no_bias decay')
	params = add_noBiasWeightDecay(net, ['bn'])
else:
	params = net.parameters()

if args.optim == 'adam':
	optimizer = torch.optim.Adam(params, lr = args.lr, weight_decay = wtdecay)
else:	
	optimizer = torch.optim.SGD(params, lr = args.lr, momentum = args.momentum, \
				weight_decay = wtdecay*5)

##################################################
# Currently supported schedulers are: 1. MultiStep and 2. ReduceOnPlateau
##################################################
if args.scheduler_policy == 'multistep':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, \
			milestones = [40,80, 100, 115, 130, 140], gamma = 0.2, \
			last_epoch = -1)
#default scheduler is ReduceLROnPlateau
else:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,\
            mode = 'min', factor = 0.5, threshold= 0.01, patience = 4)

training_acc = []
training_loss = []
test_accu = []

# calculate training set accuracy
#####################################

for epoch in range(numEpochs):
    epoch_training_loss = 0
    num_batches = 0
    start_time = time.time()
    if args.scheduler_policy == 'multistep':
    	scheduler.step() #uncomment this part if not using ReduceLROnPlateau scheduler
    for batch_num, training_batch in enumerate(train_loader):
        inputs, labels = training_batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        forward_output = net(inputs)
        loss = loss_func(forward_output, labels)
        loss.backward()

        ###############################################
        ## Following part of the code is introduced due to sparsity
        ## It converts the gradients of the pre-defined zero weights  
        ## to zero value.
        ###############################################
        i = 1
        with torch.no_grad():
            for layer in net.modules():
                if (isinstance(layer, nn.Conv2d) and (i < fltr_cnt)):
                    layer.weight.grad *= alpha[i]
                    i = i + 1
        ####################################################
                   
        optimizer.step()
        epoch_training_loss += loss.data.item()
        num_batches += 1
    #if args.scheduler_policy == 'plateau':
    #	scheduler.step(epoch_training_loss/num_batches) #comment this line if not using ReduceLROnPlateau scheduler
    print("epoch: ", epoch, "loss: ", epoch_training_loss/num_batches)
    training_loss.append(epoch_training_loss)

    correct = 0.0
    total = 0.0
    num_batches = 0
	
    for i, train_batch in enumerate(train_loader, 0):
    		inputs, labels = train_batch
    		inputs, labels = inputs.to(device), labels.to(device)
    		outputs = net(inputs)
    		_, predicted = outputs.max(1)
    
    		total += labels.size(0)
    		correct += predicted.eq(labels).sum().item()
    		num_batches += 1

    training_acc.append((correct/total) * 100)
    print("epoch: ", epoch, "Train_acc: ", (correct/total) * 100, " %")
    print("epoch: ", epoch, "current_lr: ", optimizer.param_groups[0]['lr'])

##################################    
# test the model on test dataset    
##################################

    test_acc = 0.0    
    correct = 0.0
    total = 0.0
    num_batches = 0
    epoch_test_loss = 0

    for i, test_batch in enumerate(test_loader, 0):
        inputs, labels = test_batch
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        test_loss = loss_func(outputs, labels)
        epoch_test_loss += test_loss.data.item()
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        num_batches += 1

    test_acc = correct/total
    test_accu.append((test_acc) * 100)
    print("epoch: ", epoch, "test_acc: ", (correct/total) * 100, " %", "test_loss: ", epoch_test_loss/num_batches)

    #When there is only training and test dataset split, it's not wise to do Plateau scheduler with test_loss
    if args.scheduler_policy == 'plateau':
    	scheduler.step(epoch_test_loss/num_batches) #comment this line if not using ReduceLROnPlateau scheduler

    if (test_acc > best_acc):
        best_acc = test_acc
        best_epoch = epoch
        then_lr = optimizer.param_groups[0]['lr']
        bestPATH = 'VGG16_pSC{}_{}_with3x3_P{}_cifar10_bestEpoch_{}_{}_model.pth'.format(nZero, args.scheduler_policy, periodicity, epoch, test_acc*100)
        torch.save(net, bestPATH)
 
    #if ((epoch + 1) % 10 == 0):
    #	newPATH = 'VGG16_pSC{0}_{1}_{2}_{3}.pth'.format(nZero, \
    #		epoch, args.scheduler_policy, optimizer.param_groups[0]['lr'] )
    #	torch.save(net, newPATH)


   

#with open('VGG16_pSC{0}_{1}_test_accur_{2}_{3}.txt'.format(nZero, args.scheduler_policy, wtdecay, epoch), 'w') as f:
#    for item in test_acc:
#        f.write("%s\n" % item)

###########For plotting graph later #############
with open('VGG16_pSC{}_{}_with3x3_P{}_cifar10_train_accur_{}_{}.txt'.format(nZero, args.scheduler_policy, periodicity, wtdecay, epoch), 'w') as f:
    for item in training_acc:
        f.write("%s\n" % item)

with open('VGG16_pSC{}_{}_with3x3_P{}_cifar10_test_accur_{}_{}.txt'.format(nZero, args.scheduler_policy, periodicity, wtdecay, epoch), 'w') as f:
    for item in test_accu:
        f.write("%s\n" % item)
        
with open('VGG16_pSC{}_{}_with3x3_P{}_cifar10_train_loss_{}_{}.txt'.format(nZero, args.scheduler_policy, periodicity, wtdecay, epoch), 'w') as f:
    for item in training_loss:
        f.write("%s\n" % item)






