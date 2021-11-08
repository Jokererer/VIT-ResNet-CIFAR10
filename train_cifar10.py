# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from da import CutMix, MixUp
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time

from models import *
from models.vit import ViT
from utils import progress_bar
from models.convmixer import ConvMixer
from randomaug import RandAugment
from torch.optim import lr_scheduler
from cross_entropy import LabelSmoothingCrossEntropy
from schedular import WarmupCosineSchedule
from torch.optim.lr_scheduler import _LRScheduler

torch.cuda.empty_cache()

#learning_rate = 3e-2                                # The initial learning rate for SGD
#learning_rate = 1e-4                                # The initial learning rate for Adam
# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')  # resnets.. 1e-3, Vit..1e-4?
parser.add_argument('--opt', default="sgd")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--aug', action='store_true', help='use randomaug')
parser.add_argument('--amp', action='store_true', help='enable AMP training')
#parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', default='64')
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument("--img_size", default=224, type=int, help="Resolution size")
parser.add_argument('--patch', default='32', type=int)
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--convkernel', default='8', type=int)
# parser.add_argument("--cutmix", action="store_true")
# parser.add_argument("--mixup", action="store_true")
parser.add_argument('--cos', action='store_true', help='Train with cosine annealing scheduling')

args = parser.parse_args()



# take in args
import wandb

watermark = "{}_lr{}".format(args.net, args.lr)
if args.amp:
    watermark += "_useamp"

wandb.init(project="Vit-CIFAR10-224-PATCH",
           name=watermark)
wandb.config.update(args)

# if args.aug:
#     import albumentations
bs = int(args.bs)

use_amp = args.amp

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if args.net == "vit_timm":
    size = 384
else:
    size = 32
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.Resize(size),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
#
# transform_test = transforms.Compose([
#     transforms.Resize(size),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
#transforms.ToTensor(),
#transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
transform_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomVerticalFlip(p = 0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    	                                  std=[0.229, 0.224, 0.225])
                                     ]
)

transform_test = transforms.Compose([ transforms.Resize(256),
            transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
                                   ]
)
# transform_train = transforms.Compose([
#         transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#     ])
# transform_test = transforms.Compose([
#         transforms.Resize((args.img_size, args.img_size)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#     ])

# Add RandAugment with N, M(hyperparameter)
if args.aug:
    N = 2;
    M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=8)

# trainset = torchvision.datasets.ImageNet('./data/ImageNet/train', transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)
#
# testset = torchvision.datasets.ImageNet('./data/ImageNet/val',transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
if args.net == 'res18':
    net = ResNet18()
elif args.net == 'vgg':
    net = VGG('VGG19')
elif args.net == 'res34':
    net = ResNet34()
elif args.net == 'res50':
    net = ResNet50()
elif args.net == 'res101':
    net = ResNet101()
elif args.net == "convmixer":
    # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
    net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=10)

elif args.net == "vit":
    # ViT for cifar10
    net = ViT(
        image_size=224,
        patch_size=args.patch,
        num_classes=10,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1

    )
elif args.net == "vit_timm":
    import timm

    net = timm.create_model("vit_large_patch16_384", pretrained=True)
    net.head = nn.Linear(net.head.in_features, 10)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)  # make parallel
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


# Loss is CE
criterion = nn.CrossEntropyLoss()
# criterion = LabelSmoothingCrossEntropy()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=1e-4)
    # optimizer = optim.SGD(net.parameters(), lr=args.lr)
    #optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


# class WarmUpLR(_LRScheduler):
#     """warmup_training learning rate scheduler
#     Args:
#         optimizer: optimzier(e.g. SGD)
#         total_iters: totoal_iters of warmup phase
#     """
#
#     def __init__(self, optimizer, total_iters, last_epoch=-1):
#         self.total_iters = total_iters
#         super().__init__(optimizer, last_epoch)
#
#     def get_lr(self):
#         """we will use the first m batches, and set the learning
#         rate to base_lr * m / total_iters
#         """
#         return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

# use cosine or reduce LR on Plateau scheduling
if args.cos:
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=100, t_total=10000)
else:
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-3 * 1e-5,
                                               factor=0.1)

# warmup_epoch = 5
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100 - warmup_epoch)
#
# iter_per_epoch = len(trainset)
# warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warmup_epoch)


if args.cos:
    wandb.config.scheduler = "cosine"
else:
    wandb.config.scheduler = "ReduceLROnPlateau"
#wandb.config.scheduler = "cosine"

##### Training
# scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
gradient_accumulation_steps = 1
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

use_cuda = torch.cuda.is_available()
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_step = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # if epoch < 5:
        #     warmup_scheduler.step()
        #     warm_lr = warmup_scheduler.get_lr()
        #     print("warm_lr:%s" % warm_lr)
        inputs, targets = inputs.to(device), targets.to(device)

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                       args.alpha, use_cuda)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))

        outputs = net(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        #loss = criterion(outputs, targets)
        # if gradient_accumulation_steps > 1:
        #     loss = loss / gradient_accumulation_steps
        # Train with amp
        loss.backward()
        # with torch.cuda.amp.autocast(enabled=use_amp):
    # if (batch_idx + 1) % gradient_accumulation_steps == 0:
        scheduler.step()
        optimizer.step()
        optimizer.zero_grad()

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        # train_loss += loss.item()
        # _, predicted = outputs.max(1)
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()
        # train_step = train_step + 1

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a.data).sum().item()
                    + (1 - lam) * predicted.eq(targets_b.data).sum().item())
        train_step = train_step + 1

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return train_loss / (batch_idx + 1), 100. * correct / total, train_step


##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # # # # Update scheduler
    # if not args.cos:
    #  scheduler.step(test_loss)

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {"model": net.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scaler": scheduler.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + args.net + '-{}2-ckpt.pth'.format(args.patch))
        best_acc = acc

    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss / (batch_idx + 1), acc

def main():
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    # 记录总训练次数
    steps = 0
    wandb.watch(net)
    start_time = time.time()
    # learn_rate = 0.
    for epoch in range(start_epoch, args.n_epochs):
        start = time.time()
        # if epoch >= warmup_epoch:
        #     scheduler.step()
        #     learn_rate = scheduler.get_lr()[0]
        train_loss, train_acc, train_step = train(epoch)
        steps = steps + train_step
        val_loss, acc = test(epoch)

        if args.cos:
            scheduler.step(epoch - 1)

        train_accs.append(train_acc)
        train_losses.append(train_loss)

        test_accs.append(acc)
        test_losses.append(val_loss)


         #optimizer.param_groups[0]["lr"]
        # Log training..
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'train_acc ': train_acc, 'val_loss': val_loss, "val_acc": acc,
                   "lr": optimizer.param_groups[0]["lr"],
                   "epoch_time": time.time() - start, "steps": steps})
        print(
            f"Epoch : {epoch} - train_acc: {train_acc:.4f} - train_loss : {train_loss:.4f} -test acc: {acc:.4f} - test loss : {val_loss:.4f} -steps:{steps}\n")

        # Write out csv..
        with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(train_losses)
            writer.writerow(train_accs)
            writer.writerow(test_losses)
            writer.writerow(test_accs)
    print('ALL  Time', time.time() - start_time)
    print('ALL steps:', steps)
    print(train_accs)
    print(test_accs)
    print(train_losses)
    print(test_losses)

    # writeout wandb
    wandb.save("wandb_{}.h5".format(args.net))
if __name__ == '__main__':
    main()
