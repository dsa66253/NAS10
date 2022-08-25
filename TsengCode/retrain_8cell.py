from __future__ import print_function
import datetime
import math
import os
import sys
import time
import random
from pathlib import Path
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from PIL import ImageFile
from sklearn.model_selection import train_test_split
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Subset
from torchvision import transforms, datasets
from data.config import cfg_newnasmodel
from tensorboardX import SummaryWriter
import numpy as np
from models.newmodel_8cell import NewNasModel
from feature.learning_rate import adjust_learning_rate
from feature.normalize import normalize
from feature.resume_net import resumeNet
from feature.make_dir import makeDir
from feature.plot_image import plot_img
import matplotlib.pyplot as plt
from feature.split_data import split_data
from feature.random_seed import set_seed_cpu, set_seed_gpu
from PIL import ImageFile


def parse_args(k):
    parser = argparse.ArgumentParser(description='imagenet nas Training')
    parser.add_argument('--network', default='newnasmodel', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--resume_net', default=None, help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--save_folder', default='./weights_retrain_8cell/',
                        help='Location to save checkpoint models')
    parser.add_argument('--log_dir', default='./tensorboard_retrain_8cell/',
                        help='Location to save logging')
    parser.add_argument('--decode_folder', type=str, default='./weights_8cell',
                        help='put decode folder')
    parser.add_argument('--genotype_file', type=str, default='genotype_' + str(k) + 'npy',
                        help='put decode file')

    args = parser.parse_args()
    return args


def train(number, seed_img, seed_weight):
    global last_epoch_val_acc
    PATH_train = r"./dataset1/train"
    print('PATH:'+str(PATH_train))
    TRAIN = Path(PATH_train)

    train_transforms = normalize(seed_img, img_dim)

    # choose the training datasets
    all_data = datasets.ImageFolder(TRAIN, transform=train_transforms)
    train_data, val_data = split_data(all_data, 0.2)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,
                                               shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=num_workers,
                                             shuffle=False)
    print('-------first image------------')
    print(train_data[0][0])

    # 確認training dataset的大小
    for batch_x, batch_y in train_loader:
        print((batch_x.shape, batch_y.shape))
        break

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    if os.path.isdir(args.decode_folder):
        genotype_filename = os.path.join(args.decode_folder, args.genotype_file)
        cell_arch = np.load(genotype_filename)
        print(cell_arch)
        print('Load best alpha for each cells from %s' % (genotype_filename))
    else:
        print('Decode path is not exist!')
        sys.exit(0)

    if cfg['name'] == 'NewNasModel':
        set_seed_cpu(seed_weight)
        net = NewNasModel(num_classes=num_classes, cell_arch=cell_arch, num_cells=num_cell)
        print("Printing net...")
        print(net)

    print('--------check first image-------')
    print(train_data[0][0])
    resumeNet(args.resume_net, net)

    if num_gpu > 1 and gpu_train:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = net.cuda()

    cudnn.benchmark = True

    print("Training with learning rate = %f, momentum = %f, lambda = %f " % (initial_lr, momentum, weight_decay))

    writer = SummaryWriter(log_dir=args.log_dir,
                           comment="LR_%.3f_BATCH_%d".format(initial_lr, batch_size))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum,
                          weight_decay=weight_decay)  # 是否采取 weight_decay

    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')
    # 加載數據集
    epoch_size = math.ceil(len(train_data) / batch_size)
    max_iter = max_epoch * epoch_size
    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0
    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    print('Start to train...')
    i = 0
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            print('hi')
            # create batch iterator
            train_batch_iterator = iter(train_loader)
            '''
            if (epoch % 20 == 0 and epoch > 0) or (epoch % 50 == 0 and epoch > cfg['decay1']):
                torch.save(
                    net.state_dict(),
                    os.path.join(
                        save_folder,
                        cfg['name'] + '_' + str(number) + '_epoch_' + str(epoch) + '.pth'
                    ),
                )
            '''
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1

        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load test data
        train_images, train_labels = next(train_batch_iterator)

        val_batch_iterator = iter(val_loader)
        val_images, val_labels = next(val_batch_iterator)

        # plot_img(train_images, train_labels, val_images, val_labels)

        record_train_loss = []
        record_val_loss = []

        train_images = train_images.cuda()
        train_labels = train_labels.cuda()
        val_images = val_images.cuda()
        val_labels = val_labels.cuda()

        optimizer.zero_grad()
        # 正向傳播
        train_outputs = net(train_images)
        # 計算loss
        train_loss = criterion(train_outputs, train_labels)
        # 反向傳播
        train_loss.backward()

        optimizer.step()

        _, predicts = torch.max(train_outputs.data, 1)

        record_train_loss.append(train_loss.item())
        val_outputs = net(val_images)
        _, predicts_val = torch.max(val_outputs.data, 1)
        val_loss = criterion(val_outputs, val_labels)
        record_val_loss.append(val_loss.item())

        # 計算訓練集準確度
        total_images = 0
        correct_images = 0
        total_images += train_labels.size(0)
        correct_images += (predicts == train_labels).sum()
        # 計算驗證集準確度
        total_images_val = 0
        correct_images_val = 0
        total_images_val += val_labels.size(0)
        correct_images_val += (predicts_val == val_labels).sum()

        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        print(
            'Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || train_loss: {:.4f} || val_loss: {:.4f} || train_Accuracy: {:.4f} || val_Accuracy: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || '
            'ETA: {} '
                .format(epoch, max_epoch, (iteration % epoch_size) + 1,
                        epoch_size, iteration + 1, max_iter, train_loss.item(), val_loss.item(),
                        100 * correct_images.item() / total_images,
                        100 * correct_images_val.item() / total_images_val,
                        lr, batch_time, str(datetime.timedelta(seconds=eta))))

        writer.add_scalar('Train_Loss', train_loss.item(), iteration + 1)
        writer.add_scalar('Val_Loss', val_loss.item(), iteration + 1)
        writer.add_scalar('train_Acc', 100 * correct_images / total_images, iteration + 1)
        writer.add_scalar('val_Acc', 100 * correct_images_val / total_images_val, iteration + 1)
        last_epoch_val_acc = 100 * correct_images_val / total_images_val

    torch.save(net.state_dict(), os.path.join(save_folder, cfg['name'] + str(number) + '_Final.pth'))

    return last_epoch_val_acc

if __name__ == '__main__':
    val_1, val_2, val_3 = 0, 0, 0

    for k in range(3):
        if k == 0:
            image = 10
            weight = 20
        elif k == 1:
            image = 255
            weight = 278
        else:
            image = 830
            weight = 953
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        args = parse_args(k)
        makeDir(args.save_folder, args.log_dir)

        cfg = None
        if args.network == "newnasmodel":
            cfg = cfg_newnasmodel
        else:
            print('Retrain Model %s doesn\'t exist!' % (args.network))
            sys.exit(0)

        num_classes = 10
        num_cell = 8
        img_dim = cfg['image_size']
        num_gpu = cfg['ngpu']
        batch_size = cfg['batch_size']
        max_epoch = cfg['epoch']
        gpu_train = cfg['gpu_train']

        num_workers = args.num_workers
        momentum = args.momentum
        weight_decay = args.weight_decay
        initial_lr = args.lr
        gamma = args.gamma
        save_folder = args.save_folder

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        seed_img = image
        seed_weight = weight
        set_seed_cpu(seed_img)
        last_epoch_val_acc = train(k, seed_img, seed_weight)

        if k == 0:
            val_1 = last_epoch_val_acc
        elif k == 1:
            val_2 = last_epoch_val_acc
        else:
            val_3 = last_epoch_val_acc

    print('result:', str(val_1), str(val_2), str(val_3))
