'''
MIT License

Copyright (c) 2018 Erik Linder-Norén

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Modifier: Guangyu Shen
Last modified: 11/30/2018

This module contains following components

    1.Receive parameters from user on command line
    2.Build cycle_GAN network instances 
    3.Train network 
    4.Sample images in certain batches and save them in dict.
    5.Print training loss for each epoch
    6.Save modal parameters 
    
'''


import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from visdom import Visdom
from model_v27 import *
from dataset import *
from utils import *
from dataset import ImageDataset
import torch.nn as nn
import torch.nn.functional as F
import torch
from dice_loss import dice_loss

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=53, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="2d_music_data", help='name of the dataset')
parser.add_argument('--dataset_path', type = str, default ='',help = 'path of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=25, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=128, help='size of image height')
parser.add_argument('--img_width', type=int, default=100, help='size of image width')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=100, help='interval between sampling images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=25, help='interval between saving model checkpoints')
parser.add_argument('--figure_interval', type=int, default=1, help='interval between showing figure')
parser.add_argument('--val_batch_size',type=int, default = 5, help = 'batch size for validation')
parser.add_argument('--n_residual_blocks', type=int, default=6, help='number of residual blocks in generator')
opt = parser.parse_args()
print(opt)

#vis = Visdom(env = 'CYCLEGAN')

# Create sample and checkpoint directories
if os.path.exists('images') == False:
    os.makedirs('images/%s' % opt.dataset_name)
if os.path.exists('saved_models') == False:
    os.makedirs('saved_models/%s' % opt.dataset_name)

# Losses
criterion_GAN = torch.nn.MSELoss()
#criterion_cycle = torch.nn.L1Loss()
#criterion_identity = torch.nn.L1Loss()


cuda = True if torch.cuda.is_available() else False
print(cuda)
# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2**4, opt.img_width // 2**4)


# Initialize generator and discriminator
G_AB = GeneratorResNet(res_blocks=opt.n_residual_blocks)
G_BA = GeneratorResNet(res_blocks=opt.n_residual_blocks)
D_A = Discriminator()
D_B = Discriminator()

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    #criterion_cycle.cuda()
    #criterion_identity.cuda()



if opt.epoch != 0:
    # Load pretrained models
    G_AB.load_state_dict(torch.load('saved_models/%s/G_AB_%d.pth' % (opt.dataset_name, opt.epoch)))
    G_BA.load_state_dict(torch.load('saved_models/%s/G_BA_%d.pth' % (opt.dataset_name, opt.epoch)))
    D_A.load_state_dict(torch.load('saved_models/%s/D_A_%d.pth' % (opt.dataset_name, opt.epoch)))
    D_B.load_state_dict(torch.load('saved_models/%s/D_B_%d.pth' % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

# Loss weights
lambda_cyc = 10
lambda_id =0.5 * lambda_cyc

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()),
                               lr=opt.lr, betas=(opt.b1, opt.b2))
#optimizer_G_AB = torch.optim.Adam(G_AB.parameters(),lr=opt.lr, betas=(opt.b1, opt.b2))
#optimizer_G_BA = torch.optim.Adam(G_BA.parameters(),lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
#lr_scheduler_G_AB = torch.optim.lr_scheduler.LambdaLR(optimizer_G_AB, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
#lr_scheduler_G_BA = torch.optim.lr_scheduler.LambdaLR(optimizer_G_BA, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B= torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

#weights_ce = torch.ones(128).type(Tensor)

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
'''
transforms_ = [ #transforms.Resize(int(opt.img_height), Image.BICUBIC),
                #transforms.RandomCrop((opt.img_height, opt.img_width)),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
                ]
'''
transforms_ = [ #transforms.Resize(int(opt.img_height*1.12), Image.BICUBIC),
                #transforms.RandomCrop((opt.img_height, opt.img_width)),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
                ]


# Training data loader
dataloader = DataLoader(ImageDataset(opt.dataset_path, transforms_=transforms_, unaligned=True),
                                batch_size=opt.batch_size, shuffle=True, num_workers=0)

# Test data loader
val_dataloader = DataLoader(ImageDataset(opt.dataset_path, transforms_=transforms_, unaligned=True),
                        batch_size=opt.val_batch_size, shuffle=True, num_workers=0)

leng = len(dataloader)-1
print(leng)
def to_binary(tensor):
    tensor[tensor > 0.5] = 127
    tensor[tensor <= 0.5] =0
    return tensor

def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs['A'].type(Tensor)) 
    fake_B = G_AB(real_A)
    recov_A = G_BA(fake_B)
    real_B = Variable(imgs['B'].type(Tensor))
    fake_A = G_BA(real_B)
    recov_B = G_AB(fake_A)
    fake_B = to_binary(fake_B)
    recov_A = to_binary(recov_A)
    fake_A = to_binary(fake_A)
    recov_B = to_binary(recov_B) 
    img_sample = torch.cat((real_A.data, fake_B.data, recov_A.data,
                            real_B.data, fake_A.data, recov_B.data), 0)
    save_image(img_sample, 'images/%s/%s.png' % (opt.dataset_name, batches_done), nrow=5, normalize=True)
    #vis.images(img_sample,opts = dict(title = 'images/%s/%s.png' % (opt.dataset_name, batches_done),nrow = 5))
# ----------
#  Training
# ----------

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    avg_G_loss = 0
    avg_D_loss = 0  
    avg_adv_loss = 0
    avg_idt_loss = 0
    avg_cycle_loss = 0
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(batch['A'].type(Tensor))/127
        real_B = Variable(batch['B'].type(Tensor))/127
        #real_A_long = real_A.type(torch.LongTensor)
        #real_B_long = real_B.type(torch.LongTensor)
        #real_A_one_hot = torch.zeros(opt.batch_size,128,opt.img_height,opt.img_width).scatter_(1,real_A_long,1)
        #real_B_one_hot = torch.zeros(opt.batch_size,128,opt.img_height,opt.img_width).scatter_(1,real_B_long,1)
        #real_A_long = real_A.type(LongTensor)
        #real_B_long = real_B.type(LongTensor)
        #real_A_one_hot = Variable(real_A_one_hot.type(Tensor))
        #real_B_one_hot = Variable(real_B_one_hot.type(Tensor))
    
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), patch[0],patch[1],patch[2]))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), patch[0],patch[1],patch[2]))), requires_grad=False)

        # ------------------
        #  Train Generators : This step is only used to train the G !
        # ------------------

        optimizer_G.zero_grad()
       # optimizer_G_AB.zero_grad()
        #optimizer_G_BA.zero_grad()

        # Identity loss do we need this loss ?? not mention in the paper: does exist in auther's naryto_binaryimplementation
        id_A = G_BA(real_A)
        id_B = G_AB(real_B)
        loss_id_A = dice_loss(id_A,real_A)
        loss_id_B = dice_loss(id_B,real_B)

        loss_identity = (loss_id_A + loss_id_B) / 2
        avg_idt_loss = avg_idt_loss + loss_identity.item()
        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
        avg_adv_loss = avg_adv_loss + loss_GAN.item()

        # Cycle loss
        recov_A = G_BA(fake_B)
        #real_A_sq = torch.squeeze(real_A,1)
        #print(real_A_sq.size())
        loss_cycle_A = dice_loss(recov_A,real_A)
        recov_B = G_AB(fake_A)
        #real_B_sq = torch.squeeze(real_B,1)
        loss_cycle_B = dice_loss(recov_B,real_B)


        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
        avg_cycle_loss = avg_cycle_loss + loss_cycle.item()
        # Total loss
        
        loss_G =    loss_GAN + \
                    lambda_cyc * loss_cycle + \
                    lambda_id * loss_identity
        avg_G_loss = avg_G_loss + loss_G.item()
        
        #loss_G_AB = loss_GAN_AB + lambda_cyc * loss_cycle_B + lambda_id * loss_id_B
        #loss_G_BA = loss_GAN_BA + lambda_cyc * loss_cycle_A + lambda_id * loss_id_A
        #avg_G_AB_loss = avg_G_AB_loss + loss_G_AB.item()
        #avg_G_BA_loss = avg_G_BA_loss + loss_G_BA.item()
        #loss_G_AB.backward(retain_graph=True)
        #optimizer_G_AB.step()
        #loss_G_BA.backward(retain_graph=True)
        #optimizer_G_BA.step()
        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2
        avg_D_loss = avg_D_loss + loss_D.item()
        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        localtime = time.asctime(time.localtime(time.time()))

        # Print log
        
        if batches_done % 100 == 0: 
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f]" %
                                                        (epoch, opt.n_epochs,
                                                        i, len(dataloader),
                                                        loss_D.item(), loss_G.item(),
                                                        loss_GAN.item(),loss_cycle.item()))
        
        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)
        '''
        if ((i ==0) & (epoch == 0)):
            loss_plot = vis.line(
                     Y=np.column_stack((np.array([loss_G.item()]),np.array([loss_D.item()]),np.array([loss_GAN.item()]),np.array([loss_identity.item()]),np.array([loss_cycle.item()]))),
                     X = np.column_stack((np.array(batches_done),np.array(batches_done),np.array(batches_done),np.array(batches_done),np.array(batches_done))),
                     opts = dict(title = '5 lines'))
            vis.text(str(localtime) + ": [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, idt: %f , cycle: %f] \n" %
                                                        (epoch, opt.n_epochs,
                                                        i, len(dataloader),
                                                        loss_D.item(), loss_G.item(),
                                                        loss_GAN.item(), loss_identity.item(),loss_cycle.item()),win='output')

        elif (batches_done % opt.figure_interval == 0):
            vis.line(
                Y=np.column_stack((np.array([loss_G.item()]),np.array([loss_D.item()]),np.array([loss_GAN.item()]),np.array([loss_identity.item()]),np.array([loss_cycle.item()]))),
                X = np.column_stack((np.array(batches_done),np.array(batches_done),np.array(batches_done),np.array(batches_done),np.array(batches_done))),
                win = loss_plot,
                update = 'append')
            vis.text(str(localtime) + ": [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, idt: %f , cycle: %f] \n" %
                                                        (epoch, opt.n_epochs,
                                                        i, len(dataloader),
                                                        loss_D.item(), loss_G.item(),
                                                        loss_GAN.item(), loss_identity.item(),loss_cycle.item()),win='output',append = True)


    '''
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    #lr_scheduler_G_AB.step()
    #lr_scheduler_G_BA.step()
    print("------------------[Epoch %d/%d] [D loss: %f] [AVG G loss: %f, AVG adv: %f, AVG cycle: %f]" %
                                                        (epoch, opt.n_epochs,
                                                        avg_D_loss/leng, avg_G_loss/leng,
                                                        avg_adv_loss/leng,avg_cycle_loss/leng))


    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), 'saved_models/%s/G_AB_%d.pth' % (opt.dataset_name, epoch))
        torch.save(G_BA.state_dict(), 'saved_models/%s/G_BA_%d.pth' % (opt.dataset_name, epoch))
        torch.save(D_A.state_dict(), 'saved_models/%s/D_A_%d.pth' % (opt.dataset_name, epoch))
        torch.save(D_B.state_dict(), 'saved_models/%s/D_B_%d.pth' % (opt.dataset_name, epoch))
