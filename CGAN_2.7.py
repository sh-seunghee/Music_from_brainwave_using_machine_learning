
'''
author:eriklindernoren
last modified: 11/1/2018

This module is used to define a GAN model and train it based on our music image dataset.


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

'''

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from torch.autograd import Variable
import time
import torch.nn as nn
import torch.nn.functional as F
import torch
from visdom import Visdom
from PIL import Image

#os.makedirs('images')

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('--data_path', type=str, default='', help='dataset path')
parser.add_argument('--batch_size', type=int, default=100, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=1000, help='dimensionality of the latent space')
parser.add_argument('--n_classes', type=int, default=2, help='number of classes for dataset')
parser.add_argument('--n_features', type=int, default=200, help='number of features for generating data')
parser.add_argument('--img_length', type=int, default=1000, help='size of each image dimension')
parser.add_argument('--img_width', type=int, default=90, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=50, help='interval between image sampling')
parser.add_argument('--model_path', type=str, default='')
opt = parser.parse_args()
print(opt)

vis = Visdom(env = 'CGAN_model')


img_shape = (opt.channels, opt.img_length, opt.img_width)
output_shape = (128,opt.img_length, opt.img_width)
cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_features)

        def block(in_feat, out_feat, normalize=True):
            layers = [  nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim+opt.n_features, 128),
            nn.Linear(128,256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256,512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512,1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(np.prod(output_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *(output_shape))# img.size(0) stands for the batch_size
        _,img = torch.max(img,1)
        print(img.size())
        img = img.type(torch.FloatTensor)
        return img  # get a batch of generated images

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_features)

        self.model = nn.Sequential(
            nn.Linear(opt.n_features + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity

# Loss functions
adversarial_loss = torch.nn.MSELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Configure data loader
#os.makedirs('../../data/mnist', exist_ok=True)
trans = transforms.Compose([transforms.ToTensor()])
train_dataset = dset.ImageFolder(opt.data_path,transform = trans)
dataloader = data.DataLoader(train_dataset,batch_size = opt.batch_size, shuffle = True)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
print('generater params:' + str(generator.parameters()))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row*2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    gen_imgs_np = gen_imgs.numpy()
    np.save('images/%d.npy'%batches_done,gen_imgs_np)

    #save_image(gen_imgs.data, 'images/%d.png' % batches_done, nrow=n_row, normalize=False)
    vis.image(gen_imgs.data,opts = dict(title = 'sample_image',nrow = n_row))

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, data in enumerate(dataloader,0):
        imgs, labels = data
        
        a1 = imgs[1,0,:,:]
        a2 = imgs[1,2,:,:]
        a3 = imgs[1,1,:,:]
        print((a1==a2).all())
        print((a1==a3).all())
        print((a2==a3).all())
        
        
        imgs = imgs[:,0,:,:]
        batch_size = imgs.shape[0]
        print('image size is:' + str(imgs.size()))

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))


        labels = Variable(labels.type(LongTensor))  # one-hot? 0-9

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))# generate randomly
        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()
        localtime = time.asctime(time.localtime(time.time()))

        
        if (i == 0) & (epoch == 0):
            loss_plot = vis.line(Y=np.column_stack((np.array([g_loss.item()]),np.array([d_loss.item()]))),
                                 X = np.column_stack((np.array(epoch*12+i),np.array(epoch*12+i))),
                                 opts = dict(title = 'Two_lines'))
            vis.text(str(localtime) + ": [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                            d_loss.item(), g_loss.item()),win='output')
        elif (i == 1) & (epoch == 0):
            print('So big')
        else:
            vis.line(
                Y=np.column_stack((np.array([g_loss.item()]),np.array([d_loss.item()]))),
                X = np.column_stack((np.array(epoch*12+i),np.array(epoch*12+i))),
                win = loss_plot,
                update = 'append')
            vis.text(str(localtime) + ": [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                            d_loss.item(), g_loss.item()),win='output',append = True)
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=2, batches_done=batches_done)
torch.save(generator.state_dict(),opt.model_path)