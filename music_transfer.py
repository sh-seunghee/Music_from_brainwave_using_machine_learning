'''
Author: Guangyu Shen
Last modified: 11/28/2018

This module is for transfering brainwave music into classical and jazz music
You need to load two params files into function to_transfer G_AB_classical,G_AB_jazz
filename is the path of input brainwave midi file
The generated classical and jazz files named filename_classical and filename_jazz will be saved in the current dict.

'''


import torch
import numpy as np 
from mat2midi import piano_roll2midi
from midi2mat import piano_roll_generator
from model_v27 import GeneratorResNet

from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

def to_binary(tensor):
    '''
    one_hot_tensor = torch.zeros(1,1,128,100)
    _,index = torch.max(tensor,3)
    index = torch.squeeze(index)
    for i in range(100):
        one_hot_tensor[:,:,i,index[i]] = 127
    '''
    tensor[tensor > 0.5] = 127
    tensor[tensor <= 0.5] = 0
    return tensor
    #return one_hot_tensor




def to_transfer(filename,G_AB_classical_1 = None,G_AB_jazz_1 = None,fs = 10,batch_size = 100):

    G_AB_classical = GeneratorResNet(res_blocks=6)
    G_AB_jazz = GeneratorResNet(res_blocks=6)
    G_AB_classical.load_state_dict(torch.load(G_AB_classical_1,map_location='cpu'))
    G_AB_jazz.load_state_dict(torch.load(G_AB_jazz_1,map_location='cpu'))
    output = piano_roll_generator(filename,fs=fs,batch_size = batch_size)

    filename_classical = ""
    filename_jazz = ""

    if len(output)>0:
        dim = len(output)
        style = np.zeros([128,1])
        og = np.zeros([128,1])
        for i in range(dim):
            real_A = output[i]
            real_A_input = real_A
            real_A_input[real_A_input != 0] = 127

            real_A_tensor = torch.from_numpy(real_A_input)

            real_A_tensor = torch.unsqueeze(real_A_tensor,0)
            real_A_tensor = torch.unsqueeze(real_A_tensor,1).type(torch.FloatTensor)

            fake_B_classical = torch.squeeze(to_binary(G_AB_classical(real_A_tensor)))
            fake_B_jazz = torch.squeeze(to_binary(G_AB_jazz(real_A_tensor)))

            fake_B_numpy_classical = fake_B_classical.data.numpy()
            fake_B_numpy_jazz = fake_B_jazz.data.numpy()


            style_classical = np.concatenate((style,fake_B_numpy_classical),axis = 1)
            style_jazz = np.concatenate((style,fake_B_numpy_jazz),axis = 1)
            og = np.concatenate((og,real_A),axis = 1)


        style_jazz = style_jazz[:,1:]
        style_classical = style_classical[:,1:]
        og = og[:,1:]
        filename_split = filename.split('/')
        num = len(filename_split)
        filename = filename_split[num-1]
        filename = filename.replace('.mid','')
        filename_classical = filename + '_classical'
        filename_jazz = filename + '_jazz'
        piano_roll2midi(style_jazz,filename_jazz,3)
        piano_roll2midi(style_classical,filename_classical,3)

    return filename_classical, filename_jazz


