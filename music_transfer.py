'''
Author: SolidShen
Last modified: 11/27/2018

This module is for transfering brainwave music into classical and jazz music
You need to load two params files into function to_transfer G_AB_classical,G_AB_jazz
filename is the path of input brainwave midi file
The generated classical and jazz files will be saved in the current dict.

'''






import torch
import numpy as np 
from mat2midi import piano_roll2midi
from midi2mat import piano_roll_generator
from cycleGan.model_v27 import * 
from cycleGan.dataset import *
from cycleGan.utils import *
from torch.utils.data import DataLoader
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




def to_transfer(filename,fs = 10,batch_size = 100,G_AB_classical,G_AB_jazz):
	G_AB_classical = GeneratorResNet(res_blocks=6)
	G_AB_jazz = GeneratorResNet(res_blocks=6)
	G_AB_classical.load_state_dict(torch.load(G_AB_classical,map_location='cpu'))
	G_AB_jazz.load_state_dict(torch.load(G_AB_jazz,map_location='cpu'))
	#G_BA.load_state_dict(torch.load('/Users/shenguangyu1/Desktop/purdue/CS501/proSE/code/cycleGan/saved_model_brainmusic_jazz_v2/G_BA_50.pth',map_location='cpu'))
	output = piano_roll_generator(filename,fs=fs,batch_size = batch_size)
	if len(output)>0:
		dim = len(output)
		style = np.zeros([128,1])
		og = np.zeros([128,1])
		for i in range(dim):
			real_A = output[i]
			real_A_input = real_A
			#real_B_input = real_B
			real_A_input[real_A_input != 0] = 127
			#real_B_input[real_B_input != 0] = 127

			real_A_tensor = torch.from_numpy(real_A_input)
			#real_B_tensor = torch.from_numpy(real_B_input)

			real_A_tensor = torch.unsqueeze(real_A_tensor,0)
			real_A_tensor = torch.unsqueeze(real_A_tensor,1).type(torch.FloatTensor)
			'''
			real_B_tensor = torch.unsqueeze(real_B_tensor,0)
			real_B_tensor = torch.unsqueeze(real_B_tensor,1).type(torch.FloatTensor)
			'''
			fake_B_classical = torch.squeeze(to_binary(G_AB_classical(real_A_tensor)))
			fake_B_jazz = torch.squeeze(to_binary(G_AB_jazz(real_A_tensor)))

			fake_B_numpy_classical = fake_B_classical.data.numpy()
			fake_B_numpy_jazz = fake_B_jazz.data.numpy()
			#fake_A_numpy = fake_A.data.numpy()

			style_classical = np.concatenate((style,fake_B_numpy_classical),axis = 1)
			style_jazz = np.concatenate((style,fake_B_numpy_jazz),axis = 1)
			og = np.concatenate((og,real_A),axis = 1)
			
			#fake_A_output = fake_A_numpy
		style_jazz = style_jazz[:,1:]
		style_classical = style_classical[:,1:]
		og = og[:,1:]
		piano_roll2midi(style_jazz,filename,5) 
		#piano_roll2midi(og,'sample_music/' + filename,5)
		piano_roll2midi(style_classical,filename,5)


#to_classical('EEG_Cat_Study4_Resting_S21channel52_SR250.mid')
