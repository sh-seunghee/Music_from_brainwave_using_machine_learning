
'''
Author: Guangyu Shen
Last modified: 11/01/2018
This moudle is for generating labels for music images
'''

import os
from midi2mat import midi2mat
import argparse
import numpy as np
from PIL import Image
import sys
import math


parser = argparse.ArgumentParser()
parser.add_argument('--sourse_path',type = str)
parser.add_argument('--sink_path',type = str)
parser.add_argument('--mode',type = int)
parser.add_argument('--lower_bound',type = int, default = 1)
parser.add_argument('--upper_bound',type = int, default = 88)

args = parser.parse_args()
sourse_path = args.sourse_path
sink_path = args.sink_path
files = os.listdir(sourse_path)
txtfile_name = args.sink_path + '/' + 'label.txt'
s = []
for file in files:
	if not os.path.isdir(file):
		if 'mid' in file:
			filename = sourse_path + '/' + file
			output = midi2mat(filename,args.upper_bound,args.lower_bound)
			output = output[0]
			dim = len(output)
			num = math.floor(dim / 1000)
			file = file.replace('.mid', '')
			for i in range(num):
				file_new = file + '_part_' + str(i)
				file_img = sink_path + '/' + file_new 
				#im = Image.fromarray(output[i:i+1000][:])
				'''
				if im.mode != 'RGB':
					
				'''
				#im = im.convert('L')
				#im.save(file_img + '.png')
				np.save(file_img + '.png',output[i:i+1000][:])
				print(file_new)
				with open(txtfile_name,'a') as f:
					f.write(file_new + '\t' + str(0) + '\n')


