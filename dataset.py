import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(root +'brainmusic/brainmusic_data_2d/' + '*.npy'))
        self.files_B = sorted(glob.glob(root +'jazz/jazz_data_2d/' + '*.npy'))
    def __getitem__(self, index):
        A = np.load(self.files_A[index % (len(self.files_A)-1)])
        A_image = Image.fromarray(A)
        item_A = self.transform(A_image)

        if self.unaligned:
            B = np.load(self.files_B[random.randint(0, len(self.files_B) - 1)])
            B_image = Image.fromarray(B)
            item_B = self.transform(B_image)
        else:
            B = np.load(self.files_B[random.randint(0, len(self.files_B) - 1)])
            B_image = Image.fromarray(B)
            item_B = self.transform(B_image)

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
