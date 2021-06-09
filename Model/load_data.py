import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageData(Dataset):
    '''
    This module intend to implement map-style data loader
    '''
    def __init__(self, landscape_dir, photo_dir, size=(256, 256), normalize=True):
        super().__init__()
        self.landscape_dir = landscape_dir
        self.photo_dir = photo_dir
        self.landscape_idx = dict()
        self.photo_idx = dict()

        if normalize:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])                                
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor()                               
            ])
        for i, j in enumerate(os.listdir(self.landscape_dir)):
            try: 
                self.landscape_idx[i] = j
            except:
                pass
        for i, j in enumerate(os.listdir(self.photo_dir)):
            
            try:
                self.photo_idx[i] = j
            except:
                pass

    def __getitem__(self, idx):
        min_len = min(len(self.photo_idx.keys()),len(self.landscape_idx.keys()))
        i = int(np.random.uniform(0,min_len))
        photo_path = os.path.join(self.photo_dir, self.photo_idx[i])
        landscape_path = os.path.join(self.landscape_dir, self.landscape_idx[i])
        photo_img = Image.open(photo_path).convert('RGB') 
        photo_img = self.transform(photo_img)
        landscape_img = Image.open(landscape_path).convert('RGB') 
        landscape_img = self.transform(landscape_img)
        return photo_img, landscape_img

    def __len__(self):
        min_len = min(len(self.photo_idx.keys()),len(self.landscape_idx.keys()))
        return min_len
