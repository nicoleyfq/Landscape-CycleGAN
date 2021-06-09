import os
import argparse
import time
import PIL
import numpy as np
import pandas as pd
import itertools
from glob import glob
from PIL import Image
import random
import shutil
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_curve
from sklearn import metrics
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from load_data import *
from cycleGAN import *
from checkpoint import *

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main():

    #Required parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default='Harvard', type=str, required=False,
                        help="The input data dir. Should be Haravd etc.")

    parser.add_argument("--epoch", default=1, type=int, required=False,
                        help="Number of epochs of training")
    args = parser.parse_args()
    data_path1 = args.subset + '/jpg/'
    data_path2 = args.subset + '/'
    painting_path = os.path.join('../data/paintings/',data_path1)
    sketch_path = os.path.join('../data/sketch/',data_path2)
    #print(painting_path)
    #print(sketch_path)

    #set seed and file path
    set_seed(888)

    #read data
    images_data = ImageData(painting_path,sketch_path)
    images_loader = DataLoader(images_data, batch_size=1, pin_memory=True)
    photo_img,landscape_img = next(iter(images_loader))


    run_CycleGAN(args.epoch,images_loader)

    

if __name__ == '__main__':
    main()