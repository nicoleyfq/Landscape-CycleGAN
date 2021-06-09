import os
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
from checkpoint import *

'''
Code implementation of CycleGAN model referrencing: 
Public Github Repository: nachiket273,aitorzip
'''
def unnorm(img, mean=[0.5, 0.5,0.5], std=[0.5, 0.5,0.5]):
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img

def downsample(in_channel, out_channel, apply_dropout=True):
    '''
    For generator model, first build the downsample and upsample functions, 
    where upsample increase the dimension of the images 
    and downsample do the opposite.
    '''
    downsample_model = []
    downsample_model.append(nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1, output_padding=1))
    downsample_model.append(nn.InstanceNorm2d(out_channel))
    if apply_dropout:
        downsample_model.append(nn.Dropout(0.5))
    downsample_model.append(nn.ReLU())
    
    return nn.Sequential(*downsample)

def upsample(in_channel, out_channel, apply_dropout=True):
    upsample_model = []
    upsample_model.append(nn.ConvTranspose2d(in_channel, out_channel, 3, stride=2, padding=1, output_padding=1))
    upsample_model.append(nn.InstanceNorm2d(out_channel))
   
    if apply_dropout:
        upsample_model.append(nn.Dropout(0.5))
    upsample_model.append(nn.ReLU())
    
    return nn.Sequential(*upsample_model)

def Conv_2(in_channel, out_channel, kernels=3, stride=2, leaky=True, inst_norm=True, pad=True):
    model = []
    
    if pad:
        model.append(nn.Conv2d(in_channel, out_channel, kernels, stride, 1, bias=True))  
    else:
        model.append(nn.Conv2d(in_channel, out_channel, kernels, stride, 0, bias=True))

    if inst_norm:
        model.append(nn.InstanceNorm2d(out_channel))
    else:
        model.append(nn.BatchNorm2d(out_ch))
        
    if leaky:
        model.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
    else:
        model.append(nn.ReLU())

    return nn.Sequential(*model)


class ResidualBlock(nn.Module):
    '''
    Residual Blocks are skip-connection blocks that learn 
    residual functions with reference to the layer inputs, 
    instead of learning unreferenced functions.
    '''
    def __init__(self, dim, apply_dropout=True):
        super().__init__()
        self.conv1 = Conv_2(dim, dim, kernels=3, stride=1, leaky=False, inst_norm=True,pad=True)
        self.conv2 = Conv_2(dim,dim, kernels=3, stride=1, leaky=False, inst_norm=True,pad=True)

    def forward(self, x):
        out_1 = F.relu(self.conv1(x))
        out_2 = x + self.conv2(out_1)
        return out_2

    
class Generator(nn.Module):
    '''
    Generator Model:
    '''
    def __init__(self, in_chanel, out_chanel, num_res_blocks=6):
        super().__init__()
        model = list()
        model.append(nn.ReflectionPad2d(3))
        model.append(Conv_2(in_chanel, 64, 7, 1, False, True, False))
        model.append(Conv_2(64, 128, 3, 2, False))
        model.append(Conv_2(128, 256, 3, 2, False))
        for _ in range(num_res_blocks):
            model.append(ResidualBlock(256))
        model.append(upsample(256, 128))
        model.append(upsample(128, 64))
        model.append(nn.ReflectionPad2d(3))
        model.append(nn.Conv2d(64, out_chanel, kernel_size=7, padding=0))
        model.append(nn.Tanh())

        self.generate = nn.Sequential(*model)

    def forward(self, x):
        return self.generate(x)

class Discriminator(nn.Module):
    '''
    Discriminator Model:
    '''
    def __init__(self, in_chanel, n=4):
        super().__init__()
        model = list()
        model.append(nn.Conv2d(in_chanel, 64, 4, stride=2, padding=1))
        model.append(nn.LeakyReLU(0.2, inplace=True))
        for i in range(1, n):
            in_chanel = 64 * 2**(i-1)
            out_chanel = in_chanel * 2
            if i == n -1:
                model.append(Conv_2(in_chanel, out_chanel, 4, 1))
            else:
                model.append(Conv_2(in_chanel, out_chanel, 4, 2))
        model.append(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1))
        self.disc = nn.Sequential(*model)

    def forward(self, x):
        return self.disc(x)

def weights_init(m):
    """
    Applies initial weights to certain layers in a model.
    The weights are taken from a normal distribution with mean = 0, std dev = 0.02.
    Param m: A module or layer in a network    
    """
    classname = m.__class__.__name__
    
    std = 0.02
    mean = 0.0
    
    if hasattr(m, 'weight') and (classname.find('Conv') != -1):
        init.normal_(m.weight.data, mean, std)

def update_req_grad(models, requires_grad=True):
    for model in models:
        for param in model.parameters():
            param.requires_grad = requires_grad

class sample_fake(object):
    '''
    Referrencing https://arxiv.org/pdf/1612.07828.pdf
    '''

    def __init__(self, max_imgs=50):
        self.max_imgs = max_imgs
        self.cur_img = 0
        self.imgs = list()

    def __call__(self, imgs):
        ret = list()
        for img in imgs:
            if self.cur_img < self.max_imgs:
                self.imgs.append(img)
                ret.append(img)
                self.cur_img += 1
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_imgs)
                    ret.append(self.imgs[idx])
                    self.imgs[idx] = img
                else:
                    ret.append(img)
        return ret
class lr_sched():
    def __init__(self, decay_epochs=100, total_epochs=200):
        self.decay_epochs = decay_epochs
        self.total_epochs = total_epochs

    def step(self, epoch_num):
        if epoch_num <= self.decay_epochs:
            return 1.0
        else:
            fract = (epoch_num - self.decay_epochs)  / (self.total_epochs - self.decay_epochs)
            return 1.0 - fract
class AvgStats(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.losses =[]
        self.its = []
        
    def append(self, loss, it):
        self.losses.append(loss)
        self.its.append(it)

#set hyperparameters
lr=0.0002 
beta1=0.5  #exponential decay rate for the first moment estimates
beta2=0.999 #exponential decay rate for the second-moment estimates
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('********device is:',device,'********')

class CycleGAN(object):

    def __init__(self, in_chanel, out_chanel, epochs, device,lr=2e-4, lmbda=10, idt_coef=0.5):
        self.epochs = epochs
        self.decay_epoch = int(self.epochs/2)
        self.lmbda = lmbda
        self.idt_coef = idt_coef
        self.device = device
        self.gen1 = Generator(in_chanel, out_chanel)
        self.gen2 = Generator(in_chanel, out_chanel)
        self.disc1 = Discriminator(in_chanel)
        self.disc2 = Discriminator(in_chanel)
        self.init_models()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.adam_gen = torch.optim.Adam(itertools.chain(self.gen1.parameters(), self.gen2.parameters()),
                                         lr = lr, betas=(beta1, beta2))
        self.adam_disc = torch.optim.Adam(itertools.chain(self.disc1.parameters(), self.disc2.parameters()),
                                          lr=lr, betas=(beta1, beta2))
        self.sample_l = sample_fake()
        self.sample_p = sample_fake()
        gen_lr = lr_sched(self.decay_epoch, self.epochs)
        disc_lr = lr_sched(self.decay_epoch, self.epochs)
        self.gen_lr_sched = torch.optim.lr_scheduler.LambdaLR(self.adam_gen, gen_lr.step)
        self.disc_lr_sched = torch.optim.lr_scheduler.LambdaLR(self.adam_disc, disc_lr.step)
        self.gen_stats = AvgStats()
        self.disc_stats = AvgStats()
        
    def init_models(self):
        weights_init(self.gen1)
        weights_init(self.gen2)
        weights_init(self.disc1)
        weights_init(self.disc2)
        self.gen1 = self.gen1.to(self.device)
        self.gen2 = self.gen2.to(self.device)
        self.disc1 = self.disc1.to(self.device)
        self.disc2 = self.disc2.to(self.device)
        
    def train(self, photo_dl):
        for epoch in range(self.epochs):
            start_time = time.time()
            avg_gen_loss = 0.0
            avg_disc_loss = 0.0
            t = tqdm(photo_dl, leave=False, total=photo_dl.__len__())
            for i, (p, l) in enumerate(t):
                photo_img, landscape_img = p.to(device), l.to(device) #load real images
                update_req_grad([self.disc1, self.disc2], False)
                
                # Zero out all of the gradients for the variables which the optimizer
                # will update.
                self.adam_gen.zero_grad()

                # Forward pass through generator
                fake_p = self.gen1(landscape_img)
                fake_l = self.gen2(photo_img)

                cycl_l = self.gen2(fake_p)
                cycl_p = self.gen1(fake_l)

                id_l = self.gen2(landscape_img)
                id_p = self.gen1(photo_img)

                # generator losses - identity, Adversarial, cycle consistency
                idt_loss_l = self.l1_loss(id_l, landscape_img) * self.lmbda * self.idt_coef
                idt_loss_p = self.l1_loss(id_p, photo_img) * self.lmbda * self.idt_coef

                cycle_loss_l = self.l1_loss(cycl_l, landscape_img) * self.lmbda
                cycle_loss_p = self.l1_loss(cycl_p, photo_img) * self.lmbda

                disc_l = self.disc1(fake_l)
                disc_p = self.disc2(fake_p)

                real = torch.ones(disc_l.size()).to(self.device)

                adv_loss_l = self.mse_loss(disc_l, real)
                adv_loss_p = self.mse_loss(disc_p, real)

                # compute total generator loss and average loss
                total_gen_loss = cycle_loss_l + adv_loss_l\
                              + cycle_loss_p + adv_loss_p\
                              + idt_loss_l + idt_loss_p
                
                avg_gen_loss += total_gen_loss.item()


                # This is the backwards pass: compute the gradient of the loss with
                # respect to each  parameter of the model.
                total_gen_loss.backward()
                # Actually update the parameters of the model using the gradients
                # computed by the backwards pass.
                self.adam_gen.step()

                # Forward pass through discriminators
                update_req_grad([self.disc1, self.disc2], True)
                self.adam_disc.zero_grad()

                fake_l = self.sample_l([fake_l.cpu().data.numpy()])[0]
                fake_p = self.sample_p([fake_p.cpu().data.numpy()])[0]
                fake_l = torch.tensor(fake_l).to(self.device)
                fake_p = torch.tensor(fake_p).to(self.device)

                land_disc_real = self.disc1(landscape_img)
                land_disc_fake = self.disc1(fake_l)
                photo_disc_real = self.disc2(photo_img)
                photo_disc_fake = self.disc2(fake_p)

                real = torch.ones(land_disc_real.size()).to(self.device)
                fake = torch.ones(land_disc_fake.size()).to(self.device)

                # discriminators loss
                land_disc_real_loss = self.mse_loss(land_disc_real, real)
                land_disc_fake_loss = self.mse_loss(land_disc_fake, fake)
                photo_disc_real_loss = self.mse_loss(photo_disc_real, real)
                photo_disc_fake_loss = self.mse_loss(photo_disc_fake, fake)

                land_disc_loss = (land_disc_real_loss + land_disc_fake_loss) / 2
                photo_disc_loss = (photo_disc_real_loss + photo_disc_fake_loss) / 2
                total_disc_loss = land_disc_loss + photo_disc_loss
                avg_disc_loss += total_disc_loss.item()

                # Backward
                land_disc_loss.backward()
                photo_disc_loss.backward()
                self.adam_disc.step()
                
                t.set_postfix(gen_loss=total_gen_loss.item(), disc_loss=total_disc_loss.item())

            save_dict = {
                'epoch': epoch+1,
                'gen1': gan.gen1.state_dict(),
                'gen2': gan.gen2.state_dict(),
                'disc1': gan.disc1.state_dict(),
                'disc2': gan.disc2.state_dict(),
                'optimizer_gen': gan.adam_gen.state_dict(),
                'optimizer_disc': gan.adam_disc.state_dict()
            }
            save_checkpoint(save_dict, 'current.ckpt')
            
            avg_gen_loss /= photo_dl.__len__()
            avg_disc_loss /= photo_dl.__len__()
            time_req = time.time() - start_time
            
            self.gen_stats.append(avg_gen_loss, time_req)
            self.disc_stats.append(avg_disc_loss, time_req)
            
            print("Epoch: (%d) | Generator Loss:%f | Discriminator Loss:%f" %  (epoch+1, avg_gen_loss, avg_disc_loss))
      
            self.gen_lr_sched.step()
            self.disc_lr_sched.step()

def run_CycleGAN(num_epoch,images_loader):

    global gan 
    gan = CycleGAN(3, 3, num_epoch, device)

    #initialize the dictionary before training
    save_dict = {
        'epoch': 0,
        'gen1': gan.gen1.state_dict(),
        'gen2': gan.gen2.state_dict(),
        'disc1': gan.disc1.state_dict(),
        'disc2': gan.disc2.state_dict(),
        'optimizer_gen': gan.adam_gen.state_dict(),
        'optimizer_disc': gan.adam_disc.state_dict()
    }

    save_checkpoint(save_dict, 'init.ckpt')
    gan.train(images_loader)

    # save output
    if not os.path.exists('../output'):
        os.mkdir('../output')

    trans = transforms.ToPILImage()
    for i in tqdm(range(len(images_loader))):
        photo_img, _ = next(iter(images_loader))
        with torch.no_grad():
            pred = gan.gen2(photo_img.to(device)).cpu().detach()
        pred = unnorm(pred)
        img = trans(pred[0]).convert("RGB")
        img.save("../output/" + str(i+1) + ".jpg")

    print('********outputs saved successfully********')