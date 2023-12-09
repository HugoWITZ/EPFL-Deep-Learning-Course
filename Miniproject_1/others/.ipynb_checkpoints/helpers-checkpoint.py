from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torch

'''
contains functions and classes necessary for training of the models
'''


# =========================== TO LOAD DATA =========================== #

class NoisyDataset(Dataset):
    '''
    customed dataset class for our training dataset
    # tuto https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    '''

    def __init__(self, noisy, clean):
      '''Initialize the dataset'''
      self.noisy = noisy
      self.clean = clean
    
    def __len__(self):
      '''Return total number of samples'''
      return len(self.noisy)

    def __getitem__(self, index):
      '''Return one sample of data'''
      n = self.noisy[index]
      c = self.clean[index]
      return n, c



# =========================== FOR TRAINING OF MODELS =========================== #


def psnr(denoised, ground_truth):
    ''' Peak Signal to Noise Ratio: denoised and ground_truth have range [0 , 1] '''
    mse = torch.mean((denoised - ground_truth)**2)
    return -10*torch.log10(mse + 10**-8)
