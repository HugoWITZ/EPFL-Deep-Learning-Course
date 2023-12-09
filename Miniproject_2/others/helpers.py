from torch.utils.data import DataLoader, Dataset
# !pip install kornia
# import kornia as K
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torch


# =========================== FOR TRAINING AND EVALUATION OF MODELS =========================== #


def psnr(denoised, ground_truth):
    ''' Peak Signal to Noise Ratio: denoised and ground_truth have range [0 , 1] '''
    mse = torch.mean((denoised - ground_truth)**2)
    return -10*torch.log10(mse + 10**-8)


def plot_loss(N, losses):
    '''
    plot losses and psnr scores' evolution during traning
    '''
    n_epochs = range(N)

    plt.figure(figsize=(7, 6))
    plt.plot(n_epochs, losses)
    plt.xlabel('Epochs')
    plt.ylabel('Mean loss')
    plt.grid()

    plt.savefig('drive/Shareddrives/DeepLearning/graphemodel2.png', format='png')



def show_prediction(noisy, clean, predicted):
    '''
    visualize 3 images together: clean and noisy pair of images of the testing set,
    along with the dneoised prediction of the model
    '''
    noisy_img = noisy
    clean_img = clean

    plt.figure(figsize=(16,4.5))
    plt.subplot(1,3,1)
    clean_img1 = clean_img[2]
    clean_img1 = torch.permute(clean_img1, (1, 2, 0))
    plt.imshow(clean_img1)
    plt.title('Clean image')

    plt.subplot(1,3,2)
    noisy_img = noisy_img[2]
    noisy_img = torch.permute(noisy_img, (1, 2, 0))
    plt.imshow(noisy_img)
    plt.title('Noisy image')

    plt.subplot(1,3,3)
    prediction_img = predicted[2].cpu()
    prediction_img = torch.permute(prediction_img, (1, 2, 0))
    plt.imshow(prediction_img.detach().numpy()) # FAIT CHIER CE NUMPY LA !!!
    plt.title('Denoiser output')

    plt.savefig('drive/Shareddrives/DeepLearning/outputmodel2.png', format='png')

    print(type(predicted), type(clean_img))
    psnr_score = psnr(predicted.to(device), clean_img.to(device)) # (prediction_img, clean_img)
    print('PSNR score on the first image of the validation set :', psnr_score)