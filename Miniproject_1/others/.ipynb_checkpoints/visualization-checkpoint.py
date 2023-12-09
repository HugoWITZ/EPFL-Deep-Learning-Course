from torch.utils.data import DataLoader, Dataset
# !pip install kornia
import kornia as K
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torch

'''
contains functions used for experimentations on models and visualizing of results
'''


# =========================== TO LOAD AND PREPROCESS DATA =========================== #


def augment_training_data(noisy_imgs_1, noisy_imgs_2):
    '''
    augment the training dataset by horizontal flipping and random rotation
    '''

    transform1 = K.augmentation.RandomHorizontalFlip3D(p=0.5, keepdim=True)
    transform2 = K.augmentation.RandomRotation(180, p=1, keepdim=True, resample='nearest')

    images = NoisyDataset(noisy_imgs_1, noisy_imgs_2)
    train_loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=2, shuffle=False) # load images by batch of size batch_size

    trans1 = []
    trans2 = []

    for noisy1_batch, noisy2_batch in iter(train_loader):

        new_noisy1 = transform1(noisy1_batch)
        new_noisy2 = transform1(noisy2_batch, params=transform1._params)

        new_noisy1 = transform2(new_noisy1)
        new_noisy2 = transform2(new_noisy2, params=transform2._params)
        
        trans1.extend(new_noisy1)
        trans2.extend(new_noisy2)

    return trans1, trans2


def load_and_preprocess_data(data_dir, augment=False):
    '''
    load training images set, transform them to float tensor and normalize them,
    augment the training dataset if augment==True
    '''
    noisy_imgs_1, noisy_imgs_2 = torch.load(data_dir + '/train_data.pkl')  # 50 000 pairs of images of 3 × H × W = 3 x 32 x 32
    noisy_imgs, clean_imgs = torch.load (data_dir + '/val_data.pkl')       #  1 000 pairs of images of 3 x H x W = 3 x 32 x 32

    noisy_imgs = noisy_imgs.float()/255
    clean_imgs = clean_imgs.float()/255

    if augment==True:
        noisy_imgs_1 = noisy_imgs_1.float() / 255
        noisy_imgs_2 = noisy_imgs_2.float() / 255

        new_noisy_1, new_noisy_2 = augment_training_data(noisy_imgs_1, noisy_imgs_2)
        new_noisy_1 = torch.stack(new_noisy_1, dim=0)
        new_noisy_2 = torch.stack(new_noisy_2, dim=0)

        noisy_imgs_1 = torch.cat((noisy_imgs_1, new_noisy_1))
        noisy_imgs_2 = torch.cat((noisy_imgs_2, new_noisy_2))

    else:
        noisy_imgs_1 = noisy_imgs_1.float() / 255
        noisy_imgs_2 = noisy_imgs_2.float() / 255

    return noisy_imgs_1, noisy_imgs_2, noisy_imgs, clean_imgs



# =========================== FOR EVALUATION OF MODELS =========================== #


def plot_loss_psnr(N, losses, psnrs):
    '''
    plot losses and psnr scores' evolution during traning
    '''
    n_epochs = range(N)

    plt.figure(figsize=(14, 6))
    plt.subplot(121)
    plt.plot(n_epochs, losses)
    plt.xlabel('Epochs')
    plt.ylabel('Mean loss')
    plt.grid()

    plt.subplot(122)
    plt.plot(n_epochs, psnrs)
    plt.xlabel('Epochs')
    plt.ylabel('Mean PSNR')
    plt.grid()

    plt.savefig('drive/Shareddrives/DeepLearning/basemodelReLU_aug_graphloss.png', format='png')



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

    plt.savefig('drive/Shareddrives/DeepLearning/basemodelReLU_aug_output.png', format='png')

    print(type(predicted), type(clean_img))
    psnr_score = psnr(predicted.to(device), clean_img.to(device)) # (prediction_img, clean_img)
    print('PSNR score on the first image of the validation set :', psnr_score)