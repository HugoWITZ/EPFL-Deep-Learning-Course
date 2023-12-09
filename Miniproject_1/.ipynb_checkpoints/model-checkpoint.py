import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, Dataset

from pathlib import Path

from Miniproject_1 import device
from .others.helpers import NoisyDataset, psnr


class Model(nn.Module):

  def __init__(self):
    '''instantiate model + optimizer + loss function + any other stuff you need'''
    super().__init__()

    # Select device
    self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Instantiate model
    self.encoder = nn.Sequential(
                  nn.Conv2d(3, 32, 3, stride=1, padding=1),
                  nn.ReLU(True),
                  nn.Conv2d(32, 32, 3, stride=1, padding=1),
                  nn.ReLU(True),
                  nn.MaxPool2d(2)
                  )
    self.encoder2 = nn.Sequential(
                    nn.Conv2d(32, 64, 3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(64, 64, 3, stride=1, padding=1),
                    nn.ReLU(True)
                  )
    self.encoder3 = nn.Sequential(
                    nn.Conv2d(64, 64, 3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(64, 64, 3, stride=1, padding=1),
                    nn.ReLU(True)
                    )
    self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1),
                    nn.ReLU(True)
                    )
    self.decoder2 = nn.Sequential(
                    nn.ConvTranspose2d(32+64, 32, 3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1),
                    nn.ReLU(True)
                    )
    self.decoder3 = nn.Sequential(
                    nn.ConvTranspose2d(32+32, 32, 3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.Upsample(scale_factor=2, mode='bicubic')
                    )
    self.final = nn.Sequential(
                nn.Conv2d(32+3, 32, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, 3, stride=1, padding=1),
                nn.ReLU(inplace=True)
                )
        
    # Select optimizer
    self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=1e-08) 
    # self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.01, weight_decay=1e-05) # other tests 

    # Select loss function
    self.criterion = torch.nn.MSELoss()

    # Select scheduler to update learning rate
    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=2, factor=0.5, mode='max', verbose=True) 
    # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, patience=2, factor=0.5, mode='max', verbose=True)  # other tests


  def forward(self, x):

    e1 = self.encoder(x)
    e2 = self.encoder2(e1)
    e3 = self.encoder3(e2)
    d1 = self.decoder(e3)
    c1 = torch.cat((d1,e2), dim=1)
    d2 = self.decoder2(c1)
    c2 = torch.cat((d2,e1), dim=1)
    d3 = self.decoder3(c2)
    c3 = torch.cat((d3,x), dim=1)
    f = self.final(c3)
    return f


  def train(self, noisy1, noisy2, num_epochs=28) -> None: # val_noisy, val_clean) -> None:
    '''
    train_input : tensor of size (N, C, H, W) containing a noisy version of the images, here named noisy1
    train_target : tensor of size (N, C, H, W) containing another noisy version of the
     same images , which only differs from the input by their noise, here named noisy2
    training function, to evaluate wieight values for the training set composed of noisy1 and noisy2,
    uses the dataset val_noisy, val_clean for validation
    insired by DeepLearning Pratical 5
    '''

    # Parameters, to optimize
    batch_size = 512
    
    # load images by batch of size batch_size
    noisy1 = noisy1.float()/255.0
    noisy2 = noisy2.float()/255.0
    images = NoisyDataset(noisy1, noisy2)  # deblur images of blurry set 1 based on blurry set 2
    train_loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=0, shuffle=True)

    # # Used for training on our side, can not be used with the demanding format of function for the project hand out
    # # load 500 first test images for reduction of the learning rate
    # val_noisy, val_clean = torch.load (Path(__file__).child + '/val_data.pkl')
    # val_noisy = val_noisy.float()/255.0
    # val_clean = val_clean.float()/255.0

    # store losses and pnsr scores to visualize training
    losses = []
    psnrS = []

    for epoch in range(num_epochs):
        train_loss = 0.0

        for noisy1_batch, noisy2_batch in iter(train_loader):

          self.optimizer.zero_grad(set_to_none=True)
          output = self.forward(noisy1_batch.to(device)) #noisy1_batch)
          loss = self.criterion(output, noisy2_batch.to(device)) #output, noisy2_batch)
          loss.requires_grad = True
          loss.backward()
          self.optimizer.step()

          train_loss += loss.item()
            
        train_loss = train_loss/len(train_loader)
        losses.append(train_loss)
        print('Epoch: {}, training loss: {:.6f}'.format(epoch+1, train_loss))

        # # Used for training on our side, can not be used with the demanding format of function for the porjet hand out
        # with torch.no_grad():
        #   valid_loss = psnr(self.forward(val_noisy), val_clean) #val_noisy.to(device)).cpu(), val_clean)
        #   psnrS.append(valid_loss)
        #   print("val psnr: ", valid_loss)
        #   # update learning rate if no improvement for 3 batches in a row
        #   self.scheduler.step(valid_loss)
    
    # # To save the model parameters
    # torch.save(self.state_dict(), 'bestmodel.pth') # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        
    return


  def load_pretrained_model(self) -> None :
    '''This loads the parameters saved in bestmodel.pth into the model
    https://pytorch.org/tutorials/beginner/saving_loading_models.html'''
    model_path = Path(__file__).parent / "bestmodel.pth"
    model_dict = torch.load(model_path, map_location=torch.device('cpu'))
    self.load_state_dict(model_dict)


  def predict(self, test_input) -> torch.Tensor :
    '''
    test_input : tensor of size (N1 , C, H, W) that has to be denoised by the trained
      or the loaded network .
    returns a tensor of the size (N1 , C, H, W)
    '''
    test_input = test_input.float()/255.0
    test_input = test_input.to(self.device)
    output = self.forward(test_input)
    output = output * 255.0
    return output


# ====== OTHER MODELS TESTED : take the __init__() and forward() and put it in the above class to test them ================#


# class BaseModel(nn.Module):             __init__() and forward() for the model variant2; change the activation function of self.final to obtain base model

    #     def __init__(self):
    #         super().__init__()
    #         self.encoder = nn.Sequential(
    #                 nn.Conv2d(3, 32, 3, stride=1, padding=1),
    #                 nn.ReLU(True),
    #                 nn.MaxPool2d(2)
    #             )
    #         self.encoder2 = nn.Sequential(
    #                 nn.Conv2d(32, 64, 3, stride=1, padding=1),
    #                 nn.ReLU(True)
    #             )
    #         self.encoder3 = nn.Sequential(
    #                 nn.Conv2d(64, 64, 3, stride=1, padding=1),
    #                 nn.ReLU(True)
    #             )
    #         self.decoder = nn.Sequential(
    #                 nn.Conv2d(64, 32, 3, stride=1, padding=1),
    #                 nn.ReLU(True)
    #             )
    #         self.decoder2 = nn.Sequential(
    #                 nn.Conv2d(32+64, 32, 3, stride=1, padding=1),
    #                 nn.ReLU(True)
    #             )
    #         self.decoder3 = nn.Sequential(
    #                 nn.Conv2d(32+32, 32, 3, stride=1, padding=1),
    #                 nn.ReLU(True),
    #                 nn.Upsample(scale_factor=2, mode='bicubic')
    #             )
    #         self.final = nn.Sequential(
    #                 nn.Conv2d(32+3, 32, 3, stride=1, padding=1),
    #                 nn.ReLU(inplace=True),
    #                 nn.Conv2d(32, 3, 3, stride=1, padding=1),
    #                 nn.ReLU(inplace=True)
    #                 # nn.Sigmoid()
    #             )

    #     def forward(self, x):
    #     # uses model to compute the output (="deblur" image) for a given blurry image, input
    #         e1 = self.encoder(x)
    #         e2 = self.encoder2(e1)
    #         e3 = self.encoder3(e2)
    #         d1 = self.decoder(e3)
    #         c1 = torch.cat((d1,e2), dim=1)
    #         d2 = self.decoder2(c1)
    #         c2 = torch.cat((d2,e1), dim=1)
    #         d3 = self.decoder3(c2)
    #         c3 = torch.cat((d3,x), dim=1)
    #         f = self.final(c3)
    #         return f


    
    # class Variant2(nn.Module):      # __init__() and forward() for the model variant2

    #     def __init__(self):
    #         super().__init__()
    #         self.encoder = nn.Sequential(
    #                 nn.Conv2d(3, 32, 3, stride=1, padding=1),
    #                 nn.ReLU(True),
    #                 nn.MaxPool2d(2)
    #                 )
    #         self.encoder2 = nn.Sequential(
    #                 nn.Conv2d(32, 64, 3, stride=1, padding=1),
    #                 nn.ReLU(True)
    #             )
    #         self.encoder3 = nn.Sequential(
    #                 nn.Conv2d(64, 64, 3, stride=1, padding=1),
    #                 nn.ReLU(True)
    #             )
    #         self.decoder = nn.Sequential(
    #                 nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
    #                 nn.ReLU(True)
    #             )
    #         self.decoder2 = nn.Sequential(
    #                 nn.ConvTranspose2d(32+64, 32, 3, stride=1, padding=1),
    #                 nn.ReLU(True)
    #             )
    #         self.decoder3 = nn.Sequential(
    #                 nn.ConvTranspose2d(32+32, 32, 3, stride=1, padding=1),
    #                 nn.ReLU(True),
    #                 nn.Upsample(scale_factor=2, mode='bicubic')
    #             )
    #         self.final = nn.Sequential(
    #                 nn.Conv2d(32+3, 32, 3, stride=1, padding=1),
    #                 nn.ReLU(inplace=True),
    #                 nn.Conv2d(32, 3, 3, stride=1, padding=1),
    #                 nn.ReLU(inplace=True)
    #             )
        
    #     def forward(self, x):
    #         e1 = self.encoder(x)
    #         e2 = self.encoder2(e1)
    #         e3 = self.encoder3(e2)
    #         d1 = self.decoder(e3)
    #         c1 = torch.cat((d1,e2), dim=1)
    #         d2 = self.decoder2(c1)
    #         c2 = torch.cat((d2,e1), dim=1)
    #         d3 = self.decoder3(c2)
    #         c3 = torch.cat((d3,x), dim=1)
    #         f = self.final(c3)
    #         return f




    # class Variant3(nn.Module):    ## __init__() and forward() for the model variant3

    #     def __init__(self):
    #         super().__init__()
    #         self.encoder = nn.Sequential(
    #                 nn.Conv2d(3, 32, 3, stride=1, padding=1),
    #                 nn.ReLU(True),
    #                 nn.Conv2d(32, 32, 3, stride=1, padding=1),
    #                 nn.ReLU(True),
    #                 nn.MaxPool2d(2)
    #             )
    #         self.encoder2 = nn.Sequential(
    #                 nn.Conv2d(32, 64, 3, stride=1, padding=1),
    #                 nn.ReLU(True),
    #                 nn.Conv2d(64, 64, 3, stride=1, padding=1),
    #                 nn.ReLU(True)
    #             )
    #         self.encoder3 = nn.Sequential(
    #                 nn.Conv2d(64, 64, 3, stride=1, padding=1),
    #                 nn.ReLU(True),
    #                 nn.Conv2d(64, 64, 3, stride=1, padding=1),
    #                 nn.ReLU(True)
    #             )
    #         self.decoder = nn.Sequential(
    #                 nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
    #                 nn.ReLU(True),
    #                 nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1),
    #                 nn.ReLU(True)
    #             )

    #         self.decoder2 = nn.Sequential(
    #                 nn.ConvTranspose2d(32+64, 32, 3, stride=1, padding=1),
    #                 nn.ReLU(True),
    #                 nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1),
    #                 nn.ReLU(True)
    #             )
    #         self.decoder3 = nn.Sequential(
    #                 nn.ConvTranspose2d(32+32, 32, 3, stride=1, padding=1),
    #                 nn.ReLU(True),
    #                 nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1),
    #                 nn.ReLU(True),
    #                 nn.Upsample(scale_factor=2, mode='bicubic')
    #             )
    #         self.final = nn.Sequential(
    #                 nn.Conv2d(32+3, 32, 3, stride=1, padding=1),
    #                 nn.ReLU(inplace=True),
    #                 nn.Conv2d(32, 3, 3, stride=1, padding=1),
    #                 nn.ReLU(inplace=True)
    #             )

    #     def forward(self, x):
    #         e1 = self.encoder(x)
    #         e2 = self.encoder2(e1)
    #         e3 = self.encoder3(e2)
    #         d1 = self.decoder(e3)
    #         c1 = torch.cat((d1,e2), dim=1)
    #         d2 = self.decoder2(c1)
    #         c2 = torch.cat((d2,e1), dim=1)
    #         d3 = self.decoder3(c2)
    #         c3 = torch.cat((d3,x), dim=1)
    #         f = self.final(c3)
    #         return f