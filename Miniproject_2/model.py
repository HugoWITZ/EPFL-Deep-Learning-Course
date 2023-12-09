# import torch.nn as nn
# from torch.nn import functional as F
# import torch.optim as optim
import torch
# from torch.utils.data import DataLoader, Dataset

from pathlib import Path
from .others.module import Conv2d, TransposeConv2d, ReLU, Sigmoid, MSE, SGD, Sequential
from .others.helpers import psnr
import pickle



class Model:

    def __init__(self) -> None:
        super().__init__()

        # Parameters to ptimize
        self.batch_size, self.nb_epochs = 100, 10

        # Instantiate model
        self.sequence = Sequential(
            Conv2d(3, 3),
            ReLU(),
            Conv2d(3, 48),
            ReLU(),
            TransposeConv2d(48, 3),
            ReLU(),
            TransposeConv2d(3, 3),
            Sigmoid()
        )

        # Select optimizer
        self.optimizer = SGD(self, learning_rate=1e-2)

        # Select loss function
        self.loss = MSE()

        # Define parameters
        self.param = self.sequence.param

        # Select device
        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    def train(self, train_input, train_target):

        torch.set_grad_enabled(False)
        losses = []

        # convert input images to float and divide by 255.0
        train_input = train_target.float()/255.0
        train_input = train_target.float()/255.0

        for epoch in range(self.nb_epochs):

            ind = torch.randperm(len(train_input))
            train_input = train_input[ind]
            train_target = train_target[ind]
            
            # load images by batch of size batch_size
            for batch_in, batch_target in zip(train_input.split(self.batch_size), train_target.split(self.batch_size)):

                self.optimizer.zero_grad()
                batch_out = self.predict(batch_in)

                loss = self.loss(batch_out, batch_target)
                self.loss.backward()
                self.optimizer.step()

            losses.append(loss)

            print('Epoch: {}, training loss: {:.6f}'.format(epoch+1, loss))
        
        # # To save model parameters
        # params = []
        # for l in self.sequence.layers:
        #     params.append(l.param())
        # pickle.dump(params, open("bestmodel.pth", "wb"))
        

        return losses


    def load_pretrained_model(self) -> None:
    #   '''This loads the parameters saved in bestmodel.pth into the model'''
    #   model_path = Path(__file__).parent / "bestmodel.pth"
    #   with open(model_path, "rb") as bestmodel:
    #     self.Model.load(pickle.load(bestmodel))

        model_path = Path(__file__).parent / "bestmodel.pth"
        parameters = pickle.load(open( model_path, "rb" ))  #"wb"
        for i, param in enumerate(parameters):
            self.sequence.layers[i].weight = param[i][0]
            self.sequence.layers[i].bias = param[i+1][0]

        # for i, param in enumerate(parameters):
        #   if len(param) > 0:
        #     self.sequence.layers[i].weight = param[0][0]
        #     self.sequence.layers[i].bias = param[1][0]


    def predict(self, test_input) -> torch.Tensor:

        output = self.sequence.forward(test_input)
        return output