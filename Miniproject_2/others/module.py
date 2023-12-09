import torch
from torch import empty , cat , arange
from torch.nn.functional import fold , unfold
torch.set_grad_enabled(False)

class Module(object):
    def forward(self, *input):
        raise NotImplementedError
    #should get for input and returns, a tensor or a tuple of tensors.
    
    def backward(self, *gradwrtoutput): #
        raise NotImplementedError
    #should get as input a tensor or a tuple of tensors containing the gradient of the loss with respect to
    #the module’s output, accumulate the gradient wrt the parameters, and return a tensor or a tuple of tensors
    #containing the gradient of the loss wrt the module’s input.

    def param(self):
        return []
    #should return a list of pairs composed of a parameter tensor and a gradient tensor of the same size.
    #This list should be empty for parameterless modules (such as ReLU).




# =========================================== SEQUENTIAL =========================================== #

class Sequential(Module):
  
    def __init__(self, *layers_list):
        self.layers = layers_list

    def  forward(self, input):
        x = input
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, gradwrtoutput):
        new_grad = gradwrtoutput
        for layer in reversed(self.layers) :
            new_grad = layer.backward(new_grad)
        return new_grad
  
    def param(self):
        parameters_list = []
        for layer in self.layers:
                for parameter in layer.param():
                    parameters_list.append(parameter)
        return parameters_list



# ==================================== ACTIVATION FUNCTIONS ==================================== #


# ================= ReLU

class ReLU(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        self.t = input
        return input.clamp(0)

    def backward(self, gradwrtoutput): #for the derivative, if x < 0, output is 0. if x > 0, output is 1.
        input = self.t
        sign = input.sign().clamp(0)
        return sign * gradwrtoutput #.sign() gives 1 if positive, -1 of negative therefore if we again clamp it to zero we get 0 for the derivative if x is negative and x if it is positive
        
    def param(self):
        return []

    
# ================= Sigmoid

class Sigmoid(Module):
  
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.t = input
        x = 1 / (1 + self.t.mul(-1).exp())
        return x #input.mul(-1).exp().add(1).pow(-1)

    
    def backward(self, gradwrtoutput):
        grad = self.t.mul(-1).exp() / ((1 + self.t.mul(-1).exp())**2)
        return gradwrtoutput * grad

    def param(self):
        return []


# ==================================== OTHER LAYERS ==================================== #


# ================= 2d Convolution

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = (1,1), padding = 0, mean=0, std=1):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        
        
        self.stride = stride
        self.padding = padding
        
        self.weight = torch.empty(out_channels,in_channels, kernel_size, kernel_size).normal_(mean, std)
        self.bias = torch.empty(out_channels).normal_(mean, std)
        self.gradWeight = torch.empty(out_channels, in_channels, self.kernel_size, self.kernel_size).zero_()
        self.gradBias = torch.empty(out_channels).zero_()


    def forward(self, input):
        
        self.t = input
        self.batch_size = input.shape[0]
        self.height_in = input.shape[2]
        self.width_in = input.shape[3]
        self.C_in = input.shape[1]
        assert self.C_in == self.in_channels
        
        self.height_out = (self.height_in + 2*self.padding - self.kernel_size)//self.stride[0] + 1
        self.width_out = (self.width_in + 2*self.padding - self.kernel_size)//self.stride[1] + 1
        
        self.height_out_padded = self.height_out + 2*self.padding
        self.width_out_padded = self.width_out + 2*self.padding
        
        
        

        unfolded = torch.nn.functional.unfold(input , kernel_size = self.kernel_size, padding = self.padding, stride = self.stride)
        wxb = self.weight.view(self.out_channels , -1) @ unfolded + self.bias.view(1 , -1 , 1)
        self.out = wxb.view(self.batch_size , self.out_channels , self.height_out, self.width_out)
        
        
        return self.out
        
        
    def backward(self, gradwrtoutput):
        
        padded_input = torch.empty(self.batch_size, self.in_channels, self.height_in + 2*self.padding, self.width_in + 2*self.padding).zero_()
        padded_input[:, :, self.padding : self.height_in + self.padding, self.padding : self.width_in + self.padding] = self.t
        
                
        self.dt = torch.empty(self.batch_size, self.in_channels, self.height_in + 2*self.padding, self.width_in + 2*self.padding).zero_()

        
        for ch_out in range(self.out_channels):
            for H in range(self.height_out):
                for W in range(self.width_out):
                    
                    padded_input[:, :, H*self.stride[0] : H*self.stride[0]+self.kernel_size, W*self.stride[1] : W*self.stride[1]+self.kernel_size] += \
                    gradwrtoutput[:, ch_out, H, W].reshape(-1, 1, 1, 1) * \
                    self.weight[ch_out, :, 0:self.kernel_size, 0:self.kernel_size]
                    
                    
                    self.gradWeight[ch_out, : , 0:self.kernel_size, 0:self.kernel_size] += \
                    (padded_input[:, :, H*self.stride[0] : H*self.stride[0]+self.kernel_size, W*self.stride[1] : W*self.stride[1]+self.kernel_size] * \
                    gradwrtoutput[:, ch_out, H, W].reshape(-1, 1, 1, 1)).sum(axis=0)
                    
        self.dt = padded_input[:, :, self.padding : self.height_in + self.padding, self.padding : self.width_in + self.padding]
        
        self.gradBias = gradwrtoutput.sum(axis = (0, 2, 3))
        
        return self.dt, self.gradWeight, self.gradBias
        
    def param(self):
        return [(self.weight, self.gradWeight), (self.bias, self.gradBias)]


# ================= 2d Transposed Convolution

class TransposeConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = (1,1), padding = 0, mean=0, std=1):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        
        
        self.weight = torch.empty(in_channels,out_channels, kernel_size, kernel_size).normal_(mean, std)
        
        self.bias = torch.empty(out_channels).normal_(mean, std)
        
        self.gradWeight = torch.empty(in_channels,out_channels, kernel_size, kernel_size).zero_()
        self.gradBias = torch.empty(out_channels).zero_()
        
        
    def forward(self, input):
        self.t = input
        self.batch_size = input.shape[0]
        self.height_in = input.shape[2]
        self.width_in = input.shape[3]
        self.C_in = input.shape[1]
        assert self.C_in == self.in_channels
        
        self.height_out = (self.height_in-1)*self.stride[0] -2*self.padding +self.kernel_size
        self.width_out = (self.width_in-1)*self.stride[1] -2*self.padding +self.kernel_size
        self.height_out_padded = self.height_out + 2*self.padding
        self.width_out_padded = self.width_out + 2*self.padding
        
        y = torch.empty(self.batch_size, self.out_channels, self.height_out_padded, self.width_out_padded).zero_()
        
        for ch in range(self.in_channels):
            for H in range(self.height_in):
                for W in range(self.width_in):
                    
                    y[:, :, H*self.stride[0] : H*self.stride[0]+self.kernel_size, W*self.stride[1] : W*self.stride[1]+self.kernel_size] += \
                    self.t[:,ch,H,W].reshape(-1, 1, 1, 1)*self.weight[ch, :, 0:self.kernel_size, 0:self.kernel_size]
                    
        self.out = y[:, :, self.padding:self.height_out + self.padding, self.padding : self.width_out + self.padding]
        
        self.out += self.bias.view(1, -1, 1, 1)
            
        return self.out
    
    
    def backward(self, gradwrtoutput):
        
        self.dt = torch.empty(self.batch_size, self.in_channels, self.height_in, self.width_in).to(device)
        
        gradwrtoutput_padded = torch.empty(self.batch_size, self.out_channels, self.height_out_padded, self.width_out_padded).zero_()
        gradwrtoutput_padded[:, :, self.padding:self.height_out + self.padding, self.padding : self.width_out + self.padding] = gradwrtoutput
        
        for ch in range(self.in_channels):
            for H in range(self.height_in):
                for W in range(self.width_in):
                    
                    self.dt[:, ch, H, W] = \
                    (gradwrtoutput_padded[:, :, H*self.stride[0] : H*self.stride[0]+self.kernel_size, W*self.stride[1] : W*self.stride[1]+self.kernel_size] * \
                    self.weight[ch, :, 0:self.kernel_size, 0:self.kernel_size]).sum(axis = (1, 2, 3))
                    
                    
                    self.gradWeight[ch, :, 0:self.kernel_size, 0:self.kernel_size] += \
                    (gradwrtoutput_padded[:, :, H*self.stride[0] : H*self.stride[0]+self.kernel_size, W*self.stride[1] : W*self.stride[1]+self.kernel_size] * \
                    self.t[:, ch, H, W].reshape(-1, 1, 1, 1)).sum(axis = 0)
                    
                    self.gradBias = gradwrtoutput.sum(axis = (0, 2, 3))
        
        
        return self.dt, self.gradWeight, self.gradBias
    
    def param(self):
        return[(self.weight, self.gradWeight), (self.bias, self.gradBias)]


# ================= 2d Upsampling

class NearestUpsampling(Module):
    def __init__(self, in_channels, size = (1,1)):
        super().__init__()
        
        self.in_channels = in_channels
        self.size = size
        
    def forward(self, input):
        
        self.t = input
        self.batch_size = input.shape[0]
        self.height_in = input.shape[2]
        self.width_in = input.shape[3]
        self.C_in = input.shape[1]
        
        self.height_out = self.height_in*self.size[0]
        self.width_out = self.width_in*self.size[1]
        
        self.out = torch.empty(self.batch_size, self.in_channels, self.height_out, self.width_out).zero_()
        
        assert self.C_in == self.in_channels
        
        for ch in range(self.in_channels):
            for H in range(self.height_in):
                for W in range(self.width_in):
                    
                    self.out[:, ch, H*self.size[0]:(H+1)*self.size[0], W*self.size[1]:(W+1)*self.size[1]] = self.t[:,ch,H,W]
            
        return self.out
    
    
    def backward(self, gradwrtoutput):
        
        self.dt = torch.empty(self.batch_size, self.in_channels, self.height_in, self.width_in).zero_().to(device)
        
        for ch in range(self.in_channels):
            for H in range(self.height_in):
                for W in range(self.width_in):
                    
                    self.dt[:, ch, H, W] = gradwrtoutput[:, ch, H*self.size[0], W*self.size[1]]
        
        return self.dt
    
    def param(self):
        return[]



# ==================================== LOSS FUNCTION : MSE ==================================== #

class MSE(Module):
  
    def __init__(self):
        super().__init__()
      
    def __call__(self, prediction, target):
        return self.forward(prediction, target)
    
    def forward(self, prediction, target):
        self.pred = prediction
        self.targ = target
        return torch.mean((prediction - target)**2)

    def backward(self):
        return 2 * (self.pred - self.targ)

    def param(self):
        return []
    
    
    


# ==================================== OPTIMIZER : SGD ==================================== #

class SGD(Module):

            
            
    def __init__(self, model, learning_rate):
        self.model = model
        self.lr = learning_rate
       # loss = MSE()

    def step(self): 
        for module in self.model.param():
          param, grad = module
          if (param is not None) and (grad is not None):
              param.sub_(grad, alpha=self.lr)


    def zero_grad(self):
        for module in self.model.param():
            param, grad = module
            if (param is not None) and (grad is not None):
                grad.zero_()