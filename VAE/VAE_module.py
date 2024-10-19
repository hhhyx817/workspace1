'''
use VAE to generate image 
data from: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
'''

import torch
import torch.nn as nn

# 组成：核心部分 __init__， 前向过程，(sample); DDPM.module 也是这么写的
class VAE(nn.Module):
    """VAE for 64x64 face generation.

    The hidden dimensions can be tuned.
    """

    # __init__中放的都是核心参数和方法
    def __init__(self, hiddens=[16, 32, 64, 128, 256], latent_dim=128) -> None:
        super().__init__()

        # class中的 encoder 和 decoder 都是没有具体输入对象的，只体现方法。 对象都在 forward 传入并产生结果
        # encoder
        prev_channels = 3 # the image is of RGB mode
        modules = []
        img_length = 64
        
        # the main part of hidden layer(composed by Conv2d and ...), contract from 16 to 256
        for cur_channels in hiddens:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels,
                              cur_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1), 
                    nn.BatchNorm2d(cur_channels),
                    nn.ReLU()))
            prev_channels = cur_channels
            img_length //= 2
            
        self.encoder = nn.Sequential(*modules) # the output of hidden layer
        self.mean_linear = nn.Linear(prev_channels * img_length * img_length, latent_dim) # the parameters not defined yet
        self.var_linear = nn.Linear(prev_channels * img_length * img_length, latent_dim)
        self.latent_dim = latent_dim
        
          
       
        # decoder
        modules = []
        self.decoder_projection = nn.Linear(latent_dim, prev_channels * img_length * img_length)
        self.decoder_input_chw = (prev_channels, img_length, img_length) # here 526, 64, 64 指定输入形状
        
         # decode from 256 to 16
        for i in range(len(hiddens) - 1, 0, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hiddens[i],
                                       hiddens[i - 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hiddens[i - 1]), nn.ReLU()))
            
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hiddens[0],
                                   hiddens[0],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hiddens[0]), nn.ReLU(),
                nn.Conv2d(hiddens[0], 3, kernel_size=3, stride=1, padding=1),
                nn.ReLU()))
        self.decoder = nn.Sequential(*modules)

    # 完成一轮从 image encode到 vector 到 decode image 的过程
    def forward(self, x):
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, 1)
        mean = self.mean_linear(encoded)
        logvar = self.var_linear(encoded)
        eps = torch.randn_like(logvar)
        std = torch.exp(logvar / 2)
        z = eps * std + mean
        z_1 = self.decoder_projection(z)
        x_1 = torch.reshape(z_1, (-1, *self.decoder_input_chw))
        decoded = self.decoder(x_1)

        return decoded, mean, logvar

    def sample(self, device='cuda'):
        z = torch.randn(1, self.latent_dim).to(device)
        x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        decoded = self.decoder(x)
        return decoded



    
    
      
        
  
      