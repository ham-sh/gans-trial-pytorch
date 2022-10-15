import torch
import torch.nn as nn

# 引用先: http://cedro3.com/ai/pytorch-conditional-gan/


# 重みの初期化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:            
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:        
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:    
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# 生成器
class Generator(nn.Module):
    def __init__(self, nz=100, nch_g=64, nch=3):
        super(Generator, self).__init__()
        self.layers = nn.ModuleDict({
            'layer0': nn.Sequential(
                nn.ConvTranspose2d(nz, nch_g * 8, 4, 1, 0),     
                nn.BatchNorm2d(nch_g * 8),                      
                nn.ReLU()                                       
            ),  # (100, 1, 1) -> (512, 4, 4)
            'layer1': nn.Sequential(
                nn.ConvTranspose2d(nch_g * 8, nch_g * 4, 4, 2, 1),
                nn.BatchNorm2d(nch_g * 4),
                nn.ReLU()
            ),  # (512, 4, 4) -> (256, 8, 8)
            'layer2': nn.Sequential(
                nn.ConvTranspose2d(nch_g * 4, nch_g * 2, 4, 2, 1),
                nn.BatchNorm2d(nch_g * 2),
                nn.ReLU()
            ),  # (256, 8, 8) -> (128, 16, 16)
 
            'layer3': nn.Sequential(
                nn.ConvTranspose2d(nch_g * 2, nch_g, 4, 2, 1),
                nn.BatchNorm2d(nch_g),
                nn.ReLU()
            ),  # (128, 16, 16) -> (64, 32, 32)
            'layer4': nn.Sequential(
                nn.ConvTranspose2d(nch_g, nch, 4, 2, 1),
                nn.Tanh()
            )   # (64, 32, 32) -> (3, 64, 64)
        })
 
    def forward(self, z):
        for layer in self.layers.values():  
            z = layer(z)
        return z


# 識別器
class Discriminator(nn.Module):
    def __init__(self, nch=3, nch_d=64, dropout=0.5):
        super(Discriminator, self).__init__()
        self.layers = nn.ModuleDict({
            'layer0': nn.Sequential(
                nn.Conv2d(nch, nch_d, 4, 2, 1),     
                nn.LeakyReLU(negative_slope=0.2)    
            ),  # (3, 64, 64) -> (64, 32, 32)
            'layer1': nn.Sequential(
                nn.Conv2d(nch_d, nch_d * 2, 4, 2, 1),
                nn.BatchNorm2d(nch_d * 2),
                nn.LeakyReLU(negative_slope=0.2)
            ),  # (64, 32, 32) -> (128, 16, 16)
            'layer2': nn.Sequential(
                nn.Conv2d(nch_d * 2, nch_d * 4, 4, 2, 1),
                nn.BatchNorm2d(nch_d * 4),
                nn.LeakyReLU(negative_slope=0.2)
            ),  # (128, 16, 16) -> (256, 8, 8)
            'layer3': nn.Sequential(
                nn.Conv2d(nch_d * 4, nch_d * 8, 4, 2, 1),
                nn.BatchNorm2d(nch_d * 8),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(dropout)
            ),  # (256, 8, 8) -> (512, 4, 4)
            'layer4': nn.Sequential(
                nn.Conv2d(nch_d * 8, 1, 4, 1, 0),
                # nn.Sigmoid()
                # (512, 4, 4) -> (1, 1, 1)
            ),
        })
 
    def forward(self, x):
        for layer in self.layers.values():  
            x = layer(x)
        return x.squeeze()
