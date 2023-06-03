import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------------------------------------------------------
# Generator according to the paper: https://arxiv.org/pdf/1703.10593.pdf
# Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (Jun-Yan Zhu et.al. 2017/2020)
# Use 9 resblocks
# c7s1-k denotes a 7×7 Convolution-InstanceNorm-ReLU with k filters and stride 1
# dk denotes a 3×3 Convolution-InstanceNorm-ReLU with k filters and stride 2
# Reflection padding was used to reduce artifacts
# Rk denotes a residual block that contains two 3×3 convolutional layers with the same number of filters on both layer
# uk denotes a 3×3 fractional-strided-ConvolutionInstanceNorm-ReLU layer with k filters and stride 1/2
# Use reflection padding to reduce artifacts:
# for Conv nn.ReflectionPad2d(3)
# for Res nn.ReflectionPad2d(1)
# The generator archetecture is: c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3
# --------------------------------------------------------------------------------------------------------------------------------


class Generator(nn.Module):
    '''
    Generator network with 9 residual blocks
    '''
    def __init__(self, num_res_blocks=9):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            ConvBlock(1, 64, kernel_size=7, stride=1, padding=3),  # c7s1-64
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),  # d128
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=1),  # d256
            *[ResBlock(256) for _ in range(num_res_blocks)],  # R256
            FractionalStridedConvBlock(256, 128),  # u128
            FractionalStridedConvBlock(128, 64),  # u64
            ConvBlock(64, 1, kernel_size=7, stride=1, activation=nn.Tanh(), padding=3)  # u64
        )

    def forward(self, x):
        '''Forward pass'''
        output = self.generator(x)
        # Resize output to match input
        if x.shape != output.shape:
            output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        return output


class ConvBlock(nn.Module):
    '''
    Convolutional for c7s1-k, and dk
    '''
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, activation=nn.ReLU(), **kwargs):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding_mode="reflect", **kwargs),
            nn.InstanceNorm2d(out_channels),
            activation
        )

    def forward(self, x):
        '''Forward pass'''
        return self.conv_block(x)


class ResBlock(nn.Module):
    '''
    Residual block for Rk
    '''
    def __init__(self, channels, kernel_size=3, stride=1):
        super(ResBlock, self).__init__()

        layers = []
        layers.append(nn.Conv2d(channels, channels, kernel_size, stride, padding_mode="reflect", padding=1)) 
        layers.append(nn.InstanceNorm2d(channels))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(channels, channels, kernel_size, stride, padding_mode="reflect", padding=1))
        layers.append(nn.InstanceNorm2d(channels))

        self.res_block = nn.Sequential(*layers)

    def forward(self, x):
        '''Forward pass'''
        return x + self.res_block(x)


class FractionalStridedConvBlock(nn.Module):
    '''
    Fractional strided convolutional block with optional normalization and activation. Increases the spatial resolution by a factor of 2
    '''
    def __init__(self, in_channels, out_channels):
        super(FractionalStridedConvBlock, self).__init__()

        layers = []
        layers += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)]
        layers += [nn.InstanceNorm2d(out_channels)]
        layers += [nn.ReLU()]

        self.fractional_block = nn.Sequential(*layers)

    def forward(self, x):
        '''Forward pass'''
        return self.fractional_block(x)