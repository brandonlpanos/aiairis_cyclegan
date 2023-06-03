import torch
import torch.nn as nn

#-----------------------------------------------------------------------------------------------------------------------------
# Discriminator according to the paper: https://arxiv.org/pdf/1703.10593.pdf
# Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (Jun-Yan Zhu et.al. 2017/2020)
# Use 70 × 70 PatchGAN. 
# Let Ck denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2. 
# After the last layer, apply a convolution to produce a 1-dimensional output. 
# Do not use InstanceNorm for the first C64 layer. 
# Use leaky ReLUs with a slope of 0.2. 
# The discriminator architecture is: C64-C128-C256-C512
#-----------------------------------------------------------------------------------------------------------------------------

class Discriminator(torch.nn.Module):
    '''
    Discriminator network for PatchGAN
    Outputs (n x m) where each value represents the probability that the patch is real. Each pixel in the output represents a 70 by 70 patch in the input image.
    These patches overlap. 
    '''
    def __init__(self, in_channels=1, num_filters=64, num_layers=3):
        super(Discriminator, self).__init__()

        layers = []

        # C64
        layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1, padding_mode="reflect"))
        layers += [nn.LeakyReLU(0.2)]

        # C128-C256-C512
        for _ in range(num_layers-1):
            layers.append(nn.Conv2d(num_filters, num_filters*2, kernel_size=4, stride=2, padding=1, padding_mode="reflect"))
            layers += [nn.InstanceNorm2d(num_filters*2)]
            layers += [nn.LeakyReLU(0.2)]
            num_filters *= 2

        # output 1 channel prediction map
        layers.append(nn.Conv2d(num_filters, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))

        self.discriminator = torch.nn.Sequential(*layers)

    def forward(self, x):
        '''Forward pass'''
        return self.discriminator(x)