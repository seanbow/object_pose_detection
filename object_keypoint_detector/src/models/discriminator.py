import torch.nn as nn
from .layers import ConvBlock

class PatchDiscriminator(nn.Module):

    def __init__(self, in_channels=1, num_filters=64, num_layers=5, activation=None):
        super(PatchDiscriminator, self).__init__()
        layers = [ConvBlock(in_channels, num_filters, kernel_size=4, stride=2, padding=1,
                            activation=nn.LeakyReLU(negative_slope=0.2), batchnorm=False)]
        for i in range(0,num_layers-2):
            layers.append(ConvBlock(num_filters * 2**i, num_filters * 2**(i+1), kernel_size=4, stride=2, padding=1,
                          activation=nn.LeakyReLU(negative_slope=0.2), batchnorm=True))
        layers.append(ConvBlock(num_filters * 2**(num_layers-2), 1, kernel_size=4, stride=2, padding=1,
                      activation=activation, batchnorm=False))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


        
