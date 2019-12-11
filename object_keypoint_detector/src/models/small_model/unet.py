import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import ConvBlock, ConvTransposeBlock


# U-net Module with transpose convolutions
# Beware of the artifacts! https://distill.pub/2016/deconv-checkerboard/
class UNet(nn.Module):

    def __init__(self, in_channels, num_filters=64, num_blocks=5):
        super(UNet, self).__init__()
        self.in_channels = 3
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.conv1 = ConvBlock(in_channels, num_filters, kernel_size=7, stride=1, padding=3)
        self.down_path = nn.ModuleList([ConvBlock(num_filters * (2**i), num_filters * (2**(i+1)), kernel_size=4, stride=2, padding=1) for i in range(num_blocks)])
        self.bottleneck_conv = ConvTransposeBlock(num_filters * (2**num_blocks), num_filters * (2**(num_blocks-1)), kernel_size=4, stride=2, padding=1, output_padding=0)
        self.up_path = nn.ModuleList([ConvTransposeBlock(num_filters * (2*2**(i+1)), num_filters * (2**(i)), kernel_size=4, stride=2, padding=1, output_padding=0)\
                                      for i in range(num_blocks-1)])
        self.out_conv1 = ConvBlock(2*num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.out_conv2 = ConvBlock(num_filters, 1, kernel_size=3, stride=1, padding=1, batchnorm=False, activation=nn.Sigmoid())

    def num_trainable_parameters(self):
        trainable_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in trainable_parameters])
        
    def forward(self, x):
        x = self.conv1(x)
        skip_blocks = []
        for i in range(self.num_blocks):
            skip_blocks.append(x)
            x = self.down_path[i](x)
        x = self.bottleneck_conv(x)
        for i in reversed(range(self.num_blocks-1)):
            x = self.up_path[i](torch.cat((x, skip_blocks[i+1]), dim=1))
        return self.out_conv2(self.out_conv1(torch.cat((x, skip_blocks[0]), dim=1)))

# U-net Module with Bilinear upsampling instead of transpose convolutions
class UNetv2(nn.Module):

    def __init__(self, in_channels, num_filters=64, num_blocks=5):
        super(UNetv2, self).__init__()
        self.in_channels = 3
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.conv1 = ConvBlock(in_channels, num_filters, kernel_size=7, stride=1, padding=3)
        self.down_path = nn.ModuleList([ConvBlock(num_filters * (2**i), num_filters * (2**(i+1)), kernel_size=3, stride=2, padding=1) for i in range(num_blocks)])
        self.bottleneck_conv = ConvBlock(num_filters * (2**num_blocks), num_filters * (2**(num_blocks-1)), kernel_size=3, stride=1, padding=1)
        self.up_path = nn.ModuleList([ConvBlock(num_filters * (2*2**(i+1)), num_filters * (2**(i)), kernel_size=3, stride=1, padding=1) for i in range(num_blocks-1)])
        self.out_conv1 = ConvBlock(2*num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.out_conv2 = ConvBlock(num_filters, 1, kernel_size=3, stride=1, padding=1, batchnorm=False, activation=nn.Sigmoid())

    def num_trainable_parameters(self):
        trainable_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in trainable_parameters])
        
    def forward(self, x):
        x = self.conv1(x)
        skip_blocks = []
        for i in range(self.num_blocks):
            skip_blocks.append(x)
            x = self.down_path[i](x)
        x = F.upsample(self.bottleneck_conv(x), scale_factor=2, mode='bilinear')
        for i in reversed(range(self.num_blocks-1)):
            x = F.upsample(self.up_path[i](torch.cat((x, skip_blocks[i+1]), dim=1)), scale_factor=2, mode='bilinear')
        return self.out_conv2(self.out_conv1(torch.cat((x, skip_blocks[0]), dim=1)))
