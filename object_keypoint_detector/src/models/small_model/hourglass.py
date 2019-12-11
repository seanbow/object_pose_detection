import torch.nn as nn
import numpy as np
from .layers import ConvBlock, ResBlock

# Stacked Hourglass Module
# Naming scheme and architecture is based on the Torch implementation in https://github.com/umich-vl/pose-hg-train
class StackedHourglass(nn.Module):
    
    def __init__(self, in_channels=3, hg_channels=256, out_channels=15, num_hg=2, num_blocks=1):
        super(StackedHourglass, self).__init__()
        self.in_channels = in_channels
        self.hg_channels = hg_channels
        self.out_channels = out_channels
        self.num_hg = num_hg

        self.pre_hg = nn.Sequential(
                      ConvBlock(in_channels, 64, kernel_size=7, stride=2, padding=3),
                      ResBlock(64,128),
                      nn.MaxPool2d(kernel_size=2),
                      ResBlock(128,128),
                      ResBlock(128,hg_channels)
                      )
        self.stacked_hg = nn.ModuleList([Hourglass(num_channels=hg_channels, pred_channels=hg_channels, num_blocks=num_blocks, n=4) for _ in range(num_hg)])
        self.resblock_out = nn.ModuleList([nn.Sequential(*[ResBlock(hg_channels, hg_channels) for _ in range(num_blocks)]) for _ in range(num_hg)])
        self.lin_out1 = nn.ModuleList([ConvBlock(hg_channels, hg_channels, kernel_size=1) for _ in range(num_hg)])
        self.lin_pred = nn.ModuleList([ConvBlock(hg_channels, out_channels, kernel_size=1, batchnorm=False, activation=None) for _ in range(num_hg)])
        self.lin_out2 = nn.ModuleList([ConvBlock(hg_channels, hg_channels, kernel_size=1, batchnorm=False, activation=None) for _ in range(num_hg-1)])
        self.lin_out3 = nn.ModuleList([ConvBlock(out_channels, hg_channels, kernel_size=1, batchnorm=False, activation=None) for _ in range(num_hg-1)])
        return

    def num_trainable_parameters(self):
        trainable_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in trainable_parameters])

    def forward(self,x):
        inter_out = []
        inter = self.pre_hg(x)
        
        for i in range(self.num_hg):
            ll = self.stacked_hg[i](inter)
            ll = self.resblock_out[i](ll)
            ll = self.lin_out1[i](ll)
            tmp_out = self.lin_pred[i](ll)
            inter_out.append(tmp_out)
            if i < self.num_hg-1:
                ll_ = self.lin_out2[i](ll)
                tmp_out_ = self.lin_out3[i](tmp_out)
                inter = inter + ll_ + tmp_out_
        return inter_out

# Hourglass Module
# Naming scheme and architecture is based on the Torch implementation in https://github.com/umich-vl/pose-hg-train
class Hourglass(nn.Module):

    def __init__(self, num_channels=256, pred_channels=15, num_blocks=1, n=4):
        super(Hourglass, self).__init__()
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.n = n

        self.resblock_up1 = nn.Sequential(*[ResBlock(num_channels, num_channels) for _ in range(num_blocks)])
        self.resblock_low1 = nn.Sequential(*[ResBlock(num_channels, num_channels) for _ in range(num_blocks)])
        self.resblock_low3 = nn.Sequential(*[ResBlock(num_channels, num_channels) for _ in range(num_blocks)])
        if self.n > 1:
            self.hg_nm1 = Hourglass(num_channels=num_channels, pred_channels=pred_channels, num_blocks=num_blocks, n=n-1)
        else:
            self.resblock_low2 = nn.Sequential(*[ResBlock(num_channels, num_channels) for _ in range(num_blocks)])
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        return
    
    def forward(self, x):
        up1 = self.resblock_up1(x)
        
        low1 = self.resblock_low1(self.maxpool(x))

        if self.n > 1:
            low2 = self.hg_nm1(low1)
        else:
            low2 = self.resblock_low2(low1)

        low3 = self.resblock_low3(low2)
        up2 = self.upsample(low3)
        return up1+up2
