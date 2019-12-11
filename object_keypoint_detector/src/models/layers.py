import torch.nn as nn

# Wrapper around Conv2d
class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(ConvBlock, self).__init__()
        modules = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)]
        if batchnorm:
            modules.append(nn.BatchNorm2d(out_channels))
        if activation is not None:
            modules.append(activation)
        if dropout:
            modules.append(dropout)
        self.conv_block = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.conv_block(x)

# Wrapper around ConvTranspose2d
class ConvTransposeBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, output_padding=0, batchnorm=True, activation=nn.ReLU(inplace=True), dropout=False):
        super(ConvTransposeBlock, self).__init__()
        modules = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        if batchnorm:
            modules.append(nn.BatchNorm2d(out_channels))
        if activation is not None:
            modules.append(activation)
        if dropout:
            modules.append(dropout)
        self.conv_block = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.conv_block(x)


# Bottleneck Residual Block
class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_block = nn.Sequential(
                          nn.BatchNorm2d(in_channels),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(in_channels, out_channels//2, kernel_size=1, padding=0),
                          nn.BatchNorm2d(out_channels//2),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(out_channels//2, out_channels//2, kernel_size=3, padding=1),
                          nn.BatchNorm2d(out_channels//2),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(out_channels//2, out_channels, kernel_size=1, padding=0)
                          )
        if out_channels != in_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)


    def forward(self,x):
        if self.in_channels == self.out_channels:
            return x + self.conv_block(x)
        else:
            return self.skip_conv(x) + self.conv_block(x)
