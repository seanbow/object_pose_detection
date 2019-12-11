import torch
import torch.nn as nn

class convBlock(nn.Module):
    def __init__(self, numIn, numOut):
        super(convBlock, self).__init__()
        self.layer1 = nn.BatchNorm2d(numIn)
        self.layer2 = nn.ReLU(True)
        self.layer3 = nn.Conv2d(numIn,numOut//2,1)
        self.layer4 = nn.BatchNorm2d(numOut//2)
        self.layer5 = nn.ReLU(True)
        self.layer6 = nn.Conv2d(numOut//2,numOut//2,3,1,1)
        self.layer7 = nn.BatchNorm2d(numOut//2)
        self.layer8 = nn.ReLU(True)
        self.layer9 = nn.Conv2d(numOut//2,numOut,1)

    def forward(self,inp):
        x1 = self.layer1(inp)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)
        x8 = self.layer8(x7)
        out = self.layer9(x8)
        return out


class skipLayer(nn.Module):
    def __init__(self, numIn, numOut):
        super(skipLayer, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        if not (numIn == numOut):
            self.layer1 = nn.Conv2d(numIn,numOut,1)

    def forward(self,inp):
        if self.numIn == self.numOut:
            out = inp
        else:
            out = self.layer1(inp)
        return out


class Residual(nn.Module):
    def __init__(self, numIn, numOut):
        super(Residual, self).__init__()
        self.layer1 = convBlock(numIn, numOut)
        self.layer2 = skipLayer(numIn, numOut)

    def forward(self,inp):
        x1 = self.layer1(inp)
        x2 = self.layer2(inp)
        return x1+x2


class hourglass(nn.Module):
    def __init__(self, n, numIn, numOut):
        super(hourglass, self).__init__()
        self.n = n
        self.layer1 = Residual(numIn,256)
        self.layer2 = Residual(256,256)
        self.layer3 = Residual(256,numOut)
        self.layer4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer5 = Residual(numIn,256)
        self.layer6 = Residual(256,256)
        self.layer7 = Residual(256,256)
        if n > 1:
            self.layer8 = hourglass(n-1,256,numOut)
        else:
            self.layer8 = Residual(256,numOut)
        self.layer9 = Residual(numOut,numOut)
        self.layer10 = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self,inp):
        up1 = self.layer1(inp)
        up2 = self.layer2(up1)
        up4 = self.layer3(up2)
        pool = self.layer4(inp)
        low1 = self.layer5(pool)
        low2 = self.layer6(low1)
        low5 = self.layer7(low2)
        if self.n > 1:
            low6 = self.layer8(low5)
        else:
            low6 = self.layer8(low5)
        low7 = self.layer9(low6)
        up5 = self.layer10(low7)
        return up4+up5


class lin(nn.Module):
    def __init__(self, numIn, numOut):
        super(lin, self).__init__()
        self.layer1 = nn.Conv2d(numIn,numOut,1,1,0)
        self.layer2 = nn.BatchNorm2d(numOut)
        self.layer3 = nn.ReLU(True)

    def forward(self,inp):
        l_ = self.layer1(inp)
        return self.layer3(self.layer2(l_))


class CreateModel(nn.Module):
    def __init__(self):
        super(CreateModel, self).__init__()
        self.layer1 = nn.Conv2d(3,64,7,2,3)
        self.layer2 = nn.BatchNorm2d(64)
        self.layer3 = nn.ReLU(True)
        self.layer4 = Residual(64,128)
        self.layer5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer6 = Residual(128,128)
        self.layer7 = Residual(128,128)
        self.layer8 = Residual(128,256)
        self.layer9 = hourglass(4,256,512)
        self.layer10 = lin(512,512)
        self.layer11 = lin(512,256)
        self.layer12 = nn.Conv2d(256,54,1,1,0)
        self.layer13 = nn.Conv2d(54,256+128,1,1,0)
        self.layer14 = nn.Conv2d(256+128,256+128,1,1,0)
        self.layer15 = hourglass(4,256+128,512)
        self.layer16 = lin(512,512)
        self.layer17 = lin(512,512)
        self.layer18 = nn.Conv2d(512,54,1,1,0)

    def forward(self,inp):
        cnv1_ = self.layer1(inp)
        cnv1 = self.layer3(self.layer2(cnv1_))
        r1 = self.layer4(cnv1)
        pool = self.layer5(r1)
        r4 = self.layer6(pool)
        r5 = self.layer7(r4)
        r6 = self.layer8(r5)
        hg1 = self.layer9(r6)
        l1 = self.layer10(hg1)
        l2 = self.layer11(l1)
        out1 = self.layer12(l2)
        out1_ = self.layer13(out1)
        cat1 = torch.cat([l2,pool],dim=1)
        cat1_ = self.layer14(cat1)
        int1 = cat1_ + out1_
        hg2 = self.layer15(int1)
        l3 = self.layer16(hg2)
        l4 = self.layer17(l3)
        out2 = self.layer18(l4)
        return out1, out2
