import torch
import torch.nn as nn
import torch.utils.data


class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1, upsample=False):
        super(ConvBlock, self).__init__()
        self.upsample = upsample

        if (upsample):
            self.upsample = nn.Upsample(scale_factor=2)
            self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=1)
        else:
            self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride)

        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.1)

        self.reflection = nn.ReflectionPad2d(int((kernel_size-1)/2))
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        
        if (self.upsample):
            out = self.upsample(out)

        out = self.reflection(out)
        out = self.conv(out)
            
        return out
    

class deconv_block(nn.Module):
    def __init__(self):
        super(deconv_block, self).__init__()
        self.A01 = ConvBlock(2, 64, kernel_size=7)
        
        self.C11 = ConvBlock(64, 64, stride=2)
        self.C12 = ConvBlock(64, 64)
        self.C13 = ConvBlock(64, 64)
        self.C14 = ConvBlock(64, 64)
        
        self.C21 = ConvBlock(64, 64)
        self.C22 = ConvBlock(64, 64)
        self.C23 = ConvBlock(64, 64)
        self.C24 = ConvBlock(64, 64)
        
        self.C31 = ConvBlock(64, 128, stride=2)
        self.C32 = ConvBlock(128, 128)
        self.C33 = ConvBlock(128, 128)
        self.C34 = ConvBlock(128, 128)
        
        self.C41 = ConvBlock(128, 256, stride=2)
        self.C43 = ConvBlock(256, 256)
        self.C44 = ConvBlock(256, 256)
        self.C45 = ConvBlock(256, 256)
        
        self.C51 = ConvBlock(256, 128, upsample=True)
        self.C53 = ConvBlock(128, 128)
        self.C54 = ConvBlock(128, 128)
        self.C55 = ConvBlock(128, 128)
        
        self.C61 = ConvBlock(128, 64, upsample=True)
        self.C63 = ConvBlock(64, 64)
        self.C64 = ConvBlock(64, 64)
        self.C65 = ConvBlock(64, 64)
        
        self.C71 = ConvBlock(64, 64, upsample=True)
        self.C72 = ConvBlock(64, 8)
        
        self.C73 = nn.Conv2d(8, 1, kernel_size=1)
        nn.init.kaiming_normal_(self.C73.weight)
        nn.init.constant_(self.C73.bias, 0.1)
        
        self.B42 = nn.Conv2d(512, 256, kernel_size=1)
        nn.init.kaiming_normal_(self.B42.weight)
        nn.init.constant_(self.B42.bias, 0.1)

        self.B52 = nn.Conv2d(256, 128, kernel_size=1)
        nn.init.kaiming_normal_(self.B52.weight)
        nn.init.constant_(self.B52.bias, 0.1)

        self.B62 = nn.Conv2d(128, 64, kernel_size=1)
        nn.init.kaiming_normal_(self.B62.weight)
        nn.init.constant_(self.B62.bias, 0.1)
        
    def forward(self, current, new, previous1=None, previous2=None, previous3=None):
        x = torch.cat([current, new], dim=1)

        A01 = self.A01(x)
        
        C11 = self.C11(A01)
        C12 = self.C12(C11)
        C13 = self.C13(C12)
        C14 = self.C14(C13)
        C14 += C11
        
        C21 = self.C21(C14)
        C22 = self.C22(C21)
        C23 = self.C23(C22)
        C24 = self.C24(C23)
        C24 += C21
        
        C31 = self.C31(C24)
        C32 = self.C32(C31)
        C33 = self.C33(C32)
        C34 = self.C34(C33)
        C34 += C31
        
        C41 = self.C41(C34)
        if (previous1 is None):
            B42 = C41
        else:
            B42 = self.B42(torch.cat([C41, previous1], dim=1))
        C43 = self.C43(B42)
        C44 = self.C44(C43)
        C45 = self.C45(C44)
        C45 += C41
        
        C51 = C34 + self.C51(C45)
        if (previous1 is None):
            B52 = C51
        else:
            B52 = self.B52(torch.cat([C51, previous2], dim=1))
        C53 = self.C53(B52)
        C54 = self.C54(C53)
        C55 = self.C55(C54)
        C55 += C51
        
        C61 = C24 + self.C61(C55)
        if (previous1 is None):
            B62 = C61
        else:
            B62 = self.B62(torch.cat([C61, previous3], dim=1))
        C63 = self.C63(B62)
        C64 = self.C64(C63)
        C65 = self.C65(C64)
        C65 += C61
        
        C71 = self.C71(C65)
        C72 = self.C72(C71)
        out = current + self.C73(C72)
        
        return out, C43, C53, C63
    

class deconvolution_network(nn.Module):
    def __init__(self, n_blocks):
        super(deconvolution_network, self).__init__()

        self.block = deconv_block()
        self.n_blocks = n_blocks
        
    def forward(self, x):
        previous1 = None
        previous2 = None
        previous3 = None
        out = [None] * self.n_blocks
        current = x[:, 0:1, :, :]
        for i in range(self.n_blocks):
            out[i], previous1, previous2, previous3 = self.block(current, x[:, i+1:i+2, :, :], previous1, previous2, previous3)
            current = out[i]
        
        return out