import torch
import torch.nn as nn


class in_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(in_conv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_dw(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super(conv_dw, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_add(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_add, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.conv(x2)
        x = torch.add(x1, x2)
        return x


class up_only(nn.Module):
    def __init__(self):
        super(up_only, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.up(x)
        return x


class inverse_conv_dw(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inverse_conv_dw, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class out_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(out_conv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class MobileHairNet(nn.Module):
    def __init__(self):
        super(MobileHairNet, self).__init__()

        self.inc = in_conv(3, 32)

        self.dw1 = conv_dw(32, 64, 1)
        self.dw2 = conv_dw(64, 128, 2)
        self.dw3 = conv_dw(128, 128, 1)
        self.dw4 = conv_dw(128, 256, 2)
        self.dw5 = conv_dw(256, 256, 1)
        self.dw6 = conv_dw(256, 512, 2)
        self.dw7 = conv_dw(512, 512, 1)
        self.dw8 = conv_dw(512, 512, 1)
        self.dw9 = conv_dw(512, 512, 1)
        self.dw10 = conv_dw(512, 512, 1)
        self.dw11 = conv_dw(512, 512, 1)
        self.dw12 = conv_dw(512, 1024, 2)
        self.dw13 = conv_dw(1024, 1024, 1)

        self.up1 = up_add(512, 1024)
        self.inv_dw1 = inverse_conv_dw(1024, 64)
        self.up2 = up_add(256, 64)
        self.inv_dw2 = inverse_conv_dw(64, 64)
        self.up3 = up_add(128, 64)
        self.inv_dw3 = inverse_conv_dw(64, 64)
        self.up4 = up_add(64, 64)
        self.inv_dw4 = inverse_conv_dw(64, 64)
        self.up5 = up_only()
        self.inv_dw5 = inverse_conv_dw(64, 64)

        self.outc = out_conv(64, 2)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.dw1(x1)
        x3 = self.dw2(x2)
        x4 = self.dw3(x3)
        x5 = self.dw4(x4)
        x6 = self.dw5(x5)
        x7 = self.dw6(x6)
        x8 = self.dw7(x7)
        x9 = self.dw8(x8)
        x10 = self.dw9(x9)
        x11 = self.dw10(x10)
        x12 = self.dw11(x11)
        x13 = self.dw12(x12)
        x14 = self.dw13(x13)

        x = self.up1(x14, x12)
        x = self.inv_dw1(x)
        x = self.up2(x, x6)
        x = self.inv_dw2(x)
        x = self.up3(x, x4)
        x = self.inv_dw3(x)
        x = self.up4(x, x2)
        x = self.inv_dw4(x)
        x = self.up5(x)
        x = self.inv_dw5(x)
        x = self.outc(x)

        return x
