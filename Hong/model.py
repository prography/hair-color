import torch
import torch.nn as nn

class conv_start(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(conv_start, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3, stride= 2, padding=1)

    def forward(self,x):
        x=self.conv(x)
        return x

class encode_depthwise(nn.Module):
    def __init__(self,in_channels, out_channels,stride):
        super(encode_depthwise,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3, stride, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,out_channels,1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self,x):
        x= self.conv(x)
        return x

class decode_depthwise(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(decode_depthwise,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, padding= 1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels,1, 1, padding=0, groups=1),
            nn.ReLU()
        )

    def forward(self,x):
        x= self.conv(x)
        return x


class up_skip(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(up_skip, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)  # stride랑 padding 값 궁금??

    def forward(self,x1,x2):
        x1 = self.up(x1)
        x2 = self.conv(x2)
        x = torch.add(x1, x2)
        return x





class conv_softmax(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(conv_softmax, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=1),

        )

    def forward(self, x):
            x = self.conv(x)
            return x


class Mobilehair(nn.Module):
    def __init__(self):
        super(Mobilehair, self).__init__()


        self.inc = conv_start(3,32)
        self.edw1 = encode_depthwise(32, 64, 1)
        self.edw2 = encode_depthwise(64, 128, 2)
        self.edw3 = encode_depthwise(128, 128, 1)
        self.edw4 = encode_depthwise(128, 256, 2)
        self.edw5 = encode_depthwise(256, 256, 1)
        self.edw6 = encode_depthwise(256, 512, 2)
        self.edw7 = encode_depthwise(512, 512, 1)
        self.edw8 = encode_depthwise(512, 512, 1)
        self.edw9 = encode_depthwise(512, 512, 1)
        self.edw10 = encode_depthwise(512, 512, 1)
        self.edw11 = encode_depthwise(512, 512, 1)
        self.edw12 = encode_depthwise(512, 1024, 2)
        self.edw13 = encode_depthwise(1024, 1024, 1)


        self.up1 = up_skip(512,1024)
        self.ddw1 = decode_depthwise(1024, 64)
        self.up2 = up_skip(256,64)
        self.ddw2 = decode_depthwise(64,64)
        self.up3 = up_skip(128,64)
        self.ddw3 = decode_depthwise(64, 64)
        self.up4 = up_skip(64,64)
        self.ddw4 = decode_depthwise(64, 64)
        self.up5 = nn.Upsample(scale_factor=2)
        self.ddw5 = decode_depthwise(64, 64)
        self.lac = conv_softmax(64,2)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.edw1(x1)
        x3 = self.edw2(x2)
        x4 = self.edw3(x3)
        x5 = self.edw4(x4)
        x6 = self.edw5(x5)
        x7 = self.edw6(x6)
        x8 = self.edw7(x7)
        x9 = self.edw8(x8)
        x10 = self.edw9(x9)
        x11 = self.edw10(x10)
        x12 = self.edw11(x11)
        x13 = self.edw12(x12)
        x14 = self.edw13(x13)

        x = self.up1(x14,x12)
        x = self.ddw1(x)
        x = self.up2(x,x6)
        x = self.ddw2(x)
        x = self.up3(x,x4)
        x = self.ddw3(x)
        x = self.up4(x,x2)
        x = self.ddw4(x)
        x = self.up5(x)
        x = self.ddw5(x)
        x = self.lac(x)


        return x









