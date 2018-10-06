import torch
import torch.nn as nn
from config import get_config

config = get_config()

class _Layer_Depwise_Encode(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3, reserve=False): #nf==64
        self.stride = int(out_channels/in_channels)
        if reserve == True:
            self.stride = 1
        super(_Layer_Depwise_Encode, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=self.stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        out = self.layer(x)
        return out

class _Layer_Depwise_Decode(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3,  stride=1):
        super(_Layer_Depwise_Decode, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=kernel_size, stride=stride, padding=1),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=1),
            nn.ReLU6(inplace=True)
        )
    def forward(self, x):
        out = self.layer(x)
        return out

class _EncoderBlock(nn.Module):
    def __init__(self, im_size=224, nf=32, kernel_size=3):
        super(_EncoderBlock, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=nf, kernel_size=kernel_size, stride=2, padding=1)
        self.layer2 = nn.Sequential(
            _Layer_Depwise_Encode(nf, 2*nf, reserve=True),
            _Layer_Depwise_Encode(2*nf, 4*nf),
            _Layer_Depwise_Encode(4*nf, 8*nf),
            _Layer_Depwise_Encode(8*nf, 8*nf),
            _Layer_Depwise_Encode(8*nf, 16*nf),
            _Layer_Depwise_Encode(16*nf, 16*nf),
            _Layer_Depwise_Encode(16*nf, 16*nf),
            _Layer_Depwise_Encode(16*nf, 16*nf),
            _Layer_Depwise_Encode(16*nf, 16*nf),
            _Layer_Depwise_Encode(16*nf, 16*nf),
            _Layer_Depwise_Encode(16*nf, 32*nf),
            _Layer_Depwise_Encode(32*nf, 32*nf),
            _Layer_Depwise_Encode(32*nf, 32*nf)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        print(out.shape)
        return out

class _Decoderblock(nn.Module):
    def __init__(self, in_size = 7,nf = 32, kernel_size=3):
        super(_Decoderblock, self).__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=32*nf, out_channels=2*nf, kernel_size=1),
            _Layer_Depwise_Decode(in_channel=2*nf, out_channel=2*nf, kernel_size=kernel_size),
            nn.Upsample(scale_factor=2),
            _Layer_Depwise_Decode(in_channel=2*nf, out_channel=2*nf, kernel_size=kernel_size),
            nn.Upsample(scale_factor=2),
            _Layer_Depwise_Decode(in_channel=2*nf, out_channel=2*nf, kernel_size=kernel_size),
            nn.Upsample(scale_factor=2),
            _Layer_Depwise_Decode(in_channel=2*nf, out_channel=2*nf, kernel_size=kernel_size),
            nn.Upsample(scale_factor=2),
            _Layer_Depwise_Decode(in_channel=2*nf, out_channel=2*nf, kernel_size=kernel_size),
        )
    def forward(self, x):
        out = self.layer(x)
        return out

class MovileMatNet(nn.Module):
    def __init__(self, im_size, nf):
        super(MovileMatNet, self).__init__()
        self.layer = nn.Sequential(_EncoderBlock(im_size=im_size, nf=nf, kernel_size=3),
                                   _Decoderblock(in_size=7, nf=nf, kernel_size=3),
                                   nn.Conv2d(in_channels=2*nf,out_channels=2, kernel_size=3, stride=1, padding=1)
                                   )
    def forward(self, x):
        out = self.layer(x)
        out = nn.Softmax(out)
        return out
"""
class _EncoderBlock(nn.Module):
    def __init__(self, im_size = 224, out_channels = 512, nf = 32,kernel_size=3): #nf==64
        super(_EncoderBlock, self).__init__()
        layer1 = nn.Sequential(
            nn.Conv3d(in_channels=im_size, out_channels=nf, kernel_size=kernel_size, stride=2),

            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1),
            nn.Conv3d(in_channels=nf, out_channels=2*nf, kernel_size=(nf, 1, 1), stride=1),

            nn.Conv2d(in_channels=2*nf, out_channels=2*nf, kernel_size=3, stride=2),
            nn.Conv3d(in_channels=2*nf, out_channels=4*nf, kernel_size=(2*nf, 1, 1), stride=1),

            nn.Conv2d(in_channels=4*nf, out_channels=4*nf, kernel_size=3, stride=1),
            nn.Conv3d(in_channels=4*nf, out_channels=4*nf, kernel_size=(4*nf, 1, 1), stride=1),
            
            nn.Conv2d(in_channels=4*nf, out_channels=4*nf, kernel_size=3, stride=2),
            nn.Conv3d(in_channels=4*nf, out_channels=8*nf, kernel_size=(4*nf, 1, 1), stride=1),

            nn.Conv2d(in_channels=8*nf, out_channels=8*nf, kernel_size=3, stride=1),
            nn.Conv3d(in_channels=8*nf, out_channels=8*nf, kernel_size=(8*nf, 1, 1), stride=1),
            nn.Conv2d(in_channels=8*nf, out_channels=8*nf, kernel_size=3, stride=2),
            nn.Conv3d(in_channels=8*nf, out_channels=16*nf, kernel_size=(8*nf, 1, 1), stride=1),

            nn.Conv2d(in_channels=16*nf, out_channels=16*nf, kernel_size=3, stride=1),
            nn.Conv3d(in_channels=16*nf, out_channels=16*nf, kernel_size=(16*nf, 1, 1), stride=1),
            nn.Conv2d(in_channels=16*nf, out_channels=16*nf, kernel_size=3, stride=1),
            nn.Conv3d(in_channels=16*nf, out_channels=16*nf, kernel_size=(16*nf, 1, 1), stride=1),
            nn.Conv2d(in_channels=16*nf, out_channels=16*nf, kernel_size=3, stride=1),
            nn.Conv3d(in_channels=16*nf, out_channels=16*nf, kernel_size=(16*nf, 1, 1), stride=1),
            nn.Conv2d(in_channels=16*nf, out_channels=16*nf, kernel_size=3, stride=1),
            nn.Conv3d(in_channels=16*nf, out_channels=16*nf, kernel_size=(16*nf, 1, 1), stride=1),
            nn.Conv2d(in_channels=16*nf, out_channels=16*nf, kernel_size=3, stride=1),
            nn.Conv3d(in_channels=16*nf, out_channels=16*nf, kernel_size=(16*nf, 1, 1), stride=1),

            nn.Conv2d(in_channels=16*nf, out_channels=16*nf, kernel_size=3, stride=2),
            nn.Conv3d(in_channels=16*nf, out_channels=32*nf, kernel_size=(16*nf, 1, 1), stride=1),
            nn.Conv2d(in_channels=32*nf, out_channels=32*nf, kernel_size=3, stride=2),
            nn.Conv3d(in_channels=32*nf, out_channels=32*nf, kernel_size=(16*nf, 1, 1), stride=1),

        )
    def forward(self, x):
        return
"""

