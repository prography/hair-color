import torch
import torch.nn as nn


class _Layer_Depwise_Encode(nn.Module):
    def __init__(self, in_chennels, out_chennels,kernel_size=3, reserve=False): #nf==64
        self.stride = int(out_chennels/in_chennels)
        if reserve==True:
            self.stride = 1
        super(_Layer_Depwise_Encode, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_chennels, out_channels=in_chennels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(in_chennels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels=in_chennels, out_channels=out_chennels, kernel_size=1, stride=self.stride),
            nn.BatchNorm2d(out_chennels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class _EncoderBlock(nn.Module):
    def __init__(self, im_size, nf, kernel_size=3):
        super(_EncoderBlock, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=nf, kernel_size=kernel_size, stride=2, padding=1)
        self.layer2 = nn.Sequential(
            _Layer_Depwise_Encode(nf, 2*nf),
            _Layer_Depwise_Encode(2*nf, 2*nf),
            _Layer_Depwise_Encode(2*nf, 4*nf),
            _Layer_Depwise_Encode(4*nf, 4*nf),
            _Layer_Depwise_Encode(4*nf, 8*nf, reserve=True),
            _Layer_Depwise_Encode(8*nf, 8*nf),
            _Layer_Depwise_Encode(8*nf, 8*nf),
            _Layer_Depwise_Encode(8*nf, 8*nf),
            _Layer_Depwise_Encode(8*nf, 8*nf),
            _Layer_Depwise_Encode(8*nf, 8*nf),
            _Layer_Depwise_Encode(8*nf, 16*nf, reserve=True),
            _Layer_Depwise_Encode(16*nf, 16*nf),
            _Layer_Depwise_Encode(16*nf, 16*nf)
        )

    def forward(self, x):
        print(x.shape)
        out = self.layer1(x)
        print(out.shape)
        out2 = self.layer2(out)
        print("2")
        print(out2.shape)
        return

"""
class _EncoderBlock(nn.Module):
    def __init__(self, im_size = 224, out_chennels = 512, nf = 32,kernel_size=3): #nf==64
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

