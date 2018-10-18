import os
import time
from glob import glob

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from model import Mobilehair

from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage import filters
from sklearn.preprocessing import normalize

##Loss
class ImageGradient:
    def __init__(self, image):
        self.image = image
    def get_gradient(self):
        im = rgb2gray(imread('./dataset/Kim.PNG'))
        edges_x = filters.sobel_h(im)
        edges_y = filters.sobel_v(im)

        edges_x = normalize(edges_x)
        edges_y = normalize(edges_y)

        return edges_x, edges_y

class GradientLoss:
    def __init__(self, image, mask):
        self.image = image
        self.mask = mask
    def get_loss(self):
        image_grad_x, image_grad_y = ImageGradient(image=self.image).get_gradient()
        mask_grad_x, mask_grad_y = ImageGradient(image=self.mask).get_gradient()
        IMx = torch.mul(image_grad_x, mask_grad_x)
        IMy = torch.mul(image_grad_y, mask_grad_y)
        Mmag = torch.sqrt(torch.add(torch.pow(mask_grad_x, 2), torch.pow(mask_grad_y, 2)))
        IM = torch.add(1, torch.neg(torch.add(IMx, IMy)))
        numerator = torch.sum(torch.mul(Mmag, IM))
        denominator = torch.sum(Mmag)
        out = torch.div(numerator, denominator)
        return out

class HairMetteLoss(nn.CrossEntropyLoss):
    def __init__(self, image, mask, pred, w):
        super(HairMetteLoss, self).__init__()
        criterion = nn.CrossEntropyLoss()
        criterion = criterion(pred, mask)
        grad_loss = GradientLoss(image, pred)

        return grad_loss*w + criterion                  ##weight을 곱한다.

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Trainer(object):
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader
        self.image_size = config.image_size
        self.nf = config.nf
        self.num_classes = config.num_classes
        self.epoch = config.epoch
        self.lr = config.lr
        self.model_path = config.model_path
        self.outf = config.outf

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.build_model()

    def build_model(self):
        self.net = Mobilehair()
        self.net.apply(weights_init)
        self.net.to(self.device)

        if self.config.model_path != '':
            self.load_model()

    def load_model(self):
        print("[*] Load models from {}...".format(self.model_path))

        paths = glob(os.path.join(self.model_path, 'Mobilehair*.pth'))
        paths.sort()

        if len(paths) == 0:
            print("[!] No checkpoint found in {}...".format(self.model_path))
            return

        filename = paths[-1]
        self.net.load_state_dict(torch.load(filename, map_location=self.device))
        print("[*] Model loaded: {}".format(filename))

    def train(self):
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        # --> optimizer

        start_time = time.time()
        print('Start Training!')
        for epoch in range(self.epoch):
            for step, (imgs, masks) in enumerate(self.data_loader):
                imgage, mask = imgs.to(self.device), masks.to(self.device)
                pred = self.net(imgs)
                # pred.shape (N, 2, 224, 224)
                # mask.shape (N, 1, 224, 224)

                pred_flat = pred.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
                mask_flat = mask.squeeze(1).view(-1).long()
                # pred_flat.shape (N*224*224, 2)
                # mask_flat.shape (N*224*224, 1)

                self.net.zero_grad()
                loss = criterion(pred_flat, mask_flat)
                loss.backward()
                optimizer.step()


                # --> training process

                step_end_time = time.time()
                print('[%d/%d][%d/%d] - time_passed: %.2f, Loss: %.4f'
                      % (epoch, self.epoch, step, len(self.train_loader), step_end_time - start_time, loss))

            torch.save(self.net.state_dict(), '%s/CNN_epoch_%d.pth' % (self.outf, epoch))
            print("Saved checkpoint")
            print('Finished Training')

