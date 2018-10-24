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
        im = rgb2gray(image)
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
        self.num_classes = config.num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(HairMetteLoss, self).__init__()
        criterion = nn.CrossEntropyLoss().to(self.device)
        pred_flat = pred.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
        mask_flat = mask.squeeze(1).view(-1).long()
        # pred_flat.shape (N*224*224, 2)
        # mask_flat.shape (N*224*224, 1)

        criterion = criterion(pred_flat, mask_flat)
        grad_loss = GradientLoss(image, pred)
        grad_loss = torch.mul(grad_loss, w)
        total_loss = torch.add(grad_loss, criterion)

        return total_loss

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
        self.num_classes = config.num_classes
        self.epoch = config.epoch
        self.lr = config.lr
        self.checkpoint_dir = config.checkpoint_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.build_model()

    def build_model(self):
        self.net = Mobilehair()
        self.net.apply(weights_init)
        self.net.to(self.device)

        if self.config.checkpoint_dir != '':
            self.load_model()

    def load_model(self):
        print("[*] Load models from {}...".format(self.checkpoint_dir))

        paths = glob(os.path.join(self.checkpoint_dir, 'Mobilehair*.pth'))
        paths.sort()

        if len(paths) == 0:
            print("[!] No checkpoint found in {}...".format(self.checkpoint_dir))
            return

        filename = paths[-1]
        self.net.load_state_dict(torch.load(filename, map_location=self.device))
        print("[*] Model loaded: {}".format(filename))

    def train(self):
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        # --> optimizer

        start_time = time.time()
        print('Start Training!')
        for epoch in range(self.epoch):
            for step, (imgs, masks) in enumerate(self.data_loader):
                img, mask = imgs.to(self.device), masks.to(self.device)
                pred = self.net(img)
                # pred.shape (N, 2, 224, 224)
                # mask.shape (N, 1, 224, 224)

                self.net.zero_grad()
                loss = HairMetteLoss(image, mask, pred, 0.5)
                loss.backward()
                optimizer.step()

                step_end_time = time.time()
                print('[%d/%d][%d/%d] - time_passed: %.2f, CrossEntropyLoss: %.2f'
                      % (epoch, self.epoch, step, self.num_steps, step_end_time - start_time, loss))

                # save sample images
                if step % self.sample_step == 0:
                    self.save_sample_imgs(img[0], mask[0], torch.argmax(pred[0], 0), self.sample_dir, epoch, step)
                    print('[*] Saved sample images')

                # save checkpoints
                if step % self.checkpoint_step == 0:
                    torch.save(self.net.state_dict(), '%s/MobileHair_epoch-%d_step-%d.pth'
                               % (self.checkpoint_dir, epoch, step))
                    print("[*] Saved checkpoint")

    def save_sample_imgs(self, real_img, real_mask, prediction, save_dir, epoch, step):
        data = [real_img, real_mask, prediction]
        names = ["Image", "Mask", "Prediction"]

        fig = plt.figure()
        for i, d in enumerate(data):
            d = d.squeeze()
            im = d.data.cpu().numpy()

            if i > 0:
                im = np.expand_dims(im, axis=0)
                im = np.concatenate((im, im, im), axis=0)

            im = (im.transpose(1, 2, 0) + 1) / 2

            f = fig.add_subplot(1, 3, i + 1)
            f.imshow(im)
            f.set_title(names[i])
            f.set_xticks([])
            f.set_yticks([])

        p = os.path.join(save_dir, "epoch-%s_step-%s.png" % (epoch, step))
        plt.savefig(p)