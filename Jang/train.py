import os
import time
import numpy as np
from glob import glob

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import MobileMatNet
from loss import HairMatteLoss

# weight initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Trainer(object):
    def __init__(self, config, dataloader):
        self.config = config
        self.dataloader = dataloader
        self.num_steps = len(self.dataloader)

        self.num_classes = config.num_classes
        self.epoch = config.epoch
        self.decay_epoch = config.decay_epoch
        self.lr = config.lr
        self.sample_step = config.sample_step
        self.checkpoint_step = config.checkpoint_step
        self.checkpoint_dir = config.checkpoint_dir
        self.sample_dir = config.sample_dir

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.build_model()

    def build_model(self):
        self.net = MobileMatNet()
        self.net.apply(weights_init)
        self.net.to(self.device)

        if self.config.checkpoint_dir != '':
            self.load_model()

    def load_model(self):
        print("[*] Load models from {}...".format(self.checkpoint_dir))

        paths = glob(os.path.join(self.checkpoint_dir, 'MobileHairNet*.pth'))
        paths.sort()

        if len(paths) == 0:
            print("[!] No checkpoint found in {}...".format(self.checkpoint_dir))
            return

        filename = paths[-1]

        self.net.load_state_dict(torch.load(filename, map_location=self.device))

        print("[*] Model loaded: {}".format(filename))

    def train(self):
        CrossEntropyLoss = nn.CrossEntropyLoss().to(self.device)
        # CrossEntropyLoss사용할 때는 input이 (N, C), target이 (N) 형태여야함

        optimizer = optim.Adadelta(self.net.parameters(), lr=self.lr, rho=0.95, eps=1e-07)

        start_time = time.time()
        for epoch in range(self.epoch):

            if epoch != 0 and epoch % self.decay_epoch == 0:
                optimizer.param_groups[0]['lr'] = self.lr / 10
                print('learning rate decayed')

            for step, (img, mask) in enumerate(self.dataloader):
                img, mask = img.to(self.device), mask.to(self.device)
                pred = self.net(img)
                # pred.shape (N, 2, 224, 224)
                # mask.shape (N, 1, 224, 224)

                pred_flat = pred.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
                mask_flat = mask.squeeze(1).view(-1).long()
                # pred_flat.shape (N*224*224, 2)
                # mask_flat.shape (N*224*224, 1)

                self.net.zero_grad()
                loss = HairMatteLoss(img, mask, pred)
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
                    torch.save(self.net.state_dict(), '%s/MobileHairNet_epoch-%d_step-%d.pth'
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