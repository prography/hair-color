import os
import time
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from model import Mobilehair





class HairMatteLoss(nn.CrossEntropyLoss):
    def __init__(self, image, mask, pred, w):
        self.num_classes = 2
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(HairMatteLoss, self).__init__()
        criterion = nn.CrossEntropyLoss().to(self.device)
        pred_flat = pred.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
        mask_flat = mask.squeeze(1).view(-1).long()
        # pred_flat.shape (N*224*224, 2)
        # mask_flat.shape (N*224*224, 1)

        criterion = criterion(pred_flat, mask_flat)
        self.total_loss = criterion



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Trainer(object):
    def __init__(self, config, train_loader, valid_loader):
        self.config = config
        self.data_loader = train_loader
        self.valid_loader = valid_loader
        self.num_steps = len(self.data_loader)

        self.image_size = config.image_size
        self.num_classes = config.num_classes
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.lr = config.lr

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.train_dir = config.train_dir
        self.valid_dir = config.valid_dir
        self.sample_dir = config.sample_dir
        self.checkpoint_dir = config.checkpoint_dir
        self.sample_step = config.sample_step
        self.checkpoint_step = config.checkpoint_step
        self.ckpt_max_to_keep = config.ckpt_max_to_keep
        self.prefix = config.prefix

        self.build_model()

    def build_model(self):
        self.net = Mobilehair()
        self.net.apply(weights_init)
        self.net.to(self.device)

        # Load checkpoints
        print("[*] Load models from {}...".format(os.path.join(self.checkpoint_dir, self.model_dir())))
        paths = glob(os.path.join(self.checkpoint_dir, self.model_dir(), '*.pth'))

        if len(paths) == 0:
            print("[!] No checkpoint found in {}...".format(os.path.join(self.checkpoint_dir, self.model_dir())))
            self.counter = 0

        else:
            paths.sort()

            steps = [int(os.path.basename(path.split('-')[0])) for path in paths]
            self.counter = int(max(steps))

            filename = self.checkpoint_name(self.counter, 'MobileHairNet')

            self.net.load_state_dict(torch.load(filename, map_location=self.device))

            print("[*] Model loaded: {}".format(filename))


    def train(self):
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        # --> optimizer

        counter = self.counter

        start_time = time.time()
        print('Start Training!')
        for epoch in range(self.epoch):

            for step, (imgs, masks) in enumerate(self.data_loader):

                img, mask = imgs.to(self.device), masks.to(self.device)
                pred = self.net(img)
                # pred.shape (N, 2, 224, 224)
                # mask.shape (N, 1, 224, 224)

                self.net.zero_grad()
                loss = HairMatteLoss(img, mask, pred, 0.5).total_loss
                loss.backward()
                optimizer.step()

                counter +=1
                step_end_time = time.time()
                print('[%d/%d][%d/%d] - time_passed: %.2f, CrossEntropyLoss: %.2f'
                      % (epoch, self.epoch, step, self.num_steps, step_end_time - start_time, loss))

                # save sample images
                if step % self.sample_step == 0:
                    for num, (imgs, masks) in enumerate(self.valid_loader):
                        imgs, masks = imgs.to(self.device), masks.to(self.device)
                        preds = self.net(imgs)

                        inverse_normalize = transforms.Normalize(mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
                                                                 std=[1 / 0.5, 1 / 0.5, 1 / 0.5])

                        imgs = inverse_normalize(imgs[0]).permute(1, 2, 0).detach().cpu().numpy()[:, :, ::-1] * 255
                        masks = masks[0].repeat(3, 1, 1).permute(1, 2, 0).detach().cpu().numpy() * 255
                        preds = torch.argmax(preds[0], 0).unsqueeze(2).repeat(1, 1, 3).detach().cpu().numpy() * 255

                        if not os.path.exists(os.path.join(self.sample_dir, self.model_dir())):
                            os.makedirs(os.path.join(self.sample_dir, self.model_dir()))

                        samples = np.hstack((imgs, masks, preds))
                        cv2.imwrite('{}/{}/sample_{}-{}.png'.format(self.sample_dir, self.model_dir(), num, counter),
                                    samples)
                    print('Saved images')

                # save checkpoints
                if step % self.checkpoint_step == 0:
                    if not os.path.exists(os.path.join(self.checkpoint_dir, self.model_dir())):
                        os.makedirs(os.path.join(self.checkpoint_dir, self.model_dir()))

                    self.save_checkpoint(counter, self.ckpt_max_to_keep)
                    print("Saved checkpoint")


    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.prefix, self.batch_size,
            self.image_size, self.image_size)

    def checkpoint_name(self, counter, name):
        return '{}/{}/{}-{}.pth'.format(self.checkpoint_dir, self.model_dir(), counter, name)

    def save_checkpoint(self, counter, max_to_keep):
        torch.save(self.net.state_dict(), self.checkpoint_name(counter, 'MobileHairNet'))

        self.checkpoints_to_keep.append(self.checkpoint_name(counter, 'MobileHairNet'))

        if len(self.checkpoints_to_keep) > max_to_keep:
            os.remove(self.checkpoints_to_keep[0])
            del self.checkpoints_to_keep[0]