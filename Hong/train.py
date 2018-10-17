import os
import time
from glob import glob

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from model import Basic_CNN


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Trainer(object):
    def __init__(self, config, train_loader, test_loader):
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
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
        self.net = Basic_CNN(self.image_size, self.nf, self.num_classes)
        self.net.apply(weights_init)
        self.net.to(self.device)

        if self.config.model_path != '':
            self.load_model()

    def load_model(self):
        print("[*] Load models from {}...".format(self.model_path))

        paths = glob(os.path.join(self.model_path, 'CNN*.pth'))
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
            for step, (imgs, labels) in enumerate(self.train_loader):
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                preds = self.net(imgs)

                self.net.zero_grad()
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()


                # --> training process

                step_end_time = time.time()
                print('[%d/%d][%d/%d] - time_passed: %.2f, Loss: %.4f'
                      % (epoch, self.epoch, step, len(self.train_loader), step_end_time - start_time, loss))

            torch.save(self.net.state_dict(), '%s/CNN_epoch_%d.pth' % (self.outf, epoch))
            print("Saved checkpoint")
            print('Finished Training')

        # Test
        print('------------Test------------')
        self.net.eval()   #test용 , dropout X, batch normalizaiton 이 아니라 누적으로 normalization함
        with torch.no_grad(): ###test용 backpro안하는거
            correct = 0
            total = 0

            for step, (imgs, labels) in enumerate(self.test_loader):
                imgs = imgs.to(self.device)
                preds = self.net(imgs)
                _, predicted = torch.max(preds, dim=1)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # show first batch results with matplotlib
                batch_size = imgs.size(0)
                if step == 0:
                    fig = plt.figure()
                    inverse_normalize = transforms.Normalize(
                        mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
                        std=[1 / 0.5, 1 / 0.5, 1 / 0.5])
                    for i in range(batch_size):
                        subplot = fig.add_subplot(1, batch_size, i + 1)
                        subplot.set_xticks([])
                        subplot.set_yticks([])
                        if predicted[i] == labels[i]:
                            color = 'green'
                        else:
                            color = 'red'
                        subplot.set_title('pred: %d\n real: %d' % (predicted[i], labels[i]), color=color)
                        subplot.imshow(inverse_normalize(imgs[i]).numpy().transpose(1,2,0))

            print('Test accuracy on %d test images: %.2f%%' % (total, 100 * correct / total))
            plt.show()
