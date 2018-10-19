from model import HairMatNet
import os
from glob import glob
import torch
import torch.nn as nn



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Trainer:
    def __init__(self, config, dataloader):
        os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
        self.batch_size = config.batch_size
        self.config = config
        self.lr = config.lr
        self.epoch = config.epoch
        self.checkpoint_dir = config.checkpoint_dir
        self.model_path = config.checkpoint_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_loader = dataloader
        self.image_len = len(dataloader)
        self.build_model()
        self.num_classes = config.num_classes

    def build_model(self):
        self.net = HairMatNet()
        self.net.apply(weights_init)
        self.net.to(self.device)


    def load_model(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if os.listdir(self.model_path) == False:
            print(" * No checkpoint in ", str(self.model_path))

        model = glob(os.path.join(self.model_path, "HairMatNet*.pth"))
        model.sort()

        self.net.load_state_dict(torch.load(model[-1], map_location=self.device))
        print(" * Load Model form %s: " % str(self.model_path), str(model[-1]))

    def train(self):
        CrossEntropyLoss = nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, eps=1e-7)

        for epoch in range(self.epoch):
            for step, (image, mask, image_name) in enumerate(self.data_loader):
                image = image.to(self.device)
                mask = mask.to(self.device)
                criterion = self.net(image)

                pred_flat = criterion.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
                mask_flat = mask.squeeze(1).view(-1).long()


                self.net.zero_grad()
                loss = CrossEntropyLoss(pred_flat, mask_flat)
                loss.backward()
                optimizer.step()

                print("epoch: [%d/%d] | image: [%d/%d] | loss: %.2f" % (epoch, self.epoch, step, self.image_len, loss))
        torch.save(self.net.state_dict(), '%s/HairMatNet_epoch-%d.pth' % (self.checkpoint_dir, epoch))