from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage import filters
from sklearn.preprocessing import normalize
import torch.nn as nn
import torch
from config import get_config
import os
config = get_config()

class ImageGradient:
    def __init__(self, image_name):
        self.image = image_name
    def get_gradient(self):
        edges_x = filters.sobel_h(self.image)
        edges_y = filters.sobel_v(self.image)

        edges_x = normalize(edges_x)
        edges_y = normalize(edges_y)

        return edges_x, edges_y

class ImageGradientLoss:
    def __init__(self, image, mask):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image = image
        self.mask = mask
    def get_loss(self):
        image_grad_x, image_grad_y = ImageGradient(os.path.join(config.data_path, 'original',self.image)).get_gradient()
        mask_grad_x, mask_grad_y = ImageGradient(os.path.join(config.data_path, 'mask',self.image)).get_gradient()
        IMx = torch.mul(image_grad_x, mask_grad_x)
        IMy = torch.mul(image_grad_y, mask_grad_y)
        Mmag = torch.sqrt(torch.add(torch.pow(mask_grad_x, 2), torch.pow(mask_grad_y, 2)))
        IM = torch.add(1, torch.neg(torch.add(IMx, IMy)))
        numerator = torch.sum(torch.mul(Mmag, IM))
        denominator = torch.sum(Mmag)
        out = torch.div(numerator, denominator)
        return torch.FloatTensor(out)

class HairMatLoss:
    def __init__(self):
        self.num_classes = config.num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.CrossEntropyLoss = nn.CrossEntropyLoss().to(self.device)
    def get_loss(self, pred, mask, image):
        pred_flat = pred.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
        mask_flat = mask.squeeze(1).view(-1).long()
        criterion = self.CrossEntropyLoss(pred_flat, mask_flat)
        grad_loss = ImageGradientLoss(image.cpu, mask.cpu)
        return torch.add(grad_loss, criterion)

