from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage import filters
from sklearn.preprocessing import normalize
import torch.nn as nn
import torch

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

class HairMatteLoss:
    def __init__(self, image, mask):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

class HairMatteLoss(nn.CrossEntropyLoss):
    def __init__(self, image, mask):
        super(HairMatteLoss, self).__init__()
        CrossEntropyLoss = nn.CrossEntropyLoss().to(self.device)
        criterion = CrossEntropyLoss(image, mask)
        grad_loss = HairMatteLoss(image, mask)
        return grad_loss + criterion

