import cv2
import torch
from model import MobileHairNet

def get_mask(image, net, size = 224):
    image_h, image_w = image.shape[0], image.shape[1]

    down_size_image = cv2.resize(image, size, size)
    b, g, r = cv2.split(down_size_image)
    down_size_image = cv2.merge([r,g,b])
    down_size_image.from_numpy(down_size_image).float().div(255.0).unsqueeze(0)
    mask = net(down_size_image)

    mask = torch.argmax(torch.squeeze(mask), 0)
    mask_cv2 = mask.data.cpu().numpy()
    mask_cv2 = cv2.resize(mask_cv2, image_h, image_w)

    return mask_cv2

def alpha_image(imge, mask, alpha = 0.5):
    