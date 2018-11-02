import cv2
import torch
from model import MobileHairNet
from config import get_config
import os
import numpy as np

def get_mask(image, net, size = 224):
    image_h, image_w = image.shape[0], image.shape[1]

    down_size_image = cv2.resize(image, (size, size))
    b, g, r = cv2.split(down_size_image)
    down_size_image = cv2.merge([r,g,b])
    down_size_image = torch.from_numpy(down_size_image).float().div(255.0).unsqueeze(0)
    down_size_image = np.transpose(down_size_image, (0, 3, 1, 2)).to(device)
    mask = net(down_size_image)

    mask = torch.argmax(torch.squeeze(mask), 0)
    mask_cv2 = mask.data.cpu().numpy().astype(np.uint8) * 255
    mask_cv2 = cv2.resize(mask_cv2, (image_w, image_h))

    return mask_cv2

def alpha_image(image, mask, alpha = 0.1):
    color = np.zeros((mask.shape[0], mask.shape[1], 3))
    color[np.where(mask != 0)] = [0, 130, 255]
    alpha_hand = ((1 - alpha) * image + alpha * color).astype(np.uint8)
    alpha_hand = cv2.bitwise_and(alpha_hand, alpha_hand, mask=mask)

    return cv2.add(alpha_hand, image)

if __name__ == "__main__":
    config = get_config()
    pretrained = os.listdir(config.checkpoint_dir)[-1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = MobileHairNet().to(device)
    net.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, pretrained), map_location=device))
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise Exception("webcam is not detected")

    while (True):
        # ret : frame capture결과(boolean)
        # frame : Capture한 frame
        ret, image = cam.read()

        if (ret):
            mask = get_mask(image, net)
            add = alpha_image(image, mask)
            cv2.imshow('frame', add)
            if cv2.waitKey(1) & 0xFF == ord(chr(27)):
                break

    cam.release()
    cv2.destroyAllWindows()
