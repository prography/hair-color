import time
import numpy as np

import cv2
import torch
from model import MobileHairNet


def get_mask(cv2_frame, net, downscale_size):
    original_h, original_w = cv2_frame.shape[0], cv2_frame.shape[1]

    down_cv2 = cv2.resize(cv2_frame, (downscale_size, downscale_size))
    flipped = down_cv2[:, :, ::-1].transpose((2, 0, 1)).copy()  # BGR to RGB, HWC to CHW
    down_tensor = torch.from_numpy(flipped).float().div(255.0).unsqueeze(0) # to tensor, scale to 0~1, HWC to NHWC

    with torch.no_grad():
        mask_tensor = net(down_tensor)  # (2 x 256 x 256)

    mask_tensor = torch.argmax(mask_tensor.squeeze(), 0)  # (256 x 256)

    mask_cv2 = mask_tensor.data.cpu().numpy()
    mask_cv2 = mask_cv2.astype(np.uint8) * 255
    mask_cv2 = cv2.resize(mask_cv2, (original_h, original_w))

    return mask_cv2


def alpha_hand(frame, mask):

    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    colored_mask[np.where(mask != 0)] = [0, 130, 255] # BGR!!!
    alpha_hand = (0.7 * frame + 0.3 * colored_mask).astype(np.uint8)
    alpha_hand = cv2.bitwise_and(alpha_hand, alpha_hand, mask=mask)

    mask_inv = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(frame, frame, mask=mask_inv)

    return cv2.add(alpha_hand, background)


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = MobileHairNet().to(device)
    pretrained = 'checkpoints/MobileHairNet_epoch-0_step-0.pth'
    net.load_state_dict(torch.load(pretrained, map_location=device))

    cap = cv2.VideoCapture(0)
    frames = 0
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if ret:
            frame = frame[60:420, 140:500] # y1:y2, x1:x2. 정사각형으로 crop
            mask = get_mask(frame, net, 224)
            out = alpha_hand(frame, mask)

            frames += 1
            fps = frames / (time.time() - start_time)
            cv2.putText(out, 'FPS: %.2f' % fps, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 0], 1)

            cv2.imshow('demo', out)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()
