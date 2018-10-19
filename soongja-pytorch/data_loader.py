import os
import shutil

import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_size):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            raise Exception("[!] {} not exists.".format(self.data_dir))

        self.image_paths = os.listdir(os.path.join(self.data_dir, 'original')) #파일명은 같기에 "image"대신 "mask"해도 동일합니다.
        print('Dataset size:', len(self.image_paths))
        if len(self.image_paths) == 0:
            raise Exception("No images are found in {}".format(self.data_dir))

        self.image_size = image_size

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.data_dir, 'original', self.image_paths[index])).convert('RGB')
        mask = Image.open(os.path.join(self.data_dir, 'masks', self.image_paths[index]))

        image = image.resize((self.image_size, self.image_size))
        mask = mask.resize((self.image_size, self.image_size))

        image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])(image)

        mask = transforms.Compose([
            transforms.ToTensor()
            ])(mask)

        return image, mask

    def __len__(self):
        return len(self.image_paths)


def get_loader(data_dir, batch_size, image_size, shuffle, num_workers):
    dataset = Dataset(data_dir, image_size)

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers)

    return dataloader