import os

import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import random_split


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, image_size):
        self.data_folder = data_folder
        if not os.path.exists(self.data_folder):
            raise Exception("[!] {} not exists.".format(self.data_folder))
        self.image_size = image_size

        self.image_paths = os.listdir(os.path.join(self.data_folder, 'masks')) #파일명은 같기에 "image"대신 "mask"해도 동일합니다.
        print('Dataset size:', len(self.image_paths))
        if len(self.image_paths) == 0:
            raise Exception("No images are found in {}".format(self.data_folder))

        self.images = []


    def __getitem__(self, index):
        image = Image.open(os.path.join(self.data_folder, 'images', self.image_paths[index])).convert('RGB')
        mask = Image.open(os.path.join(self.data_folder, 'masks', self.image_paths[index]))

        image = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])(image)


        mask = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
            ])(mask)

        return image, mask


    def __len__(self):
        return len(self.images)


def get_loader(data_folder, image_size, batch_size):
    dataset = Dataset(data_folder, image_size)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=batch_size,
                                                shuffle=True)

    return data_loader
