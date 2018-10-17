import glob
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

        self.images = []
        for label, dir in enumerate(os.listdir(self.data_folder)):
            for path in glob.glob(os.path.join(self.data_folder, dir, '*')):
                self.images.append((path, label))

    def __getitem__(self, index):
        path, label = self.images[index]

        im = Image.open(path).convert('RGB')

       transform = transforms.Compose([
            transforms.CenterCrop(min(im.size[0], im.size[1])),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))            ##Normalize(평균,표준편차)
            ])

        return transform(im), label              #classification이니까 image랑 label을 return한다

    def __len__(self):
        return len(self.images)


def get_loader(data_folder, image_size, batch_size):
    dataset = Dataset(data_folder, image_size)

    train_length = int(0.9 * len(dataset))
    test_length = len(dataset) - train_length

    train_dataset, test_dataset = random_split(dataset, (train_length, test_length))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)

    return train_loader, test_loader
