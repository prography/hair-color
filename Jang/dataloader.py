#
#
# data folder 구조
# (data_folder) / original
# (data_folder) / mask
# (data_folder) / ...
# (data_folder) / ...
#
#
import os

import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, image_size):
        self.data_folder = data_folder
        if not os.path.exists(self.data_folder):
            raise Exception(" ! %s  not exists." % self.data_folder)

        self.objects_path = []
        self.image_name = os.listdir(os.path.join(data_folder, "original"))
        for p in os.listdir(data_folder):
            if p == "original":
                continue
            self.objects_path.append(os.path.join(data_folder, p))


        self.image_size = image_size


    def __getitem__(self, index):
        image = Image.open(os.path.join(self.data_folder, 'original', self.image_name[index])).convert('RGB')
        objects = []
        for p in self.objects_path:
            objects.append(Image.open(os.path.join(p, self.image_name[index])))



        transform_image = transforms.Compose([
            transforms.CenterCrop(min(image.size[0], image.size[1])),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])(image)

        transform_object = []
        for num,_ in enumerate(self.objects_path):
            transform_object.append(transforms.Compose([
                transforms.CenterCrop(min(image.size[0], image.size[1])),
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])(objects[num]))



        return transform_image, transform_object[0]

    def __len__(self):
        return len( self.image_name)


def get_loader(data_folder, batch_size, image_size, shuffle, num_workers):
    dataset = Dataset(data_folder, image_size)

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers)
    return dataloader
