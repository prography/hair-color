import os
import random

import torch
import torch.backends.cudnn as cudnn
from config import get_config
from data_loader import get_loader
from train import Trainer


def main(config):
    config.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", config.manual_seed)
    random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.manual_seed)

    cudnn.benchmark = True

    train_loader = get_loader(config.data_dir, config.image_size, config.batch_size,
                            shuffle=True, num_workers=int(config.workers))

    test_loader = get_loader(config.test_data_dir, config.image_size, batch_size=1,
                             shuffle=False, num_workers=int(config.workers))

    trainer = Trainer(config, train_loader, test_loader)
    trainer.train()


if __name__ == "__main__":
    config = get_config()
    main(config)
