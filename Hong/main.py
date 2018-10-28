import os
import random

import torch
import torch.backends.cudnn as cudnn
from config import get_config
from dataloader import get_loader
from train import Trainer


def main(config):
    os.makedirs(config.sample_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    config.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", config.manual_seed)
    random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.manual_seed)

    cudnn.benchmark = True

    train_loader = get_loader(config.train_dir, config.image_size, config.batch_size)
    validation_loader = get_loader(config.valid_dir, config.image_size, config.batch_size)

    trainer = Trainer(config, train_loader, validation_loader)
    trainer.train()


if __name__ == "__main__":
    config = get_config()
    main(config)
