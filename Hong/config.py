import argparse

parser = argparse.ArgumentParser()


parser.add_argument('--image_size', type=int, default=224, help='height/width of the input image to network')
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--decay_epoch', type=int, default=10, help='learning rate decay start epoch num')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--sample_step', type=int, default=100, help='step of saving sample images')
parser.add_argument('--checkpoint_step', type=int, default=100, help='step of saving checkpoints')
parser.add_argument('--data_dir', default='figaro', help='path to dataset')
parser.add_argument('--checkpoint_dir', default='checkpoints', help="path to saved models (to continue training)")
parser.add_argument('--sample_dir', default='samples', help='folder to output images and model checkpoints')




def get_config():
    return parser.parse_args()

