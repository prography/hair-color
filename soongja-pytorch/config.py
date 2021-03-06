import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--num_classes', type=int, default=2, help='number of model output channels')
parser.add_argument('--grad_loss_lambda', type=float, default=0.5, help='')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--decay_epoch', type=int, default=10, help='learning rate decay start epoch num')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

parser.add_argument('--sample_step', type=int, default=100, help='step of saving sample images')
parser.add_argument('--checkpoint_step', type=int, default=100, help='step of saving checkpoints')
parser.add_argument('--ckpt_max_to_keep', type=int, default=5, help='number of checkpoints to be kept')
parser.add_argument('--prefix', default='yo', help="subfolder prefixes of samples and checkpoints")
parser.add_argument('--data_dir', default='hairdata/train', help='path to dataset')
parser.add_argument('--test_data_dir', default='hairdata/test', help='folder to output images and model checkpoints')
parser.add_argument('--sample_dir', default='samples', help='folder to output images and model checkpoints')
parser.add_argument('--checkpoint_dir', default='checkpoints', help="path to saved models (to continue training)")
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')

def get_config():
    return parser.parse_args()
