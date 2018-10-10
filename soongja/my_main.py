import os
import pprint
import tensorflow as tf
from my_model import MobileHair
from my_utils import show_all_variables

flags = tf.app.flags
flags.DEFINE_integer("input_height", 256, "The height of input image. []")
flags.DEFINE_integer("input_width", 256, "The width of input image. []")

flags.DEFINE_integer("batch_size", 1, "The size of image batches []")
flags.DEFINE_integer("epoch", 25, "Epochs to train []")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")

flags.DEFINE_string("data_dir", "dataset", "Root directory of your dataset []")
flags.DEFINE_string("graph_dir", "graphs", "Directory name to save graphs []")
flags.DEFINE_string("log_dir", "logs", "Directory name to save tensorboard summaries []")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Directory name to save checkpoints []")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save test image samples []")

flags.DEFINE_integer("checkpoint_step", 10, "Step Interval to save checkpoints []")

FLAGS = flags.FLAGS

def main(_):
    # pp = pprint.PrettyPrinter()
    # pp.pprint(flags.FLAGS.__flags)

    os.makedirs(FLAGS.checkpoint_dir, exist_ok=True)
    os.makedirs(FLAGS.sample_dir, exist_ok=True)

    mobilehair = MobileHair(FLAGS)
    show_all_variables()

    mobilehair.train()

if __name__ == '__main__':
  tf.app.run()