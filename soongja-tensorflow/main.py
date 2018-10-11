import os
import pprint
import tensorflow as tf
from model import MobileHair
from utils import show_all_variables

flags = tf.app.flags
flags.DEFINE_integer("input_height", 224, "The height of input image. []")
flags.DEFINE_integer("input_width", 224, "The width of input image. []")
flags.DEFINE_integer("batch_size", 4, "The size of image batches. Must be smaller than # of train images []")
flags.DEFINE_integer("epoch", 25, "Epochs to train []")
flags.DEFINE_float("learning_rate", 1.0, "Learning rate of for adam [0.0002]")
flags.DEFINE_string("graph_dir", "graphs", "Directory name to save graphs []")
flags.DEFINE_string("log_dir", "logs", "Directory name to save tensorboard summaries []")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Directory name to save checkpoints []")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save test image samples []")
flags.DEFINE_string("data_dir", "dataset", "Root directory of your dataset []")
flags.DEFINE_string("dataset_name", "segxyhand", "Root directory of your dataset []")
flags.DEFINE_integer("checkpoint_step", 100, "Step Interval to save checkpoints []")
flags.DEFINE_integer("test_step", 100, "Step Interval to test sample images []")

FLAGS = flags.FLAGS


def main(_):
    # pp = pprint.PrettyPrinter()
    # pp.pprint(flags.FLAGS.__flags)

    os.makedirs(FLAGS.checkpoint_dir, exist_ok=True)
    os.makedirs(FLAGS.sample_dir, exist_ok=True)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        net = MobileHair(sess, FLAGS)

        show_all_variables()

        net.train()


if __name__ == '__main__':
  tf.app.run()