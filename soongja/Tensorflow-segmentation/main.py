import tensorflow as tf
import pprint

from train import Trainer

flags = tf.app.flags
flags.DEFINE_integer("input_height", 128, "The size of image to use. [128]")
flags.DEFINE_integer("input_width", 128, "The size of image to use. If None, same value as input_height [None]")
flags.DEFINE_integer("input_channels", 1, "The channels of image to use. [1]")
flags.DEFINE_integer("epoch", 100, "Epochs to train [100]")
flags.DEFINE_integer("batch_size", 16, "The size of batch images [64]")
# flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_string("data_dir", "./data", "Root directory of dataset [data]")
flags.DEFINE_string("graph_dir", "graph", "Directory name to save the graph [graph]")
flags.DEFINE_string("log_dir", "logs", "Directory name to save logs [logs]")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Directory name to save the checkpoints [checkpoints]")
flags.DEFINE_integer("test_step", 100, "Test steps. [100]")
FLAGS = flags.FLAGS


def main(_):
    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)

    trainer = Trainer(FLAGS)
    trainer.train()

if __name__ == '__main__':
    tf.app.run()