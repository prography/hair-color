import tensorflow as tf
from model import MobileHairNet
from utils import show_all_variables

flags = tf.app.flags
flags.DEFINE_integer("input_height", 224, "The height of input image")
flags.DEFINE_integer("input_width", 224, "The width of input image")
flags.DEFINE_integer("batch_size", 4, "The size of image batches")
flags.DEFINE_integer("epoch", 50, "Epochs to train")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for Adam")
flags.DEFINE_string("graph_dir", "graphs", "Directory name to save graphs")
flags.DEFINE_string("log_dir", "logs", "Directory name to save tensorboard summaries")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Directory name to save checkpoints")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save validation image samples")
flags.DEFINE_string("data_dir", "dataset/hairdata", "Root directory of your dataset")
flags.DEFINE_string("prefix", "Mon", "Used for folder names of ckpts, logs, etc.(do not use . or /")
flags.DEFINE_integer("validation_step", 200, "Step Interval to test on validation set")

FLAGS = flags.FLAGS


def main(_):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        model = MobileHairNet(sess, FLAGS)

        show_all_variables()

        model.train()


if __name__ == '__main__':
    tf.app.run()