import os
import cv2
import numpy as np
import tensorflow as tf
from model import Network

flags = tf.app.flags
flags.DEFINE_integer("input_height", 224, "The height of input image")
flags.DEFINE_integer("input_width", 224, "The width of input image")
flags.DEFINE_string("checkpoint_dir", "checkpoints/hairdata_4_224_224", "Directory where checkpoints are saved")
flags.DEFINE_string("test_image", "saeron.png", "Test image")
flags.DEFINE_string("save_dir", "test", "Directory to save test outputs")

FLAGS = flags.FLAGS

def main(_):
    net = Network(FLAGS.input_height, FLAGS.input_width)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Load success: %s' % ckpt.model_checkpoint_path)
        else:
            print('Load failed. Check your checkpoint directory!!!')

        # read image
        img = cv2.resize(cv2.imread(FLAGS.test_image), (FLAGS.input_height, FLAGS.input_width))

        _input = np.array(img, dtype=np.float32)[:, :, ::-1]
        # _inpu = np.multiply(_input, 1.0 / 255)
        # _input = sess.run(tf.image.per_image_standardization(_input))
        _input = np.expand_dims(_input, axis=0)

        out = sess.run(net.preds, feed_dict={net.inputs: _input})
        out = out[0] * 255

        cv2.imshow('test input', img)
        cv2.imshow('test output', out)
        cv2.imwrite(os.path.join(FLAGS.save_dir, FLAGS.test_image), out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    tf.app.run()