import time
import cv2
import numpy as np
import tensorflow as tf
from model import Network

flags = tf.app.flags
flags.DEFINE_integer("input_height", 224, "The height of input image")
flags.DEFINE_integer("input_width", 224, "The width of input image")
flags.DEFINE_string("checkpoint_dir", "checkpoints/hairdata_4_224_224", "Directory where checkpoints are saved")

FLAGS = flags.FLAGS


def alpha_blend(frame, mask, b, g, r):

    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    colored_mask[np.where(mask != 0)] = [b, g, r] # BGR!!!
    foreground = (0.9 * frame + 0.1 * colored_mask).astype(np.uint8)
    alpha_hand = cv2.bitwise_and(foreground, foreground, mask=mask)

    mask_inv = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(frame, frame, mask=mask_inv)

    return cv2.add(alpha_hand, background)


def nothing(x):
    pass


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

        # color
        palette = np.zeros((120, 360, 3), np.uint8)
        cv2.namedWindow('demo', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('palette', cv2.WINDOW_NORMAL)

        cv2.createTrackbar('R', 'demo', 0, 255, nothing)
        cv2.createTrackbar('G', 'demo', 0, 255, nothing)
        cv2.createTrackbar('B', 'demo', 0, 255, nothing)

        cap = cv2.VideoCapture(0)
        frames = 0
        start_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            if ret:
                frame = frame[60:420, 140:500] # 360, 360

                _input = cv2.resize(frame, (FLAGS.input_height, FLAGS.input_width))
                _input = cv2.cvtColor(_input, cv2.COLOR_BGR2RGB)
                _input = np.expand_dims(_input, axis=0)

                mask = sess.run(net.preds, feed_dict={net.inputs: _input})
                mask = mask[0].astype(np.uint8) * 255
                mask = cv2.resize(mask, (360, 360))
                _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

                r = cv2.getTrackbarPos('R', 'demo')
                g = cv2.getTrackbarPos('G', 'demo')
                b = cv2.getTrackbarPos('B', 'demo')

                palette[:] = [b, g, r]

                out = alpha_blend(frame, mask, b, g, r)

                out = np.vstack((out, palette))

                frames += 1
                fps = frames / (time.time() - start_time)
                cv2.putText(out, 'FPS: %.2f' % fps, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 0], 1)

                # cv2.imshow('palette', palette)
                cv2.imshow('demo', out)
                key = cv2.waitKey(1)
                if key & 0xFF == 27:
                    break

            else:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    tf.app.run()