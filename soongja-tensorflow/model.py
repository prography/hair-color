import os
import time
import numpy as np
import tensorflow as tf
from ops import *
from dataloader import DataLoader
from utils import draw_results


class MobileHairNet(object):
    def __init__(self, sess, config):
        self.sess = sess

        self.input_height = config.input_height
        self.input_width = config.input_width
        self.batch_size = config.batch_size
        self.epoch = config.epoch
        self.learning_rate = config.learning_rate
        self.graph_dir = config.graph_dir
        self.log_dir = config.log_dir
        self.checkpoint_dir = config.checkpoint_dir
        self.sample_dir = config.sample_dir
        self.data_dir = config.data_dir
        self.dataset_name = config.dataset_name
        self.checkpoint_step = config.checkpoint_step
        self.test_step = config.test_step

        self.build_model()

    def build_model(self):

        ##### Input Image #####
        IMAGE_DIR_PATH = os.path.join(self.data_dir, 'images')
        MASK_DIR_PATH = os.path.join(self.data_dir, 'masks')

        image_paths = [os.path.join(IMAGE_DIR_PATH, x) for x in os.listdir(IMAGE_DIR_PATH) if x.endswith('.png')]
        mask_paths = [os.path.join(MASK_DIR_PATH, x) for x in os.listdir(MASK_DIR_PATH) if x.endswith('.png')]

        self.dataloader = DataLoader(image_paths=image_paths, mask_paths=mask_paths, image_extension='png',
                                image_size=(self.input_height, self.input_width), channels=(3, 1), num_test=1100)

        self.iterator, self.n_batches = self.dataloader.train_batch(shuffle=True, augment=False, one_hot_encode=False,
                                                               batch_size=self.batch_size, num_threads=1, buffer=30)

        self.images, self.masks = self.iterator.get_next()
        # dtype and scale
        # self.images: float32, 0~1
        # self.masks: uint8, 0,1
        # self.sess.run(self.iterator.initializer)

        self.inputs = tf.placeholder(tf.float32, [None, self.input_height, self.input_width, 3], name='test_inputs')

        # Logits and predicted masks
        self.logits = self.net(self.inputs) # 원래 self.images가 들어갔음
        self.preds = tf.cast(tf.expand_dims(tf.argmax(self.logits, axis=3) * 255, 3), tf.uint8) # for tf.summary.image

        # validation
        # self.test_inputs = tf.placeholder(tf.float32, [None, self.input_height, self.input_width, 3], name='test_inputs')
        # self.test_preds = tf.expand_dims(tf.argmax(self.net(self.test_inputs), axis=3) * 255, 3)

        # Loss
        reshaped_logits = tf.reshape(self.logits, [-1, 2])
        reshaped_labels = tf.cast(tf.reshape(self.masks, [-1]) / 255, tf.int64) # for softmax cross entropy

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=reshaped_logits, labels=reshaped_labels)
        self.loss_op = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

        """ Training """
        t_vars = tf.trainable_variables()
        optim = tf.train.AdadeltaOptimizer(self.learning_rate, rho=0.95, epsilon=1e-07)
        self.train_op = optim.minimize(self.loss_op, var_list=t_vars)

        """ Summary """
        tf.summary.image("pred_masks", self.preds, max_outputs=1)
        tf.summary.image("real_masks", self.masks, max_outputs=1)
        tf.summary.scalar("loss", self.loss_op)
        self.summary_op = tf.summary.merge_all()

    def train(self):
        print(tf.global_variables())

        tf.global_variables_initializer().run()

        tf.train.write_graph(self.sess.graph_def, logdir=self.graph_dir, name='full_graph.pb', as_text=False)

        writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir(), self.sess.graph)

        self.saver = tf.train.Saver(max_to_keep=5)

        counter = 1
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        start_time = time.time()
        for epoch in range(self.epoch):
            self.sess.run(self.iterator.initializer)
            for idx in range(self.n_batches):

                _, step_loss, step_summary = self.sess.run([self.train_op, self.loss_op, self.summary_op],
                                                           feed_dict={self.inputs: self.images})

                counter += 1
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss: %.8f"
                      % (epoch, self.epoch, idx, self.n_batches, time.time() - start_time, step_loss))
                writer.add_summary(step_summary, global_step=counter)

                # save checkpoints
                if idx % self.checkpoint_step == 0 or idx == self.n_batches - 1:
                    self.save(self.checkpoint_dir, counter)
                    print("Saved checkpoints")

                # validation
                if idx % self.test_step == 0:
                    # with tf.variable_scope("net", reuse=True) as scope:
                    #     scope.reuse_variables()
                        # inputs = tf.placeholder(tf.float32,
                        #          [None, self.input_height, self.input_width, 3], name='inputs')
                        # test data
                    test_images, test_masks = self.dataloader.load_test_data()
                    preds = self.sess.run(self.test_preds, feed_dict={self.test_inputs: test_images})

                        # test_images = tf.convert_to_tensor(np.multiply(test_images, 1.0 / 255), np.float32)
                        # test_masks = tf.convert_to_tensor(test_masks, np.float32)
                        #
                        # logits = self.sess.run(self.net(test_images))
                        # preds = tf.cast(tf.expand_dims(tf.argmax(logits, axis=3) * 255, 3), tf.uint8)
                        #
                    draw_results(test_images, test_masks, preds, idx, self.sample_dir)

    def net(self, inputs):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            # Encoder blocks
            h0 = conv2d(inputs, "in_conv", 32, 3, 2)
            h1 = depthwise_seperable_conv2d(h0, "ds_conv1", 64)
            h2 = depthwise_seperable_conv2d(h1, "ds_conv2", 128, downsample=True)
            h3 = depthwise_seperable_conv2d(h2, "ds_conv3", 128)
            h4 = depthwise_seperable_conv2d(h3, "ds_conv4", 256, downsample=True)
            h5 = depthwise_seperable_conv2d(h4, "ds_conv5", 256)
            h6 = depthwise_seperable_conv2d(h5, "ds_conv6", 512, downsample=True)
            h7 = depthwise_seperable_conv2d(h6, "ds_conv7", 512)
            h8 = depthwise_seperable_conv2d(h7, "ds_conv8", 512)
            h9 = depthwise_seperable_conv2d(h8, "ds_conv9", 512)
            h10 = depthwise_seperable_conv2d(h9, "ds_conv10", 512)
            h11 = depthwise_seperable_conv2d(h10, "ds_conv11", 512)
            h12 = depthwise_seperable_conv2d(h11, "ds_conv12", 1024, downsample=True)
            h13 = depthwise_seperable_conv2d(h12, "ds_conv13", 1024)

            # Decoder blocks
            h14 = upsample_with_addition(h13, h11, "up1", 1024)
            h15 = inv_depthwise_seperable_conv2d(h14, "inv_ds_conv1", 64)
            h16 = upsample_with_addition(h15, h5, "up2", 64)
            h17 = inv_depthwise_seperable_conv2d(h16, "inv_ds_conv2", 64)
            h18 = upsample_with_addition(h17, h3, "up3", 64)
            h19 = inv_depthwise_seperable_conv2d(h18, "inv_ds_conv3", 64)
            h20 = upsample_with_addition(h19, h1, "up4", 64)
            h21 = inv_depthwise_seperable_conv2d(h20, "inv_ds_conv4", 64)
            h22 = upsample_only(h21, "up5")
            h23 = inv_depthwise_seperable_conv2d(h22, "inv_ds_conv5", 64)

            logits = conv2d(h23, "out_conv", 2, 1, 1)

            return logits

    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.input_height, self.input_width)

    def save(self, checkpoint_dir, step):
        model_name = "MobileHairNet.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir())

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir())

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0