import os
import time
import numpy as np
import tensorflow as tf
from ops import *
from dataloader import DataLoader
from utils import draw_results


class MobileHair(object):
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
        # self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Input Image """
        IMAGE_DIR_PATH = os.path.join(self.data_dir, 'images')
        MASK_DIR_PATH = os.path.join(self.data_dir, 'masks')

        image_paths = [os.path.join(IMAGE_DIR_PATH, x) for x in os.listdir(IMAGE_DIR_PATH) if x.endswith('.png')]
        mask_paths = [os.path.join(MASK_DIR_PATH, x) for x in os.listdir(MASK_DIR_PATH) if x.endswith('.png')]

        self.Dataloader_Class = DataLoader(image_paths=image_paths,
                                      mask_paths=mask_paths,
                                      image_size=(self.input_height, self.input_width),
                                      channels=(3, 1),
                                      num_test=12,
                                      crop_size=None,
                                      palette=(0, 255),
                                      seed=777)

        self.images, self.masks = self.Dataloader_Class.train_batch(shuffle=True,
                                                                         augment=False,
                                                                         one_hot_encode=False,
                                                                         batch_size=self.batch_size)
        print(self.images, self.masks)

        """ Logits """
        # self.inputs = tf.placeholder(tf.float32, [None, self.input_height, self.input_width, 3], name='inputs')
        # self.labels = tf.placeholder(tf.int64, [None, self.input_height, self.input_width, 1], name='labels')

        self.logits = self.network(self.images)
        # tf.summary.image("logits image", self.logits)

        reshaped_logits = tf.reshape(self.logits, [-1, 2])
        reshaped_labels = tf.cast(tf.reshape(self.masks, [-1]) / 255, tf.int64)

        self.pred_masks = tf.expand_dims(tf.cast(tf.argmax(self.logits, axis=3), tf.uint8), 3)

        """ Loss """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=reshaped_logits, labels=reshaped_labels)
        self.loss_op = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

        """ Training """
        t_vars = tf.trainable_variables()
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_op, var_list=t_vars)

        """ Summary """
        tf.summary.image("pred_masks", self.pred_masks)
        tf.summary.image("real_masks", self.masks)
        tf.summary.scalar("loss", self.loss_op)
        self.summary_op = tf.summary.merge_all()

    def train(self):
        tf.global_variables_initializer().run()

        tf.train.write_graph(self.sess.graph_def, logdir=self.graph_dir, name='full_graph.pb', as_text=False)
        saver = tf.train.Saver(max_to_keep=5) # 왜 tf.global_variables() 안쓰지
        writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir(), self.sess.graph)
        # get_default_graph()와 sess.graph 바꿔가며 찍어보자 뭐가 다른가.

        # global_step = tf.train.get_or_create_global_step(sess.graph)

        counter = 1
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        start_time = time.time()
        for epoch in range(self.epoch):
            batch_idxs = (len(self.Dataloader_Class.train_image_paths) // self.batch_size) + 1
            for idx in range(batch_idxs):
                image_batch, mask_batch, step_loss, step_summary, _ = self.sess.run([self.images, self.masks,
                                                                                     self.loss_op, self.summary_op, self.train_op])

                counter += 1
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss: %.8f"
                      % (epoch, self.epoch, idx, batch_idxs, time.time() - start_time, step_loss))

                writer.add_summary(step_summary, global_step=counter)

                '''
                # test and save checkpoints
                if idx % self.checkpoint_step == 0 or idx == batch_idx - 1:
                    self.save(self.checkpoint_dir, counter)

                    # test
                    test_batch, test_init_op = dataset.test_batch()
                    sess.run(test_init_op)
                    test_images, test_masks = sess.run(test_batch)

                    # print(tf.shape(test_images))
                    # print(tf.shape(test_masks))


                    # test_images = np.multiply(test_images, 1.0 / 255)

                    test_images = test_images * 1.0 / 255

                    preds = sess.run(self.logits, feed_dict={self.inputs: test_images})
                    test_images = tf.reshape(test_images, (-1, self.input_height, self.input_width, 3))
                    test_masks = tf.reshape(test_masks, (-1, self.input_height, self.input_width, 1))
                    preds = tf.argmax(preds, axis=3)
                    print(preds)
                    # pred_masks = np.reshape(preds, (-1, self.input_height, self.input_width, 1))
                    pred_masks = tf.expand_dims(preds, 3)
                    print(tf.shape(test_images))
                    print(tf.shape(test_masks))
                    print(tf.shape(pred_masks))

                    draw_results(test_images, test_masks, pred_masks, idx, self.sample_dir)
                    print(" [*] Saved checkpoint and test samples")
                    '''
    def network(self, inputs):
        with tf.variable_scope("network"):
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
            print(h15.get_shape(), h5.get_shape())
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
        model_name = "MobileHair.model"
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