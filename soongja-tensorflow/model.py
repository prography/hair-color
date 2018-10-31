import os
import time

import imgaug as ia
import numpy as np
import tensorflow as tf
from dataloader import Dataset
from imgaug import augmenters as iaa
from ops import *
from utils import draw_results


class Network(object):
    def __init__(self, input_height, input_width):
        self.inputs = tf.placeholder(tf.float32, [None, input_height, input_width, 3], name='inputs')
        self.inputs_std = tf.stack(tf.map_fn(tf.image.per_image_standardization, self.inputs), name='inputs_std')
        self.targets = tf.placeholder(tf.int32, [None, input_height, input_width, 1], name='targets')

        self.logits = tf.identity(self.feedforward(self.inputs_std), name='logits')
        self.preds = tf.cast(tf.expand_dims(tf.argmax(self.logits, axis=3), 3), tf.float32, name='preds')
        #                        dtype      scale            shape
        # self.inputs:         float32      0~255    (N,224,224,3)
        # self.inputs_std      float32   -std~std    (N,224,224,3)
        # self.targets:          int32        0,1    (N,224,224,1)

        # self.logits:         float32          -    (N,224,224,2)
        # self.preds:          float32        0,1    (N,224,224,1)

        reshaped_preds = tf.reshape(tf.argmax(self.logits, axis=3, output_type=tf.int32), [-1])
        reshaped_targets = tf.reshape(self.targets, [-1])
        #                        dtype      scale            shape
        # reshaped_preds:        int32        0,1      (N*224*224)
        # reshaped_targets:      int32        0,1      (N*224*224)
        self.iou, self.iou_op = tf.metrics.mean_iou(labels=reshaped_targets, predictions=reshaped_preds,
                                                    num_classes=2, name='IOU')

        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="IOU")
        self.running_vars_initializer = tf.variables_initializer(var_list=running_vars)

    def feedforward(self, inputs):
        with tf.variable_scope("network"):
            # Encoder blocks
            h0 = conv2d(inputs, "in_conv", 32, 3, 2)
            h1 = depthwise_separable_conv2d(h0, "ds_conv1", 64)
            h2 = depthwise_separable_conv2d(h1, "ds_conv2", 128, downsample=True)
            h3 = depthwise_separable_conv2d(h2, "ds_conv3", 128)
            h4 = depthwise_separable_conv2d(h3, "ds_conv4", 256, downsample=True)
            h5 = depthwise_separable_conv2d(h4, "ds_conv5", 256)
            h6 = depthwise_separable_conv2d(h5, "ds_conv6", 512, downsample=True)
            h7 = depthwise_separable_conv2d(h6, "ds_conv7", 512)
            h8 = depthwise_separable_conv2d(h7, "ds_conv8", 512)
            h9 = depthwise_separable_conv2d(h8, "ds_conv9", 512)
            h10 = depthwise_separable_conv2d(h9, "ds_conv10", 512)
            h11 = depthwise_separable_conv2d(h10, "ds_conv11", 512)
            h12 = depthwise_separable_conv2d(h11, "ds_conv12", 1024, downsample=True)
            h13 = depthwise_separable_conv2d(h12, "ds_conv13", 1024)

            # Decoder blocks
            h14 = upsample_with_addition(h13, "up1", h11, 1024)
            h15 = inv_depthwise_separable_conv2d(h14, "inv_ds_conv1", 64)
            h16 = upsample_with_addition(h15, "up2", h5, 64)
            h17 = inv_depthwise_separable_conv2d(h16, "inv_ds_conv2", 64)
            h18 = upsample_with_addition(h17, "up3", h3, 64)
            h19 = inv_depthwise_separable_conv2d(h18, "inv_ds_conv3", 64)
            h20 = upsample_with_addition(h19, "up4", h1, 64)
            h21 = inv_depthwise_separable_conv2d(h20, "inv_ds_conv4", 64)
            h22 = upsample_only(h21, "up5")
            h23 = inv_depthwise_separable_conv2d(h22, "inv_ds_conv5", 64)

            logits = conv2d(h23, "out_conv", 2, 1, 1)

            return logits


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
        self.prefix = config.prefix
        self.validation_step = config.validation_step

        self.build_model()

    def build_model(self):
        self.net = Network(self.input_height, self.input_width)

        """ Loss """
        reshaped_logits = tf.reshape(self.net.logits, [-1, 2])
        reshaped_targets = tf.reshape(self.net.targets, [-1])
        #                        dtype      scale            shape
        # reshaped_logits:     float32        0~1    (N*224*224,2)
        # reshaped_targets:      int32        0,1      (N*224*224)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=reshaped_targets, logits=reshaped_logits)
        self.loss_op = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

        """ Train """
        t_vars = tf.trainable_variables()
        optim = tf.train.AdamOptimizer(self.learning_rate)  # different from the paper
        self.train_op = optim.minimize(self.loss_op, var_list=t_vars)

        """ Summary """
        tf.summary.scalar("loss", self.loss_op)
        tf.summary.scalar("IOU", self.net.iou)
        tf.summary.image("std_input", self.net.inputs_std)
        self.summary_op = tf.summary.merge_all()

    def train(self):
        # initialize
        self.sess.run(tf.global_variables_initializer())

        # 여기서 저장되는 그래프는 껍데기일 뿐. freeze graph를 통해 체크포인트와 결합해야 한다.
        tf.train.write_graph(self.sess.graph_def, logdir=os.path.join(self.graph_dir, self.model_dir()),
                             name='empty_graph.pbtxt', as_text=True)
        writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.model_dir()), self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=3)

        # dataset
        dataset = Dataset(self.input_height, self.input_width, self.batch_size, self.data_dir)

        # imgaug
        aug_seq = iaa.SomeOf((1, 2),
            [
                iaa.OneOf([iaa.Affine(rotate=(-30, 30), name="Rotate"),
                           iaa.Affine(scale=(0.3, 1.0), name="Scale")]),
                iaa.OneOf([iaa.Multiply((0.5, 1.5), name="Multiply"),
                           iaa.GaussianBlur((0, 3.0), name="GaussianBlur"),
                           iaa.CoarseDropout((0.05, 0.2), size_percent=(0.01, 0.1), name="CoarseDropout")])
            ])

        # deactivate certain augmenters for binmasks
        # on binmasks, we only want to execute crop, horizontal flip, and affine transformation
        def activator_binmasks(images, augmenter, parents, default):
            if augmenter.name in ["Multiply", "GaussianBlur", "CoarseDropout"]:
                return False
            else:
                return default
        hooks_binmasks = ia.HooksImages(activator=activator_binmasks)

        # load checkpoint
        counter = 0
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # list for saving past IOUs
        val_ious = []

        # training
        start_time = time.time()
        for epoch_i in range(self.epoch):
            dataset.reset_batch_pointer()

            for batch_i in range(dataset.num_batches_in_epoch()):
                self.sess.run(self.net.running_vars_initializer)

                batch_inputs, batch_targets = dataset.next_batch()
                #                        dtype      scale            shape
                # batch_inputs:          uint8      0~255    (N,224,224,3)
                # batch_targets:         uint8      0,255    (N,224,224,1)

                batch_inputs = np.reshape(batch_inputs,
                                          (self.batch_size, self.input_height, self.input_width, 3))
                batch_targets = np.reshape(batch_targets,
                                           (self.batch_size, self.input_height, self.input_width, 1))

                aug_seq_det = aug_seq.to_deterministic()

                # augmentation
                batch_inputs = aug_seq_det.augment_images(batch_inputs)
                batch_targets = aug_seq_det.augment_images(batch_targets, hooks=hooks_binmasks)

                # per image standardization
                _batch_inputs = batch_inputs.astype(np.float32)
                _batch_targets = np.multiply(batch_targets, 1.0 / 255).astype(np.int32)

                feed_dict = {self.net.inputs: _batch_inputs, self.net.targets: _batch_targets}

                _, step_loss = self.sess.run([self.train_op, self.loss_op], feed_dict=feed_dict)

                self.sess.run(self.net.iou_op, feed_dict=feed_dict)
                step_iou = self.sess.run(self.net.iou, feed_dict=feed_dict)

                step_summary = self.sess.run(self.summary_op, feed_dict=feed_dict)

                counter += 1
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %5.2f, Loss: %.5f, IOU: %.5f"
                      % (epoch_i, self.epoch, batch_i, dataset.num_batches_in_epoch(),
                         time.time() - start_time, step_loss, step_iou))
                writer.add_summary(step_summary, global_step=counter)

                # Validation
                if batch_i % self.validation_step == 0:
                    self.sess.run(self.net.running_vars_initializer)

                    val_inputs, val_targets = dataset.val_set()
                    #                        dtype      scale            shape
                    # val_inputs:            uint8      0~255    (N,224,224,3)
                    # val_targets:           uint8      0,255    (N,224,224,1)

                    val_inputs = np.reshape(val_inputs,
                                            (-1 , self.input_height, self.input_width, 3))
                    val_targets = np.reshape(val_targets,
                                            (-1 , self.input_height, self.input_width, 1))

                    _val_inputs = val_inputs.astype(np.float32)
                    _val_targets = np.multiply(val_targets, 1.0 / 255).astype(np.int32)

                    feed_dict = {self.net.inputs: _val_inputs, self.net.targets: _val_targets}

                    self.sess.run(self.net.iou_op, feed_dict=feed_dict)
                    val_iou= self.sess.run(self.net.iou, feed_dict=feed_dict)

                    # matplotlib
                    val_preds = self.sess.run(tf.squeeze(self.net.preds, axis=3), feed_dict=feed_dict)

                    draw_results(val_iou, val_inputs, np.squeeze(val_targets, axis=3), val_preds,
                                 counter, self.sample_dir, self.model_dir(), num_samples=10)

                    print()
                    print("=====================================================================")
                    val_ious.append(val_iou)
                    max_iou = max(val_ious)
                    print("IOUs in time:", val_ious)
                    print("Best IOU: %.4f" % max_iou)
                    print("Validation IOU: %.4f" % val_iou)
                    if val_iou >= max_iou:
                        self.save(self.checkpoint_dir, counter)
                        print("Validation IOU exceeded the current best. Saved checkpoints!!!")
                    else:
                        print("IOU hasn't increased. Did not save checkpoints...")
                    print("=====================================================================")
                    print()

    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.prefix, self.batch_size,
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
            print(" [*] Success to read {}".format(os.path.join(checkpoint_dir, ckpt_name)))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
