import os
import time
import tensorflow as tf
from ops import *
from dataloader import Dataset
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
        self.validation_step = config.validation_step

        self.build_model()

    def build_model(self):
        self.dataset = Dataset(batch_size=self.batch_size, folder=self.data_dir)

        """ Inputs and predictions"""
        self.inputs = tf.placeholder(tf.float32, [None, self.input_height, self.input_width, 3], name='inputs')
        self.targets = tf.placeholder(tf.uint8, [None, self.input_height, self.input_width, 1], name='targets')

        self.logits = tf.identity(self.net(self.inputs), name='logits') # for loss op
        self.preds = tf.cast(tf.expand_dims(tf.argmax(self.logits, axis=3) * 255, 3), tf.uint8, name='preds')

        #                 dtype      scale            shape
        # self.inputs:  float32        0~1    (N,224,224,3)
        # self.targets:   uint8      0,255    (N,224,224,1)
        # self.logits:  float32        0~1    (N,224,224,2)
        # self.preds:     uint8      0,255    (N,224,224,1)

        """ Loss """
        reshaped_logits = tf.reshape(self.logits, [-1, 2])
        reshaped_targets = tf.cast(tf.reshape(self.targets, [-1]) / 255, tf.int64) # for softmax cross entropy

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=reshaped_logits, labels=reshaped_targets)
        self.loss_op = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

        """ Accuracy """
        correct_pred = tf.cast(tf.equal(self.preds, self.targets), tf.float32)
        self.accuracy_op = tf.reduce_mean(correct_pred, name='accuracy')

        """ Train """
        t_vars = tf.trainable_variables()
        optim = tf.train.AdadeltaOptimizer(self.learning_rate, rho=0.95, epsilon=1e-07)
        self.train_op = optim.minimize(self.loss_op, var_list=t_vars)

        """ Summary """
        tf.summary.image("preds", self.preds, max_outputs=1)
        tf.summary.image("targets", self.targets, max_outputs=1)
        tf.summary.scalar("loss", self.loss_op)
        tf.summary.scalar("accuracy", self.accuracy_op)
        self.summary_op = tf.summary.merge_all()

    def train(self):
        self.sess.run(tf.global_variables_initializer())

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

        val_accuracies = []

        start_time = time.time()
        for epoch_i in range(self.epoch):
            self.dataset.reset_batch_pointer()

            for batch_i in range(self.dataset.num_batches_in_epoch()):
                batch_inputs, batch_targets = self.dataset.next_batch()

                _, step_loss, step_accuracy, step_summary = \
                    self.sess.run([self.train_op, self.loss_op, self.accuracy_op, self.summary_op],
                                   feed_dict={self.inputs: batch_inputs, self.targets: batch_targets})

                counter += 1
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss: %.8f, Accuracy: %.8f"
                      % (epoch_i, self.epoch, batch_i, self.dataset.num_batches_in_epoch(),
                         time.time() - start_time, step_loss, step_accuracy))
                writer.add_summary(step_summary, global_step=counter)

                # Validation
                if batch_i % self.validation_step == 0:
                    val_inputs, val_targets = self.dataset.val_set()

                    val_accuracy = self.sess.run(self.accuracy_op, feed_dict={self.inputs: val_inputs,
                                                                              self.targets: val_targets})

                    # matplotlib reconstruction
                    val_inputs, val_masks, val_logits_a, val_logits_b, val_preds = \
                    self.sess.run([self.inputs,
                                   tf.squeeze(self.targets, axis=3),
                                   self.logits[:,:,:,0],
                                   self.logits[:,:,:,1],
                                   tf.squeeze(self.preds, axis=3)], feed_dict={self.inputs: val_inputs,
                                                                               self.targets: val_targets})

                    draw_results(val_inputs, val_masks, val_logits_a, val_logits_b, val_preds,
                                 val_accuracy, epoch_i, batch_i, self.sample_dir, self.model_dir(), num_samples=10)

                    print()
                    print("=====================================================================")
                    print("Validation accuracy: %.8f" % val_accuracy)
                    val_accuracies.append(val_accuracy)
                    print("Accuracies in time:", [val_accuracies[i] for i in range(len(val_accuracies))])
                    max_acc = max(val_accuracies)
                    print("Best accuracy: %.8f" % max_acc)
                    if val_accuracy >= max_acc:
                        self.save(self.checkpoint_dir, counter)
                        print("Validation accuracy exceeded the current best. Saved checkpoints!!!")
                    else:
                        print("Accuracy hasn't increased. Did not save checkpoints...")
                    print("=====================================================================")
                    print()

    def net(self, inputs):
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