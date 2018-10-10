import os
import tensorflow as tf
from my_ops import *
from my_dataloader import DataLoader
from my_utils import *

class MobileHair(object):
    def __init__(self, config):
        self.input_height = config.input_height
        self.input_width = config.input_width

        self.batch_size = config.batch_size
        self.epoch = config.epoch
        self.learning_rate = config.learning_rate

        self.dataset = config.dataset
        self.train_test_ratio = config.train_test_ratio

        self.checkpoint_dir = config.checkpoint_dir
        self.log_dir = config.log_dir
        self.graph_dir = config.graph_dir

        self.data_dir = config.data_dir
        self.sample_dir = config.sample_dir

        self.build_model()

    def build_model(self):
        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.input_height, self.input_width, 3],
                                     name='inputs')
        self.logits = self.network(self.inputs)
        tf.summary.image("logits image", self.logits)
        tf.summary.histogram("logits histogram", self.logits)

        labels = tf.placeholder(tf.int32, [self.batch_size, self.input_height, self.input_width, 1],
                                    name='labels')

        reshaped_logits = tf.reshape(self.logits, [-1, 2])
        reshaped_labels = tf.reshape(self.labels, [-1])

        self.loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(reshaped_logits, reshaped_labels)
        tf.summary.scalar("loss", self.loss)

        self.summary_op = tf.summary.merge_all()

        # self.t_vars = tf.trainable_variables()


    def train(self):

        IMAGE_DIR_PATH = os.path.join(self.data_dir, 'images')
        MASK_DIR_PATH = os.path.join(self.data_dir, 'masks')

        image_paths = [os.path.join(IMAGE_DIR_PATH, x) for x in os.listdir(IMAGE_DIR_PATH) if x.endswith('.png')]
        mask_paths = [os.path.join(MASK_DIR_PATH, x) for x in os.listdir(MASK_DIR_PATH) if x.endswith('.png')]

        dataset = DataLoader(image_paths=image_paths,
                             mask_paths=mask_paths,
                             image_extension=self.image_extension,
                             image_size=(self.input_height, self.input_width),
                             channels=(3, 1),
                             num_test=10,
                             crop_size=None,
                             palette=None,
                             seed=777)

        with tf.Session() as sess:
            tf.train.write_graph(sess.graph_def, logdir=self.graph_dir, name='full_graph.pb', as_text=False)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            # get_default_graph()와 sess.graph 바꿔가며 찍어보자 뭐가 다른가.

            global_step = tf.train.get_or_create_global_step(sess.graph)
            train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_op, global_step=global_step)

            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                print(" [*] Success to read {}".format(ckpt_name))
            else:
                sess.run(tf.global_variables_initializer())
                print(" [*] Failed to find a checkpoint")

            next_batch, init_op = dataset.data_batch(train=True,
                                               shuffle=True,
                                               augment=False,
                                               one_hot_encode=False,
                                               batch_size=self.batch_size)
            sess.run(init_op)

            start_time = time.time()
            for epoch in range(self.epoch):
                batch_idxs = (len(dataset.train_image_paths) // self.batch_size) + 1
                for idx in range(batch_idxs):
                    images, masks = sess.run(next_batch)

                    _, step_loss, step_summary = sess.run([train_op, self.loss_op, self.summary_op],
                                                          feed_dict={self.inputs: images, self.labels: masks})

                    writer.add_summary(step_summary, global_step=sess.run(global_step))
                    print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss: %.8f"
                          % (epoch, self.epoch, idx, batch_idxs, time.time() - start_time, step_loss))

                    # save checkpoint
                    # test
                    if idx % self.checkpoint_step == 0 or idx == batch_idx - 1:
                        saver.save(sess, os.path.join(self.checkpoint_dir, 'MobileHair'),
                                   global_step=sess.run(global_step))

                        test_images ,test_masks = sess.run(dataset.test_set)
                        outputs = sess.run(self.logits, feed_dict={self.inputs: test_images})

                        draw_results(test_images, test_masks, outputs, idx, self.sample_dir)
                        print(" [*] Saved checkpoint and test samples")


    def network(self, inputs):
        with tf.variable_scope("network"):
            # Encoder blocks
            h0 = conv2d(inputs, "in_conv", 32, 3, 2)
            h1 = depthwise_seperable_conv2d(h0, "ds_conv1", 64)
            h2 = depthwise_seperable_conv2d(h1, "ds_conv2", 128, downsample=True)
            h3 = depthwise_seperable_conv2d(h2, "ds_conv3", 128)
            h4 = depthwise_seperable_conv2d(h3, "ds_conv4", 256, downsample=True)
            h5 = depthwise_seperable_conv2d(h4, "ds_conv5", 256)
            h6 = depthwise_seperable_conv2d(h5, "ds_conv6", 512)
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
