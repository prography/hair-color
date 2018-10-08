import datetime
import io
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from imgaug import augmenters as iaa
from imgaug import imgaug
from dataloader import Dataset
from model import Network

np.set_printoptions(threshold=np.nan)


def draw_results(test_inputs, test_targets, test_segmentation, test_accuracy, network, batch_num):
    n_examples_to_plot = 12
    fig, axs = plt.subplots(4, n_examples_to_plot, figsize=(n_examples_to_plot * 3, 10))
    fig.suptitle("Accuracy: {}, {}".format(test_accuracy, network.description), fontsize=20)
    for example_i in range(n_examples_to_plot):
        axs[0][example_i].imshow(test_inputs[example_i], cmap='gray')
        axs[1][example_i].imshow(test_targets[example_i].astype(np.float32), cmap='gray')
        axs[2][example_i].imshow(
            np.reshape(test_segmentation[example_i], [network.IMAGE_HEIGHT, network.IMAGE_WIDTH]),
            cmap='gray')

        test_image_thresholded = np.array(
            [0 if x < 0.5 else 255 for x in test_segmentation[example_i].flatten()])
        axs[3][example_i].imshow(
            np.reshape(test_image_thresholded, [network.IMAGE_HEIGHT, network.IMAGE_WIDTH]),
            cmap='gray')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    IMAGE_PLOT_DIR = 'image_plots/'
    if not os.path.exists(IMAGE_PLOT_DIR):
        os.makedirs(IMAGE_PLOT_DIR)

    plt.savefig('{}/figure{}.jpg'.format(IMAGE_PLOT_DIR, batch_num))
    return buf


class Trainer:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.input_height = FLAGS.input_height
        self.input_width = FLAGS.input_width
        self.input_channels = FLAGS.input_channels

        self.epoch = FLAGS.epoch
        self.batch_size = FLAGS.batch_size
        # self.learning_rate = FLAGS.learning_rate

        self.graph_dir = FLAGS.graph_dir
        self.log_dir = FLAGS.log_dir
        self.checkpoint_dir = FLAGS.checkpoint_dir

        self.test_step = FLAGS.test_step

    def train(self):
        self.net = Network(self.FLAGS)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")

        # create directory for saving models
        os.makedirs(os.path.join(self.checkpoint_dir, timestamp), exist_ok=True)

        dataset = Dataset(folder='data{}_{}'.format(self.input_height, self.input_width), include_hair=False,
                          batch_size=self.batch_size)

        inputs, targets = dataset.next_batch()
        print(inputs.shape, targets.shape)

        # augmentation_seq = iaa.Sequential([
        #     iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
        #     iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        #     iaa.GaussianBlur(sigma=(0, 2.0))  # blur images with a sigma of 0 to 3.0
        # ])

        augmentation_seq = iaa.Sequential([
            iaa.Crop(px=(0, 16), name="Cropper"),  # crop images from each side by 0 to 16px (randomly chosen)
            iaa.Fliplr(0.5, name="Flipper"),
            iaa.GaussianBlur((0, 3.0), name="GaussianBlur"),
            iaa.Dropout(0.02, name="Dropout"),
            iaa.AdditiveGaussianNoise(scale=0.01 * 255, name="GaussianNoise"),
            iaa.Affine(translate_px={"x": (-self.input_height // 3, self.input_width // 3)}, name="Affine")
        ])

        # change the activated augmenters for binary masks,
        # we only want to execute horizontal crop, flip and affine transformation
        def activator_binmasks(images, augmenter, parents, default):
            if augmenter.name in ["GaussianBlur", "Dropout", "GaussianNoise"]:
                return False
            else:
                # default value for all other augmenters
                return default

        hooks_binmasks = imgaug.HooksImages(activator=activator_binmasks)

        with tf.Session() as sess:

            tf.train.write_graph(sess.graph_def, logdir=self.graph_dir, name='graph.pbtxt', as_text=True)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            summary_writer = tf.summary.FileWriter('{}/{}'.format(self.log_dir, timestamp),
                                                   graph=tf.get_default_graph())

            ckpt = tf.train.get_checkpoint_state(os.path.join(self.checkpoint_dir, timestamp))
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Checkpoint loaded: %s" % ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            test_accuracies = []
            # Fit all training data
            global_start = time.time()
            for epoch_i in range(self.epoch):
                dataset.reset_batch_pointer()

                for batch_i in range(dataset.num_batches_in_epoch()):
                    batch_num = epoch_i * dataset.num_batches_in_epoch() + batch_i + 1

                    augmentation_seq_deterministic = augmentation_seq.to_deterministic()

                    start = time.time()
                    batch_inputs, batch_targets = dataset.next_batch()
                    batch_inputs = np.reshape(batch_inputs,
                                              (dataset.batch_size, self.input_height, self.input_width, 1))
                    batch_targets = np.reshape(batch_targets,
                                               (dataset.batch_size, self.input_height, self.input_width, 1))

                    batch_inputs = augmentation_seq_deterministic.augment_images(batch_inputs)
                    batch_inputs = np.multiply(batch_inputs, 1.0 / 255)

                    batch_targets = augmentation_seq_deterministic.augment_images(batch_targets, hooks=hooks_binmasks)

                    loss, _ = sess.run([self.net.cost, self.net.train_op],
                                       feed_dict={self.net.inputs: batch_inputs, self.net.targets: batch_targets,
                                                  self.net.is_training: True})
                    end = time.time()
                    print('{}/{}, epoch: {}, loss: {}, batch time: {}'.format(batch_num,
                                                                              self.epoch * dataset.num_batches_in_epoch(),
                                                                              epoch_i, loss, end - start))

                    # test
                    if batch_num % self.test_step == 0 or batch_num == self.epoch * dataset.num_batches_in_epoch():
                        test_inputs, test_targets = dataset.test_set
                        # test_inputs, test_targets = test_inputs[:100], test_targets[:100]

                        test_inputs = np.reshape(test_inputs, (-1, self.input_height, self.input_width, 1))
                        test_targets = np.reshape(test_targets, (-1, self.input_height, self.input_width, 1))
                        test_inputs = np.multiply(test_inputs, 1.0 / 255)

                        print(test_inputs.shape)
                        summary, test_accuracy = sess.run([self.net.summaries, self.net.accuracy],
                                                          feed_dict={self.net.inputs: test_inputs,
                                                                     self.net.targets: test_targets,
                                                                     self.net.is_training: False})

                        summary_writer.add_summary(summary, batch_num)

                        print('Step {}, test accuracy: {}'.format(batch_num, test_accuracy))
                        test_accuracies.append((test_accuracy, batch_num))
                        print("Accuracies in time: ", [test_accuracies[x][0] for x in range(len(test_accuracies))])
                        max_acc = max(test_accuracies)
                        print("Best accuracy: {} in batch {}".format(max_acc[0], max_acc[1]))
                        print("Total time: {}".format(time.time() - global_start))

                        # Plot example reconstructions
                        n_examples = 12
                        test_inputs, test_targets = dataset.test_inputs[:n_examples], dataset.test_targets[:n_examples]
                        test_inputs = np.multiply(test_inputs, 1.0 / 255)

                        test_segmentation = sess.run(self.net.segmentation_result, feed_dict={
                            self.net.inputs: np.reshape(test_inputs,
                                                       [n_examples, self.input_height, self.input_width, 1])})

                        # Prepare the plot
                        test_plot_buf = draw_results(test_inputs, test_targets, test_segmentation, test_accuracy, self.net,
                                                     batch_num)

                        # Convert PNG buffer to TF image
                        image = tf.image.decode_png(test_plot_buf.getvalue(), channels=4)

                        # Add the batch dimension
                        image = tf.expand_dims(image, 0)

                        # Add image summary
                        image_summary_op = tf.summary.image("plot", image)

                        image_summary = sess.run(image_summary_op)
                        summary_writer.add_summary(image_summary)

                        if test_accuracy >= max_acc[0]:
                            checkpoint_path = os.path.join('checkpoints', timestamp, 'model.ckpt')
                            saver.save(sess, checkpoint_path, global_step=batch_num)