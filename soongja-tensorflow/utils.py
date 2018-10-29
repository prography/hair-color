import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim


def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def draw_results(IOU, inputs, targets, preds,
                 epoch, batch_num, sample_dir, model_dir, num_samples):
    fig, axs = plt.subplots(3, num_samples, figsize=(num_samples * 3, 10))
    fig.suptitle("IOU: %.4f" % IOU, fontsize=20)
    for example_i in range(num_samples):
        axs[0][example_i].imshow(inputs[example_i])
        axs[1][example_i].imshow(targets[example_i], cmap='gray')
        axs[2][example_i].imshow(preds[example_i], cmap='gray')

    sample_dir = os.path.join(sample_dir, model_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    plt.savefig(os.path.join(sample_dir, 'val_epoch-{}_batch-{}.jpg'.format(epoch, batch_num)))
