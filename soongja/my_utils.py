import io
import matplotlib.pyplot as plt

def draw_results(test_inputs, test_targets, test_segmentation, batch_num, sample_dir):
    n_examples_to_plot = 12
    fig, axs = plt.subplots(4, n_examples_to_plot, figsize=(n_examples_to_plot * 3, 10))
    for example_i in range(n_examples_to_plot):
        axs[0][example_i].imshow(test_inputs[example_i])
        axs[1][example_i].imshow(test_targets[example_i].astype(np.float32), cmap='gray')
        axs[2][example_i].imshow(test_segmentation[example_i].astype(np.float32), cmap='gray')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    IMAGE_PLOT_DIR = sample_dir

    plt.savefig('{}/figure{}.jpg'.format(IMAGE_PLOT_DIR, batch_num))
    return buf
