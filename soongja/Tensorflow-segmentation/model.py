import tensorflow as tf
import utils
from libs.activations import lrelu


class Layer():
    def create_layer(self, input):
        pass

    def create_layer_reversed(self, input):
        pass

    def get_description(self):
        pass


class Conv2d(Layer):
    # global things...
    layer_index = 0

    def __init__(self, kernel_size, strides, output_channels, name):
        self.kernel_size = kernel_size
        self.strides = strides
        self.output_channels = output_channels
        self.name = name

    @staticmethod
    def reverse_global_variables():
        Conv2d.layer_index = 0

    def create_layer(self, input):
        # print('convd2: input_shape: {}'.format(utils.get_incoming_shape(input)))
        self.input_shape = utils.get_incoming_shape(input)
        number_of_input_channels = self.input_shape[3]

        with tf.variable_scope('conv', reuse=False):
            W = tf.get_variable('W{}'.format(self.name[-3:]),
                                shape=(self.kernel_size, self.kernel_size, number_of_input_channels, self.output_channels))
            b = tf.Variable(tf.zeros([self.output_channels]))
        self.encoder_matrix = W
        Conv2d.layer_index += 1

        output = tf.nn.conv2d(input, W, strides=self.strides, padding='SAME')

        # print('convd2: output_shape: {}'.format(utils.get_incoming_shape(output)))

        output = lrelu(tf.add(tf.contrib.layers.batch_norm(output), b))

        return output

    def create_layer_reversed(self, input, prev_layer=None):
        # print('convd2_transposed: input_shape: {}'.format(utils.get_incoming_shape(input)))
        # W = self.encoder[layer_index]
        with tf.variable_scope('conv', reuse=True):
            W = tf.get_variable('W{}'.format(self.name[-3:]))
            b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))

        # if self.strides==[1, 1, 1, 1]:
        #     print('Now')
        #     output = lrelu(tf.add(
        #         tf.nn.conv2d(input, W,strides=self.strides, padding='SAME'), b))
        # else:
        #     print('1Now1')
        output = tf.nn.conv2d_transpose(
            input, W,
            tf.stack([tf.shape(input)[0], self.input_shape[1], self.input_shape[2], self.input_shape[3]]),
            strides=self.strides, padding='SAME')

        Conv2d.layer_index += 1
        output.set_shape([None, self.input_shape[1], self.input_shape[2], self.input_shape[3]])

        output = lrelu(tf.add(tf.contrib.layers.batch_norm(output), b))
        # print('convd2_transposed: output_shape: {}'.format(utils.get_incoming_shape(output)))

        return output

    def get_description(self):
        return "C{},{},{}".format(self.kernel_size, self.output_channels, self.strides[1])


class MaxPool2d(Layer):
    def __init__(self, kernel_size, name, skip_connection=False):
        self.kernel_size = kernel_size
        self.name = name
        self.skip_connection = skip_connection

    def create_layer(self, input):
        return utils.max_pool_2d(input, self.kernel_size)

    def create_layer_reversed(self, input, prev_layer=None):
        if self.skip_connection:
            input = tf.add(input, prev_layer)

        return utils.upsample_2d(input, self.kernel_size)

    def get_description(self):
        return "M{}".format(self.kernel_size)


class Network:
    def __init__(self, FLAGS, layers=None, per_image_standardization=True, skip_connections=True):
        # Define network - ENCODER (decoder will be symmetric).

        if layers == None:
            layers = []
            layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_1_1'))
            layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_1_2'))
            layers.append(MaxPool2d(kernel_size=2, name='max_1', skip_connection=skip_connections))

            layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_2_1'))
            layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_2_2'))
            layers.append(MaxPool2d(kernel_size=2, name='max_2', skip_connection=skip_connections))

            layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_3_1'))
            layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_3_2'))
            layers.append(MaxPool2d(kernel_size=2, name='max_3'))

        self.inputs = tf.placeholder(tf.float32, [None, FLAGS.input_height, FLAGS.input_width, FLAGS.input_channels],
                                     name='inputs')
        self.targets = tf.placeholder(tf.float32, [None, FLAGS.input_height, FLAGS.input_width, 1], name='targets')
        self.is_training = tf.placeholder_with_default(False, [], name='is_training')
        self.description = ""

        self.layers = {}

        if per_image_standardization:
            list_of_images_norm = tf.map_fn(tf.image.per_image_standardization, self.inputs)
            net = tf.stack(list_of_images_norm)
        else:
            net = self.inputs

        # ENCODER
        for layer in layers:
            self.layers[layer.name] = net = layer.create_layer(net)
            self.description += "{}".format(layer.get_description())

        print("Current input shape: ", net.get_shape())

        layers.reverse()
        Conv2d.reverse_global_variables()

        # DECODER
        for layer in layers:
            net = layer.create_layer_reversed(net, prev_layer=self.layers[layer.name])

        self.segmentation_result = tf.sigmoid(net)

        # segmentation_as_classes = tf.reshape(self.y, [50 * self.IMAGE_HEIGHT * self.IMAGE_WIDTH, 1])
        # targets_as_classes = tf.reshape(self.targets, [50 * self.IMAGE_HEIGHT * self.IMAGE_WIDTH])
        # print(self.y.get_shape())
        # self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(segmentation_as_classes, targets_as_classes))
        print('segmentation_result.shape: {}, targets.shape: {}'.format(self.segmentation_result.get_shape(),
                                                                        self.targets.get_shape()))

        # MSE loss
        self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.segmentation_result - self.targets)))
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)
        with tf.name_scope('accuracy'):
            argmax_probs = tf.round(self.segmentation_result)  # 0x1
            correct_pred = tf.cast(tf.equal(argmax_probs, self.targets), tf.float32)
            self.accuracy = tf.reduce_mean(correct_pred)

            tf.summary.scalar('accuracy', self.accuracy)

        self.summaries = tf.summary.merge_all()