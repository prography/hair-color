import tensorflow as tf


def batch_norm(inputs, scope, epsilon=1e-5, momentum=0.99, is_training=True):
    with tf.variable_scope(scope):
        return tf.contrib.layers.batch_norm(inputs,
                                             decay=momentum,
                                             epsilon=epsilon,
                                             updates_collections=None,
                                             is_training=is_training,
                                             scope=scope)


def depthwise_conv2d(inputs, scope, filter_size=3, channel_multiplier=1, strides=1):
    with tf.variable_scope(scope):
        w = tf.get_variable('w', [filter_size, filter_size, inputs.get_shape()[-1], channel_multiplier],
                            initializer=tf.truncated_normal_initializer(stddev=0.01))

        return tf.nn.depthwise_conv2d(inputs, w, strides=[1, strides, strides, 1], padding='SAME', rate=[1 ,1])


def conv2d(inputs, scope, num_filters, filter_size=1, strides=1):
    with tf.variable_scope(scope):
        w = tf.get_variable('w', [filter_size, filter_size, inputs.get_shape()[-1], num_filters],
                            initializer=tf.truncated_normal_initializer(stddev=0.01))

        return tf.nn.conv2d(inputs, w, strides=[1, strides, strides, 1], padding='SAME')


def upsample_with_addition(inputs, incoming_enc_layers, scope, num_filters):
    with tf.variable_scope(scope):
        upsampled = tf.image.resize_nearest_neighbor(inputs, size=[inputs.get_shape()[1] * 2, inputs.get_shape()[2] * 2])
        # align_corners=True는 linear, bilinear등에서만 효과있다.

        w = tf.get_variable('w', [1, 1, incoming_enc_layers.get_shape()[-1], num_filters],
                            initializer=tf.truncated_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(incoming_enc_layers, w, strides=[1, 1, 1, 1], padding='SAME')

        return tf.add(upsampled, conv)


def upsample_only(inputs, scope):
    with tf.variable_scope(scope):
        return tf.image.resize_nearest_neighbor(inputs, size=[inputs.get_shape()[1] * 2, inputs.get_shape()[2] * 2])

def depthwise_seperable_conv2d(inputs, scope, num_filters, downsample=False, is_training=True):

    strides = 2 if downsample else 1

    with tf.variable_scope(scope):
        dw_conv = depthwise_conv2d(inputs, "dw_conv", strides=strides)
        dw_bn = batch_norm(dw_conv, "dw_bn", is_training=is_training)
        relu = tf.nn.relu(dw_bn)
        pw_conv = conv2d(relu, "pw_conv", num_filters)
        pw_bn = batch_norm(pw_conv, "pw_bn", is_training=is_training)

        return tf.nn.relu(pw_bn)


def inv_depthwise_seperable_conv2d(inputs, scope, num_filters):
    with tf.variable_scope(scope):
        dw_conv = depthwise_conv2d(inputs, "inv_dw_conv")
        pw_conv = conv2d(dw_conv, "inv_pw_conv", num_filters)

        return tf.nn.relu(pw_conv)