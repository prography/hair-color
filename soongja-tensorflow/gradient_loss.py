import cv2
import tensorflow as tf
import matplotlib.pylab as plt
from skimage.io import imread
from sklearn.preprocessing import normalize
from skimage.color import rgb2gray
from skimage import filters


def image_gradient(image):
    edges_x = filters.sobel_h(image)
    edges_y = filters.sobel_v(image)
    edges_x = normalize(edges_x)
    edges_y = normalize(edges_y)
    return edges_x, edges_y


def image_gradient_loss(_input, pred):
    # _input 224,224,3
    # pred 224,224,1
    # tf.map_fn으로 batch에 적용할 거임

    h, w = _input.shape[0], pred.shape[1]

    gray_input = rgb2gray(_input)
    Ix, Iy = image_gradient(gray_input)
    Mx, My = image_gradient(pred)
    IM = tf.ones([h, w], dtype=tf.float64) - tf.square(tf.add(tf.multiply(Ix, Mx), tf.multiply(Iy, My)))
    Mmag = tf.sqrt(tf.add(tf.square(Mx), tf.square(My)))

    numerator = tf.reduce_sum(tf.multiply(Mmag, IM))
    denominator = tf.reduce_sum(Mmag)

    return numerator / denominator


im = cv2.imread('saeron.jpg')
mask = cv2.imread('saeron2.jpg', 0)

with tf.Session() as sess:
    print(sess.run(image_gradient_loss(im, mask)))
