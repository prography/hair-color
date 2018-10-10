from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage import filters
from sklearn.preprocessing import normalize

class Image_Gradient:
    def __init__(self, image):
        self.image = image
    def get_gradient(self):
        im = rgb2gray(imread('./dataset/Kim.PNG'))
        edges_x = filters.sobel_h(im)
        edges_y = filters.sobel_v(im)

        edges_x = normalize(edges_x)
        edges_y = normalize(edges_y)

        return edges_x, edges_y

class Loss:
    def __init__(self, image):
        self.image = image
    def get_loss(self):
        pass

#img = Image_Gradient('dataset/Kim.PNG')
#edge = img.get_gradient()
