import os
from PIL import Image

for i in os.listdir("./dataset/mask"):
    image = Image.open(os.path.join("./dataset/mask", i)).convert('L')
    image.save(os.path.join("./dataset/mask", i))