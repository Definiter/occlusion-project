# Make white color transparent.
from constant import *
from PIL import Image, ImageDraw
from matplotlib.pyplot import imshow
import numpy as np
import os

image_names = os.listdir(shapenet_root + 'object_orig/')
print len(image_names)

for i, image_name in enumerate(image_names):
    print '[{}/{}]: {}'.format(i, 10000, image_name)
    img = Image.open(shapenet_root + 'object_orig/' + image_name)

    img = img.convert("RGBA")
    pixdata = img.load()
    for y in xrange(img.size[1]):
        for x in xrange(img.size[0]):
            if pixdata[x, y] == (255, 255, 255, 255):
                pixdata[x, y] = (255, 255, 255, 0)
    img.save(shapenet_root + 'object_nobg/' + image_name, 'PNG')
