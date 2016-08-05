# Generate training dataset and test dataset.
from constant import *
import os
import numpy as np
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import copy
from matplotlib.pyplot import imshow
import random
#%matplotlib inline

#### Parameters. ####
type_str = 'nocrop_obj'
# type_str =  # crop / nocrop / crop_obj / nocrop_obj / aperture
#dataset = [(0.0, 0), (1.0/4, 4), (1.0/3, 3), (1.0/2, 3), (2.0/3, 3), (4.0/5, 3), (9.0/10, 3), (1.0, 1)]
#dataset = [(0.0, 0), (0.1, 10), (0.2, 5), (0.3, 4), (0.4, 3), (0.5, 3), (0.6, 3), (0.7, 3), (0.8, 3), (0.9, 3), (1.0, 1)]
dataset = [(0.0, 0), (0.2, 9), (0.4, 9), (0.6, 9), (0.8, 9), (1.0, 9)] # when crop_obj or nocrop_obj, (size, total_num)

# obj image: always 1024 * 768 pixels
obj_width = 1024
obj_height = 768
obj_ratio = float(obj_width) / obj_height
#####################

# Load labels.
imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
label_to_wnid = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
wnid_to_label = {}
for label in range(len(label_to_wnid)):
    wnid = label_to_wnid[label].split(' ')[0]
    wnid_to_label[wnid] = label

training_dataset = dataset
test_dataset = dataset

train_folders = []
test_folders = []
train_files = []
test_files = []

for (slider_size, slider_num) in training_dataset:
    percent = str(int(100 * slider_size))
    f = open('{}dataset/train_{}_{}.txt'.format(imagenet_root, type_str, percent), 'w')
    train_files.append(f)
    
    folder = '{}dataset/train_{}_{}/'.format(imagenet_root, type_str, percent)
    if not os.path.exists(folder):
        os.makedirs(folder)
    train_folders.append(folder)
        

for (slider_size, slider_num) in test_dataset:
    percent = str(int(100 * slider_size))
    f = open('{}dataset/test_{}_{}.txt'.format(imagenet_root, type_str, percent), 'w')
    test_files.append(f)

    folder = '{}dataset/test_{}_{}/'.format(imagenet_root, type_str, percent)
    if not os.path.exists(folder):
        os.makedirs(folder)
    test_folders.append(folder)
    
if type_str == 'crop_obj' or type_str == 'nocrop_obj':
    obj_images = []
    for image_name in os.listdir(shapenet_root + 'object_nobg/'):
        img = Image.open(shapenet_root + 'object_nobg/' + image_name)
        #img = img.convert("RGBA")
        obj_images.append(img)

# occluder size = slider_size * slider_size
# occluder num = slider_num * slider_num
# path = 'imagenet_root/dataset/train_0/name'
# {wnid_imgid}_{crop/nocrop}_{rect_i(ifcrop)}_{slider_size}_{i}_{j}
def generate_datum(img_orig, path, f, class_id, rects, slider_size, slider_num):
    if type_str == 'crop':
        if slider_size == 0:
            for rect_i, rect in enumerate(rects):
                img = copy.copy(img_orig)
                img = img.crop(rect)
                datum_path = '{}_{}_{}_0_0_0.jpeg'.format(path, type_str, rect_i)
                img.save(datum_path)
                f.write('{} {}\n'.format(datum_path, str(class_id)))
        else:
            for rect_i, rect in enumerate(rects):
                for i in range(slider_num):
                    for j in range(slider_num):
                        img = copy.copy(img_orig)
                        d = ImageDraw.Draw(img)
                        if (slider_num == 1):
                            delta = 1
                        else:
                            delta = (1 - slider_size) / float(slider_num - 1)
                        subrect = [0, 0, 0, 0]
                        subrect[0] = rect[0] + i * (rect[2] - rect[0]) * delta
                        subrect[1] = rect[1] + j * (rect[3] - rect[1]) * delta
                        subrect[2] = subrect[0] + (rect[2] - rect[0]) * slider_size
                        subrect[3] = subrect[1] + (rect[3] - rect[1]) * slider_size
                        d.rectangle(subrect, fill="black", outline=None)
                        img = img.crop(rect)
                        datum_path = '{}_{}_{}_{}_{}_{}.jpeg'.format(path, type_str, rect_i, str(int(100 * slider_size)), str(i), str(j))
                        img.save(datum_path)
                        f.write('{} {}\n'.format(datum_path, str(class_id)))
    if type_str == 'nocrop':
        if slider_size == 0:
            datum_path = '{}_{}_0_0_0_0.jpeg'.format(path, type_str)
            img_orig.save(datum_path)
            f.write('{} {}\n'.format(datum_path, str(class_id)))
        else:
            for i in range(slider_num):
                for j in range(slider_num):
                    img = copy.copy(img_orig)
                    d = ImageDraw.Draw(img)
                    if (slider_num == 1):
                        delta = 1
                    else:
                        delta = (1 - slider_size) / float(slider_num - 1)
                    for rect in rects:
                        subrect = [0, 0, 0, 0]
                        subrect[0] = rect[0] + i * (rect[2] - rect[0]) * delta
                        subrect[1] = rect[1] + j * (rect[3] - rect[1]) * delta
                        subrect[2] = subrect[0] + (rect[2] - rect[0]) * slider_size
                        subrect[3] = subrect[1] + (rect[3] - rect[1]) * slider_size
                        d.rectangle(subrect, fill="black", outline=None)
                    datum_path = '{}_{}_{}_{}_{}_{}.jpeg'.format(path, type_str, 0, str(int(100 * slider_size)), str(i), str(j))
                    img.save(datum_path)
                    f.write('{} {}\n'.format(datum_path, str(class_id)))                    
    
    if type_str == 'crop_obj':
        if slider_size == 0:
            for rect_i, rect in enumerate(rects):
                img = copy.copy(img_orig)
                img = img.crop(rect)
                datum_path = '{}_{}_{}_{}_0.jpeg'.format(path, type_str, rect_i, int(100 * slider_size))
                img.save(datum_path)
                f.write('{} {}\n'.format(datum_path, str(class_id)))
        else:
            for rect_i, rect in enumerate(rects):
                for num in range(slider_num):
                    img = copy.copy(img_orig)
                    random_obj = copy.copy(obj_images[random.randint(0, len(obj_images) - 1)])
                    width = rect[2] - rect[0]
                    height = rect[3] - rect[1]
                    ratio = float(width) / height
                    if ratio >= obj_ratio:
                        resize_scale = float(height) / obj_height * slider_size
                    else:
                        resize_scale = float(width) / obj_width * slider_size
                    random_obj = random_obj.resize((int(obj_width * resize_scale), int(obj_height * resize_scale)), Image.ANTIALIAS)
                    top_left = (random.randint(rect[0], rect[2] - random_obj.size[0]), \
                                random.randint(rect[1], rect[3] - random_obj.size[1]))
                    img.paste(random_obj, top_left, random_obj)
                    img = img.crop(rect)
                    datum_path = '{}_{}_{}_{}_{}.jpeg'.format(path, type_str, rect_i, int(100 * slider_size), num)
                    img.save(datum_path)
                    f.write('{} {}\n'.format(datum_path, str(class_id)))
    if type_str == 'nocrop_obj':
        if slider_size == 0:
            datum_path = '{}_{}_{}_{}_0.jpeg'.format(path, type_str, 0, int(100 * slider_size))
            img_orig.save(datum_path)
            f.write('{} {}\n'.format(datum_path, str(class_id)))
        else:
            for num in range(slider_num):
                img = copy.copy(img_orig)
                for rect_i, rect in enumerate(rects):
                    random_obj = copy.copy(obj_images[random.randint(0, len(obj_images) - 1)])
                    width = rect[2] - rect[0]
                    height = rect[3] - rect[1]
                    ratio = float(width) / height
                    if ratio >= obj_ratio:
                        resize_scale = float(height) / obj_height * slider_size
                    else:
                        resize_scale = float(width) / obj_width * slider_size
                    random_obj = random_obj.resize((int(obj_width * resize_scale), int(obj_height * resize_scale)), Image.ANTIALIAS)
                    top_left = (random.randint(rect[0], rect[2] - random_obj.size[0]), \
                                random.randint(rect[1], rect[3] - random_obj.size[1]))
                    img.paste(random_obj, top_left, random_obj)
                datum_path = '{}_{}_{}_{}_{}.jpeg'.format(path, type_str, rect_i, int(100 * slider_size), num)
                img.save(datum_path)
                f.write('{} {}\n'.format(datum_path, str(class_id)))              
                    
                    
    if type_str == 'aperture':
        if slider_size == 0: # All black.
            for rect_i, rect in enumerate(rects):
                img = copy.copy(img_orig)
                d = ImageDraw.Draw(img)
                d.rectangle(rect, fill="black", outline=None)
                img = img.crop(rect)
                datum_path = '{}_{}_{}_{}_0_0.jpeg'.format(path, type_str, rect_i, str(int(100 * slider_size)))
                img.save(datum_path)
                f.write('{} {}\n'.format(datum_path, str(class_id)))
        elif slider_size == 1: # All visible. 
            for rect_i, rect in enumerate(rects):
                img = copy.copy(img_orig)
                img = img.crop(rect)
                datum_path = '{}_{}_{}_{}_0_0.jpeg'.format(path, type_str, rect_i, str(int(100 * slider_size)))
                img.save(datum_path)
                f.write('{} {}\n'.format(datum_path, str(class_id)))
        else:
            for rect_i, rect in enumerate(rects):
                for i in range(slider_num):
                    for j in range(slider_num):
                        img = copy.copy(img_orig)
                        d = ImageDraw.Draw(img)
                        delta = (1 - slider_size) / float(slider_num - 1)
                        subrect = [0, 0, 0, 0]
                        subrect[0] = rect[0] + i * (rect[2] - rect[0]) * delta
                        subrect[1] = rect[1] + j * (rect[3] - rect[1]) * delta
                        subrect[2] = subrect[0] + (rect[2] - rect[0]) * slider_size
                        subrect[3] = subrect[1] + (rect[3] - rect[1]) * slider_size
                        d.rectangle([0, 0, img.size[0], subrect[1]], fill="black", outline=None)
                        d.rectangle([0, 0, subrect[0], img.size[1]], fill="black", outline=None)
                        d.rectangle([subrect[2], 0, img.size[0], img.size[1]], fill="black", outline=None)
                        d.rectangle([0, subrect[3], img.size[0], img.size[1]], fill="black", outline=None)
                        img = img.crop(rect)
                        datum_path = '{}_{}_{}_{}_{}_{}.jpeg'.format(path, type_str, rect_i, str(int(100 * slider_size)), str(i), str(j))
                        img.save(datum_path)
                        f.write('{} {}\n'.format(datum_path, str(class_id)))
            
        
synset_names = os.listdir(imagenet_root + 'image/')
             
for synset_index, synset_name in enumerate(synset_names):
    print "Processing synset [{}/{}]: {}".format(synset_index, len(synset_names), synset_name)
    image_names = os.listdir(imagenet_root + 'image/' + synset_name + '/' + synset_name + '_original_images')
    annotation_names = os.listdir(imagenet_root + 'Annotation/' + synset_name + '/')
    n1 = [os.path.splitext(n)[0] for n in image_names]
    n2 = [os.path.splitext(n)[0] for n in annotation_names]
    intersection_names = list(set(n1) & set(n2))
    # train : test = 300 : 100
    for i in range(400):
        print 'Processing image [{}/{}]: {}'.format(i, 400, intersection_names[i])
        # Read bounding box.
        bbx_file = open(imagenet_root + 'Annotation/' + synset_name + '/' + intersection_names[i] + '.xml')
        xmltree = ET.parse(bbx_file)
        objects = xmltree.findall('object')
        rects = []
        for obj in objects:
            bbx = obj.find('bndbox')
            rects.append([int(it.text) for it in bbx])
            
        img_orig = Image.open(imagenet_root + 'image/' + synset_name + '/' + synset_name + '_original_images/' + intersection_names[i] + '.JPEG')
        if i < 300: # Training dataset. 
            for index, (slider_size, slider_num) in enumerate(training_dataset):
                generate_datum(img_orig, '{}{}'.format(train_folders[index], intersection_names[i]), \
                               train_files[index], original_to_new_class_id[wnid_to_label[synset_name]], rects, slider_size, slider_num)
        else: # Testing dataset.
            for index, (slider_size, slider_num) in enumerate(test_dataset):
                generate_datum(img_orig, '{}{}'.format(test_folders[index], intersection_names[i]), \
                               test_files[index], original_to_new_class_id[wnid_to_label[synset_name]], rects, slider_size, slider_num)
            
for f in train_files:
    f.close()
for f in test_files:
    f.close()
            