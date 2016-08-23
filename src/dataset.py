# Generate training dataset and test dataset.
from constant import *
import os
import numpy as np
from PIL import Image, ImageDraw, ImageColor
import xml.etree.ElementTree as ET
import copy
import random
import time
import argparse
import math

#### Parameters. ####
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_index', required=True)
parser.add_argument('--type_str', required=True)
args = parser.parse_args()

#type_str = 'aperture' # {1k_}[crop | nocrop | crop_obj | nocrop_obj | aperture]
type_str = args.type_str

#dataset = [(0.0, 0), (1.0/4, 4), (1.0/3, 3), (1.0/2, 3), (2.0/3, 3), (4.0/5, 3), (9.0/10, 3), (1.0, 1)]
#dataset = [(0.0, 0), (0.1, 10), (0.2, 5), (0.3, 4), (0.4, 3), (0.5, 3), (0.6, 3), (0.7, 3), (0.8, 3), (0.9, 3), (1.0, 1)]
#dataset = [(0.0, 0), (0.2, 9), (0.4, 9), (0.6, 9), (0.8, 9), (1.0, 9)] # when crop_obj or nocrop_obj, (size, total_num)
dataset = [(0.0, 0),  (0.1, 10), (0.2, 10), (0.3, 10), (0.4, 10),\
           (0.5, 10), (0.6, 10), (0.7, 10), (0.8, 10), (0.9, 10), (1.0, 1)]
dataset_index = int(args.dataset_index)
#dataset_index = 5
print 'Processsing dataset {}, type_str {}'.format(dataset[dataset_index], type_str)
dataset = [dataset[dataset_index]]

# divide to training dataset and test dataset
training_dataset_size = 300
validation_dataset_size = 100
test_dataset_size = 100

# obj image: always 1024 * 768 pixels
obj_width = 1024
obj_height = 768
obj_ratio = float(obj_width) / obj_height
#####################

mean_color = (123, 117, 104)

# Load labels.
imagenet_labels_filename = imagenet_root + 'ilsvrc12/synset_words.txt'
label_to_wnid = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
wnid_to_label = {}
for label in range(len(label_to_wnid)):
    wnid = label_to_wnid[label].split(' ')[0]
    wnid_to_label[wnid] = label

train_folders = []
val_folders = []
test_folders = []

train_files = []
val_files = []
test_files = []

for (slider_size, slider_num) in dataset:
    percent = str(int(100 * slider_size))
    f = open('{}dataset/train_{}_{}.txt'.format(imagenet_root, type_str, percent), 'w')
    train_files.append(f)
    
    folder = '{}dataset/train_{}_{}/'.format(imagenet_root, type_str, percent)
    if not os.path.exists(folder):
        os.makedirs(folder)
    train_folders.append(folder)

for (slider_size, slider_num) in dataset:
    percent = str(int(100 * slider_size))
    f = open('{}dataset/val_{}_{}.txt'.format(imagenet_root, type_str, percent), 'w')
    val_files.append(f)

    folder = '{}dataset/val_{}_{}/'.format(imagenet_root, type_str, percent)
    if not os.path.exists(folder):
        os.makedirs(folder)
    val_folders.append(folder)

for (slider_size, slider_num) in dataset:
    percent = str(int(100 * slider_size))
    f = open('{}dataset/test_{}_{}.txt'.format(imagenet_root, type_str, percent), 'w')
    test_files.append(f)

    folder = '{}dataset/test_{}_{}/'.format(imagenet_root, type_str, percent)
    if not os.path.exists(folder):
        os.makedirs(folder)
    test_folders.append(folder)
    
if 'obj' in type_str:
    obj_images = []
    for i, image_name in enumerate(os.listdir(shapenet_root + 'object_nobg/')):
        img_temp = Image.open(shapenet_root + 'object_nobg/' + image_name)
        img = img_temp.copy()
        img_temp.close()
        #img = img.convert("RGBA")
        obj_images.append(img)
    print "{} object occluders loaded.".format(len(obj_images))

# occluder size = slider_size * slider_size
# occluder num = slider_num * slider_num
# path = 'imagenet_root/dataset/train_0/name'
# {wnid_imgid}_{crop/nocrop}_{rect_i(ifcrop)}_{slider_size}_{i}_{j}
def generate_datum(img_orig, path, f, class_id, rects, slider_size, slider_num):
    if type_str == 'crop' or type_str == '1k_crop':
        if slider_size == 0:
            for rect_i, rect in enumerate(rects):
                img = img_orig.copy()
                img = img.crop(rect)
                datum_path = '{}_{}_{}_0_0.jpeg'.format(path, type_str, rect_i)
                img.save(datum_path)
                f.write('{} {}\n'.format(datum_path, str(class_id)))
        else:
            for rect_i, rect in enumerate(rects):
                for i in range(slider_num):
                    img = img_orig.copy()
                    d = ImageDraw.Draw(img)
                    slider_width = int((rect[2] - rect[0]) * math.sqrt(slider_size))
                    slider_height = int((rect[3] - rect[1]) * math.sqrt(slider_size))
                    subrect = [0, 0, 0, 0]
                    subrect[0] = random.randint(rect[0], rect[2] - slider_width)
                    subrect[1] = random.randint(rect[1], rect[3] - slider_height)
                    subrect[2] = subrect[0] + slider_width
                    subrect[3] = subrect[1] + slider_height
                    d.rectangle(subrect, fill=mean_color, outline=None)
                    img = img.crop(rect)
                    datum_path = '{}_{}_{}_{}_{}.jpeg'.format(path, type_str, rect_i, str(int(100 * slider_size)), i)
                    img.save(datum_path)
                    f.write('{} {}\n'.format(datum_path, str(class_id)))
    if type_str == 'nocrop' or type_str == '1k_nocrop':
        if slider_size == 0:
            datum_path = '{}_{}_0_0_0.jpeg'.format(path, type_str)
            img_orig.save(datum_path)
            f.write('{} {}\n'.format(datum_path, str(class_id)))
        else:
            for i in range(slider_num):
                img = img_orig.copy()
                d = ImageDraw.Draw(img)
                for rect in rects:
                    slider_width = int((rect[2] - rect[0]) * math.sqrt(slider_size))
                    slider_height = int((rect[3] - rect[1]) * math.sqrt(slider_size))
                    subrect = [0, 0, 0, 0]
                    subrect[0] = random.randint(rect[0], rect[2] - slider_width)
                    subrect[1] = random.randint(rect[1], rect[3] - slider_height)
                    subrect[2] = subrect[0] + slider_width
                    subrect[3] = subrect[1] + slider_height
                    d.rectangle(subrect, fill=mean_color, outline=None)
                datum_path = '{}_{}_{}_{}_{}.jpeg'.format(path, type_str, 0, str(int(100 * slider_size)), i)
                img.save(datum_path)
                f.write('{} {}\n'.format(datum_path, str(class_id)))   
                
                
    if type_str == 'aperture' or type_str == '1k_aperture':
        if slider_size == 0: # All gray.
            for rect_i, rect in enumerate(rects):
                img = img_orig.copy()
                d = ImageDraw.Draw(img)
                d.rectangle(rect, fill=mean_color, outline=None)
                img = img.crop(rect)
                datum_path = '{}_{}_{}_{}_0.jpeg'.format(path, type_str, rect_i, str(int(100 * slider_size)))
                img.save(datum_path)
                f.write('{} {}\n'.format(datum_path, str(class_id)))
        elif slider_size == 1: # All visible. 
            for rect_i, rect in enumerate(rects):
                img = img_orig.copy()
                img = img.crop(rect)
                datum_path = '{}_{}_{}_{}_0.jpeg'.format(path, type_str, rect_i, str(int(100 * slider_size)))
                img.save(datum_path)
                f.write('{} {}\n'.format(datum_path, str(class_id)))
        else:
            for rect_i, rect in enumerate(rects):
                for i in range(slider_num):
                    img = img_orig.copy()
                    d = ImageDraw.Draw(img)
                    
                    slider_width = int((rect[2] - rect[0]) * math.sqrt(slider_size))
                    slider_height = int((rect[3] - rect[1]) * math.sqrt(slider_size))
                    
                    subrect = [0, 0, 0, 0]
                    subrect[0] = random.randint(rect[0], rect[2] - slider_width)
                    subrect[1] = random.randint(rect[1], rect[3] - slider_height)
                    subrect[2] = subrect[0] + slider_width
                    subrect[3] = subrect[1] + slider_height

                    d.rectangle([0, 0, img.size[0], subrect[1]], fill=mean_color, outline=None)
                    d.rectangle([0, 0, subrect[0], img.size[1]], fill=mean_color, outline=None)
                    d.rectangle([subrect[2], 0, img.size[0], img.size[1]], fill=mean_color, outline=None)
                    d.rectangle([0, subrect[3], img.size[0], img.size[1]], fill=mean_color, outline=None)
                    img = img.crop(rect)
                    datum_path = '{}_{}_{}_{}_{}.jpeg'.format(path, type_str, rect_i, str(int(100 * slider_size)), i)
                    img.save(datum_path)
                    f.write('{} {}\n'.format(datum_path, str(class_id)))
    
    if type_str == 'crop_obj' or type_str == '1k_crop_obj':
        if slider_size == 0:
            for rect_i, rect in enumerate(rects):
                img = img_orig.copy()
                img = img.crop(rect)
                datum_path = '{}_{}_{}_{}_0.jpeg'.format(path, type_str, rect_i, int(100 * slider_size))
                img.save(datum_path)
                f.write('{} {}\n'.format(datum_path, str(class_id)))
        else:
            for rect_i, rect in enumerate(rects):
                for num in range(slider_num):
                    img = img_orig.copy()
                    random_obj = obj_images[random.randint(0, len(obj_images) - 1)].copy()
                    width = rect[2] - rect[0]
                    height = rect[3] - rect[1]
                    ratio = float(width) / height
                    if ratio >= obj_ratio:
                        resize_scale = float(height) / obj_height * slider_size
                    else:
                        resize_scale = float(width) / obj_width * slider_size
                    new_width = int(obj_width * resize_scale)
                    new_height = int(obj_height * resize_scale)
                    if new_width == 0:
                        new_width = 1
                    if new_height == 0:
                        new_height = 1
                    random_obj = random_obj.resize((new_width, new_height), Image.ANTIALIAS)
                    
                    rangex = [rect[0], rect[2] - random_obj.size[0]]
                    if rangex[1] < rangex[0]:
                        rangex[1] = rangex[0]
                        
                    rangey = [rect[1], rect[3] - random_obj.size[1]]
                    if rangey[1] < rangey[0]:
                        rangey[1] = rangey[0]
                    top_left = (random.randint(rangex[0], rangex[1]), random.randint(rangey[0], rangey[1]))
                    img.paste(random_obj, top_left, random_obj)
                    img = img.crop(rect)
                    datum_path = '{}_{}_{}_{}_{}.jpeg'.format(path, type_str, rect_i, int(100 * slider_size), num)
                    img.save(datum_path)
                    f.write('{} {}\n'.format(datum_path, str(class_id)))
                    
    if type_str == 'nocrop_obj' or type_str == '1k_nocrop_obj':
        if slider_size == 0:
            datum_path = '{}_{}_{}_{}_0.jpeg'.format(path, type_str, 0, int(100 * slider_size))
            img_orig.save(datum_path)
            f.write('{} {}\n'.format(datum_path, str(class_id)))
        else:
            for num in range(slider_num):
                img = img_orig.copy()
                for rect_i, rect in enumerate(rects):
                    random_obj = obj_images[random.randint(0, len(obj_images) - 1)].copy()
                    width = rect[2] - rect[0]
                    height = rect[3] - rect[1]
                    ratio = float(width) / height
                    if ratio >= obj_ratio:
                        resize_scale = float(height) / obj_height * slider_size
                    else:
                        resize_scale = float(width) / obj_width * slider_size
                    new_width = int(obj_width * resize_scale)
                    new_height = int(obj_height * resize_scale)
                    if new_width == 0:
                        new_width = 1
                    if new_height == 0:
                        new_height = 1
                    random_obj = random_obj.resize((new_width, new_height), Image.ANTIALIAS)
                    
                    rangex = [rect[0], rect[2] - random_obj.size[0]]
                    if rangex[1] < rangex[0]:
                        rangex[1] = rangex[0]
                        
                    rangey = [rect[1], rect[3] - random_obj.size[1]]
                    if rangey[1] < rangey[0]:
                        rangey[1] = rangey[0]
                    top_left = (random.randint(rangex[0], rangex[1]), random.randint(rangey[0], rangey[1]))
                    img.paste(random_obj, top_left, random_obj)
                    
                datum_path = '{}_{}_{}_{}_{}.jpeg'.format(path, type_str, rect_i, int(100 * slider_size), num)
                img.save(datum_path)
                f.write('{} {}\n'.format(datum_path, str(class_id)))              
                    
                    
    
            
image_path = imagenet_root + 'ILSVRC2015/Data/CLS-LOC/train/'
annotation_path =  imagenet_root + 'ILSVRC2015/Annotations/CLS-LOC/train/'

if '1k' in type_str:
    synset_names = os.listdir(image_path)

start_time = time.time()

dataset_sum = training_dataset_size + validation_dataset_size + test_dataset_size
all_sum = len(synset_names) * dataset_sum
print all_sum

for synset_index, synset_name in enumerate(synset_names):
    image_names = os.listdir(image_path + synset_name)
    annotation_names = os.listdir(annotation_path + synset_name)
    n1 = [os.path.splitext(n)[0] for n in image_names]
    n2 = [os.path.splitext(n)[0] for n in annotation_names]
    intersection_names = list(set(n1) & set(n2))
    for i in range(dataset_sum):
        if (i + 1) % 50 == 0:
            second = int(time.time() - start_time)
            now_time = time.strftime("%H:%M:%S", time.gmtime(second))
            now_sum = synset_index * dataset_sum + i
            
            estimated = int(float(all_sum) / now_sum * second)
            estimated_time = time.strftime("%H:%M:%S", time.gmtime(estimated))
            estimated_day = estimated / 3600 / 24
            print '[{}/{} {}]Processing synset [{}/{}], image [{}/{}]: {}'.format(now_time, estimated_day, estimated_time, synset_index + 1, len(synset_names), i + 1, dataset_sum, intersection_names[i])
        # Read bounding box.
        bbx_file = open(annotation_path + synset_name + '/' + intersection_names[i] + '.xml')
        xmltree = ET.parse(bbx_file)
        objects = xmltree.findall('object')
        rects = []
        for obj in objects:
            bbx = obj.find('bndbox')
            rects.append([int(it.text) for it in bbx])
            
        img_orig = Image.open(image_path + synset_name + '/' + intersection_names[i] + '.JPEG')
        if img_orig.mode != "RGB":
            img_orig = img_orig.convert("RGB")
            
        if '1k' in type_str:
            class_id = wnid_to_label[synset_name]
        else:
            class_id = original_to_new_class_id[wnid_to_label[synset_name]]
        
        if i < training_dataset_size: # Training dataset. 
            for index, (slider_size, slider_num) in enumerate(dataset):
                generate_datum(img_orig, '{}{}'.format(train_folders[index], intersection_names[i]), \
                               train_files[index], class_id, rects, slider_size, slider_num)
        elif i < training_dataset_size + validation_dataset_size: # Validation dataset
            for index, (slider_size, slider_num) in enumerate(dataset):
                generate_datum(img_orig, '{}{}'.format(val_folders[index], intersection_names[i]), \
                               val_files[index], class_id, rects, slider_size, slider_num)
        else: # Test dataset.
            for index, (slider_size, slider_num) in enumerate(dataset):
                generate_datum(img_orig, '{}{}'.format(test_folders[index], intersection_names[i]), \
                               test_files[index], class_id, rects, slider_size, slider_num)
                               
for f in train_files:
    f.close()
for f in test_files:
    f.close()
            