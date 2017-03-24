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
import act_max as act
import scipy.misc, scipy.io
#import caffe

#### Parameters. ####
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_index', required=True)
parser.add_argument('--type_str', required=True)
#parser.add_argument('--gpu', required=False)
args = parser.parse_args()

#type_str = 'crop_image' # {1k_}[crop | nocrop | crop_obj | nocrop_obj | aperture | crop_img | crop_image ]
type_str = args.type_str

#dataset = [(0.0, 0),  (0.1, 10), (0.2, 10), (0.3, 10), (0.4, 10),\
#           (0.5, 10), (0.6, 10), (0.7, 10), (0.8, 10), (0.9, 10), (1.0, 1)]
dataset = [(0.0, 0),  (0.1, 1), (0.2, 1), (0.3, 1), (0.4, 1),\
           (0.5, 1), (0.6, 1), (0.7, 1), (0.8, 1), (0.9, 1), (1.0, 1)]
dataset_index = int(args.dataset_index)
#dataset_index = 5
print 'Processsing dataset {}, type_str {}'.format(dataset[dataset_index], type_str)
dataset = [dataset[dataset_index]]

'''
if args.gpu != None:
    gpu = int(args.gpu)
    caffe.set_device(gpu)
    caffe.set_mode_gpu()
'''
#caffe.set_mode_gpu()

# divide to training dataset and test dataset
training_dataset_size = 300
validation_dataset_size = 100
test_dataset_size = 100

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
    
# Initialization for 'obj'.
if 'obj' in type_str:
    obj_images = []
    for i, image_name in enumerate(os.listdir(shapenet_root + 'object_crop/')):
        img_temp = Image.open(shapenet_root + 'object_crop/' + image_name)
        img = img_temp.copy()
        img_temp.close()
        #img = img.convert("RGBA")
        obj_images.append(img)
    print "{} object occluders loaded.".format(len(obj_images))
    
'''
# On-the-fly generation of "crop_img".
# Initialization for 'imagination'.
if type_str == 'crop_img':
    act_args = lambda: None
    act_args.xy = 0
    act_args.n_iters = 200
    act_args.L2 = 0.99
    act_args.start_lr = 8.0
    act_args.end_lr = 1e-10
    act_args.seed = 0
    act_args.opt_layer = "fc6"
    act_args.act_layer = "fc8_occlusion"
    act_args.init_file = "None"
    act_args.clip = 0
    act_args.bound = synthesizing_root + "act_range/3x/fc6.txt"
    act_args.debug = 0
    act_args.output_dir = synthesizing_root + "output/baseline"

    act_args.net_weights = result_root + "model/finetune_alexnet_crop_0/finetune_alexnet_crop_0.caffemodel"
    act_args.net_definition = result_root + "model/finetune_alexnet_crop_0/deploy.prototxt"
    act_args.generator_weights = synthesizing_root + "nets/upconv/fc6/generator.caffemodel"
    act_args.generator_definition = synthesizing_root + "nets/upconv/fc6/generator.prototxt"
    act_args.encoder_weights = synthesizing_root + "nets/caffenet/bvlc_reference_caffenet.caffemodel"
    act_args.encoder_definition = synthesizing_root + "nets/caffenet/caffenet.prototxt"

    params = [
        {
            'layer': act_args.act_layer,
            'iter_n': act_args.n_iters,
            'L2': act_args.L2,
            'start_step_size': act_args.start_lr,
            'end_step_size': act_args.end_lr
        }
    ]

    # Networks.
    generator = caffe.Net(act_args.generator_definition, act_args.generator_weights, caffe.TEST)
    net = caffe.Classifier(act_args.net_definition, act_args.net_weights,
            mean = np.float32([104.0, 117.0, 123.0]), # ImageNet mean
            channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

    # input / output layers in generator
    gen_in_layer = "feat"
    gen_out_layer = "deconv0"

    # shape of the code being optimized
    shape = generator.blobs[gen_in_layer].data.shape

    # Fix the seed
    np.random.seed(act_args.seed)
'''

# Initialization for 'crop_img'.
if type_str == 'crop_img':
    img_images = []
    for class_id in range(100):
        for num in range(20):
            filename = '{}_{}.jpg'.format(class_id, num)
            img_temp = Image.open(imagenet_root + 'synthesized/' + filename)
            img = img_temp.copy()
            img_temp.close()
            img_images.append(img)
    print "{} synthesized images loaded.".format(len(img_images))
    


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
                    obj_width = random_obj.size[0]
                    obj_height = random_obj.size[1]
                    obj_ratio = float(obj_width) / obj_height
                    
                    width = rect[2] - rect[0]
                    height = rect[3] - rect[1]
                    ratio = float(width) / height
                    
                    max_occlusion = 0.0
                    if ratio >= obj_ratio:
                        max_occlusion = obj_width * height / float(width * obj_height)
                    else:
                        max_occlusion = width * obj_height / float(height * obj_width)
                    
                    if max_occlusion >= slider_size:
                        # Do not need to stretch and change obj_ratio.
                        new_width = (slider_size * width * height * obj_width / float(obj_height)) ** 0.5
                        new_height = (slider_size * width * height * obj_height / float(obj_width)) ** 0.5
                    else:
                        # Need to stretch and change obj_ratio.
                        if ratio >= obj_ratio:
                            new_width = width * slider_size
                            new_height = height
                        else:
                            new_width = width
                            new_height = height * slider_size
                    new_width = int(new_width)
                    new_height = int(new_height)
                    if new_width == 0:
                        new_width = 1
                    if new_height == 0:
                        new_height = 1
                        
                    random_obj = random_obj.resize((new_width, new_height), Image.ANTIALIAS)
                    
                    top_left = (random.randint(rect[0], rect[2] - random_obj.size[0]),\
                                random.randint(rect[1], rect[3] - random_obj.size[1]))
                    
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
                    
                    obj_width = random_obj.size[0]
                    obj_height = random_obj.size[1]
                    obj_ratio = float(obj_width) / obj_height
                    
                    width = rect[2] - rect[0]
                    height = rect[3] - rect[1]
                    ratio = float(width) / height
                    
                    max_occlusion = 0.0
                    if ratio >= obj_ratio:
                        max_occlusion = obj_width * height / float(width * obj_height)
                    else:
                        max_occlusion = width * obj_height / float(height * obj_width)
                    
                    if max_occlusion >= slider_size:
                        # Do not need to stretch and change obj_ratio.
                        new_width = (slider_size * width * height * obj_width / float(obj_height)) ** 0.5
                        new_height = (slider_size * width * height * obj_height / float(obj_width)) ** 0.5
                    else:
                        # Need to stretch and change obj_ratio.
                        if ratio >= obj_ratio:
                            new_width = width * slider_size
                            new_height = height
                        else:
                            new_width = width
                            new_height = height * slider_size
                    new_width = int(new_width)
                    new_height = int(new_height)
                    if new_width == 0:
                        new_width = 1
                    if new_height == 0:
                        new_height = 1
                        
                    random_obj = random_obj.resize((new_width, new_height), Image.ANTIALIAS)
                    
                    top_left = (random.randint(rect[0], rect[2] - random_obj.size[0]),\
                                random.randint(rect[1], rect[3] - random_obj.size[1]))
                    
                    img.paste(random_obj, top_left, random_obj)
                    
                datum_path = '{}_{}_{}_{}_{}.jpeg'.format(path, type_str, rect_i, int(100 * slider_size), num)
                img.save(datum_path)
                f.write('{} {}\n'.format(datum_path, str(class_id))) 
    
    if type_str == "crop_img":
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
                    img_image = img_images[random.randint(0, 2000 - 1)]
                    img.paste(img_image.resize((slider_width, slider_height)), (subrect[0], subrect[1]))
                    img = img.crop(rect)
                    datum_path = '{}_{}_{}_{}_{}.jpeg'.format(path, type_str, rect_i, str(int(100 * slider_size)), i)
                    img.save(datum_path)
                    f.write('{} {}\n'.format(datum_path, str(class_id)))
                    
    if type_str == "crop_image":
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
                    
                    # Get another image.
                    another_index = random.randint(0, len(synset_names) - 1)
                    another_synset_name = synset_names[another_index]
                    
                    # Read bounding box.
                    bbx_file = open(annotation_path + another_synset_name + '/' + intersection_names[another_index][i] + '.xml')
                    xmltree = ET.parse(bbx_file)
                    objects = xmltree.findall('object')
                    another_rects = []
                    for obj in objects:
                        bbx = obj.find('bndbox')
                        another_rects.append([int(it.text) for it in bbx])
                    another_rect = another_rects[random.randint(0, len(another_rects) - 1)]

                    another_image = Image.open(image_path + another_synset_name + '/' + intersection_names[another_index][i] + '.JPEG')
                    if another_image.mode != "RGB":
                        another_image = another_image.convert("RGB")
                    
                    another_obj = another_image.crop(another_rect)
                    
                    
                    
                    img.paste(another_obj.resize((slider_width, slider_height)), (subrect[0], subrect[1]))
                    img = img.crop(rect)
                    datum_path = '{}_{}_{}_{}_{}.jpeg'.format(path, type_str, rect_i, str(int(100 * slider_size)), i)
                    img.save(datum_path)
                    f.write('{} {}\n'.format(datum_path, str(class_id)))
            
    
    '''
    # On-the-fly generation.
    if type_str == "crop_img":
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
                    act_args.unit = random.randint(0, 99)
                    img = img_orig.copy()
                    d = ImageDraw.Draw(img)
                    slider_width = int((rect[2] - rect[0]) * math.sqrt(slider_size))
                    slider_height = int((rect[3] - rect[1]) * math.sqrt(slider_size))
                    subrect = [0, 0, 0, 0]
                    subrect[0] = random.randint(rect[0], rect[2] - slider_width)
                    subrect[1] = random.randint(rect[1], rect[3] - slider_height)
                    subrect[2] = subrect[0] + slider_width
                    subrect[3] = subrect[1] + slider_height
                    
                    act_args.init_file = '{}_{}_{}_{}_{}_init.jpeg'.format(path, type_str, rect_i, str(int(100 * slider_size)), i)
                    img.crop(subrect).save(act_args.init_file)
                    start_code, start_image = act.get_code(act_args.init_file, act_args.opt_layer)
                    os.remove(act_args.init_file)
                    
                    #start_code, start_image = act.get_code2(np.array(img.crop(subrect)), act_args.opt_layer)
                    
                    # Load the activation range
                    upper_bound = lower_bound = None
                    # Set up clipping bounds
                    if act_args.bound != "":
                        n_units = shape[1]
                        upper_bound = np.loadtxt(act_args.bound, delimiter=' ', usecols=np.arange(0, n_units), unpack=True)
                        upper_bound = upper_bound.reshape(start_code.shape)
                        # Lower bound of 0 due to ReLU
                        lower_bound = np.zeros(start_code.shape)
                    # Optimize a code via gradient ascent
                    output_image = act.activation_maximization(net, generator, gen_in_layer, gen_out_layer, start_code, params, 
                            clip=act_args.clip, unit=act_args.unit, xy=act_args.xy, debug=act_args.debug,
                            upper_bound=upper_bound, lower_bound=lower_bound)
                    output_image = output_image[:,::-1, :, :] # Convert from BGR to RGB
                    normalized_img = act.patchShow.patchShow_single(output_image, in_range=(-120,120))        
                    normalized_img = np.uint8(normalized_img * 255)
                    new_img = Image.fromarray(normalized_img)
                    #img_file = '{}_{}_{}_{}_{}_img.jpeg'.format(path, type_str, rect_i, str(int(100 * slider_size)), i)
                    #new_img.save(img_file)
                    
                    img.paste(new_img.resize((slider_width, slider_height)), (subrect[0], subrect[1]))
                    img = img.crop(rect)
                    
                    datum_path = '{}_{}_{}_{}_{}.jpeg'.format(path, type_str, rect_i, str(int(100 * slider_size)), i)
                    img.save(datum_path)
                    f.write('{} {}\n'.format(datum_path, str(class_id)))
    '''
                    
            
image_path = imagenet_root + 'ILSVRC2015/Data/CLS-LOC/train/'
annotation_path =  imagenet_root + 'ILSVRC2015/Annotations/CLS-LOC/train/'

if '1k' in type_str:
    synset_names = os.listdir(image_path)

start_time = time.time()

dataset_sum = training_dataset_size + validation_dataset_size + test_dataset_size
all_sum = len(synset_names) * dataset_sum
print all_sum

intersection_names = {}

for synset_index, synset_name in enumerate(synset_names):
    image_names = os.listdir(image_path + synset_name)
    annotation_names = os.listdir(annotation_path + synset_name)
    n1 = [os.path.splitext(n)[0] for n in image_names]
    n2 = [os.path.splitext(n)[0] for n in annotation_names]
    intersection_names[synset_index] = list(set(n1) & set(n2))

for synset_index, synset_name in enumerate(synset_names):
    for i in range(dataset_sum):
        if (i + 1) % 100 == 0:
            second = int(time.time() - start_time)
            now_time = time.strftime("%H:%M:%S", time.gmtime(second))
            now_sum = synset_index * dataset_sum + i
            
            estimated = int(float(all_sum) / now_sum * second)
            estimated_time = time.strftime("%H:%M:%S", time.gmtime(estimated))
            estimated_day = estimated / 3600 / 24
            print '[{}/{} {}]Processing synset [{}/{}], image [{}/{}]: {}'.format(now_time, estimated_day, estimated_time, synset_index + 1, len(synset_names), i + 1, dataset_sum, intersection_names[synset_index][i])
        # Read bounding box.
        bbx_file = open(annotation_path + synset_name + '/' + intersection_names[synset_index][i] + '.xml')
        xmltree = ET.parse(bbx_file)
        objects = xmltree.findall('object')
        rects = []
        for obj in objects:
            bbx = obj.find('bndbox')
            rects.append([int(it.text) for it in bbx])
            
        img_orig = Image.open(image_path + synset_name + '/' + intersection_names[synset_index][i] + '.JPEG')
        if img_orig.mode != "RGB":
            img_orig = img_orig.convert("RGB")
            
        if '1k' in type_str:
            class_id = wnid_to_label[synset_name]
        else:
            class_id = original_to_new_class_id[wnid_to_label[synset_name]]
        
        if i < training_dataset_size: # Training dataset. 
            for index, (slider_size, slider_num) in enumerate(dataset):
                generate_datum(img_orig, '{}{}'.format(train_folders[index], intersection_names[synset_index][i]), \
                               train_files[index], class_id, rects, slider_size, slider_num)
        elif i < training_dataset_size + validation_dataset_size: # Validation dataset
            for index, (slider_size, slider_num) in enumerate(dataset):
                generate_datum(img_orig, '{}{}'.format(val_folders[index], intersection_names[synset_index][i]), \
                               val_files[index], class_id, rects, slider_size, slider_num)
        else: # Test dataset.
            for index, (slider_size, slider_num) in enumerate(dataset):
                generate_datum(img_orig, '{}{}'.format(test_folders[index], intersection_names[synset_index][i]), \
                               test_files[index], class_id, rects, slider_size, slider_num)

for f in train_files:
    f.close()
for f in test_files:
    f.close()
            