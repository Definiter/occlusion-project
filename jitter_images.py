# Steps:
# 1. Specify image input and output paths
# 2. Specify which neurons activation considered useful for jittering. Specify thresholds if necessary
#           Two options: either rely on clusters, or rely on neurons
# 3. Merge regions, ignore boundaries, to form a region for jittering
# 4. Randomly move regions based on given radius, while avoiding collision
# 5. Smooth out boundaries

from constants import *
import argparse
from os import walk
import os, sys, math, pickle, matplotlib
import numpy as np
import get_receptive_field as rf
import random
from Queue import Queue
from PIL import Image
import fnmatch
import time

# Set Caffe output level to Warnings
os.environ['GLOG_minloglevel'] = '2'

sys.path.insert(0, caffe_root + 'python')
import caffe
if mode == 'cpu':
    caffe.set_mode_cpu()
else:
    caffe.set_mode_gpu()

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir', required=True)
parser.add_argument('-o', '--output_dir', required=True)
parser.add_argument('-ld', '--layer_dump', required=True)
parser.add_argument('-cd', '--clusters_dump', required=True)
parser.add_argument('-c', '--clusters', nargs='+', required=True)
parser.add_argument('-r', '--radius', required=True, help='Jitter radius')
parser.add_argument('-ct', '--cluster_threshold', required=False, default=0.6, help='Cluster distance threshold (0-1)')
parser.add_argument('-ot', '--overlap_threshold', required=False, default=0.6, help='Overlap threshold for equality of boxes')
parser.add_argument('--interactive', action='store_true', default=False, required=False,
    help='Show which parts are jittered in a screen instead of saving')
parser.add_argument('-bb', '--show_bounding_box', action='store_true', default=False, required=False,
                    help='Show or save detected bounding boxes')
parser.add_argument('--filename_filter', required=False, default=None, help='Controls which files are considered')
parser.add_argument('--gpu', default=0, required=False)
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if not args.interactive:
    matplotlib.use('Agg')  # For saving images to file
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

caffe.set_device(int(args.gpu))

net = caffe.Classifier(caffe_root + 'models/vgg16/VGG_ILSVRC_16_layers_deploy.prototxt',
        caffe_root + 'models/vgg16/VGG_ILSVRC_16_layers.caffemodel',
        mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
        channel_swap=(2, 1, 0),
        raw_scale=255,
        image_dims=(224, 224))
net.blobs['data'].reshape(1, 3, 224, 224) #??? necessary?

print 'Loading layer dump file from', args.layer_dump
f = open(args.layer_dump)
[_, layer, _, _, _, _, vectors, vec_origin_file, vec_location] = pickle.load(f)
f.close()
print 'Finished.'

print 'Loading clusters dump from', args.clusters_dump
f = open(args.clusters_dump)
dumped = pickle.load(f)
kmeans_scores = []
if len(dumped) == 3:
    n_clusters, kmeans_obj, predicted = dumped
elif len(dumped) == 4:
    n_clusters, kmeans_obj, predicted, kmeans_scores = dumped
f.close()
print 'Finished.'

n_vectors = len(vectors)

if len(kmeans_scores) == 0:
    print 'Calculating kmeans scores...\n'
    for vec_i in range(n_vectors):
        kmeans_scores.append(kmeans_obj.score(vectors[vec_i].reshape(1, -1)))
        sys.stdout.write("\rFinished: %f%%" % (vec_i * 100.0 / n_vectors))
        sys.stdout.flush()
    print '\nDone.'


def load_image(path):
    net.predict([caffe.io.load_image(path)], oversample=False)


class UnionFind:

    obj_arr = []
    group_arr = []
    # A passed-in function which decides if two objects should be merged
    # Returns 0 if yes, 1 otherwise
    compare_fun = None

    def union(self, i, j):
        i_root = self.find_root(i)
        j_root = self.find_root(j)

        if i_root == j_root:
            return
        self.group_arr[j] = i_root

    def find_root(self, i):
        if self.group_arr[i] == i:
            return i
        return self.find_root(self.group_arr[i])

    def merge_all(self):
        for i in range(len(self.obj_arr)):
            for j in range(i + 1, len(self.obj_arr)):
                if self.compare_fun(self.obj_arr[i], self.obj_arr[j]) == 0:
                    self.union(i, j)

    def __init__(self, data_arr, compare_fun):
        self.obj_arr = data_arr
        # Initialize group IDs
        self.group_arr = range(len(data_arr))
        self.compare_fun = compare_fun


# Returns 0 if the squares have significant overlaps, 1 otherwise
def are_squares_overlap(tup1, tup2):
    overlap_threshold = float(args.overlap_threshold)

    # Can't be any overlap at all
    left = tup2
    right = tup1
    top = tup2
    bottom = tup1
    if tup1[0] < tup2[0]:
        left = tup1
        right = tup2
    if tup1[1] < tup2[1]:
        top = tup1
        bottom = tup2
    if left[2] < right[0] or top[3] < bottom[1]:
        return 1

    p1_x = max(tup1[0], tup2[0])
    p1_y = max(tup1[1], tup2[1])
    p2_x = min(tup1[2], tup2[2])
    p2_y = min(tup1[3], tup2[3])

    area_overlap = (p2_x - p1_x) * (p2_y - p1_y) * 1.0
    area1 = (tup1[2] - tup1[0]) * (tup1[3] - tup1[1]) * 1.0
    area2 = (tup2[2] - tup2[0]) * (tup2[3] - tup2[1]) * 1.0

    if area_overlap / area1 > overlap_threshold or area_overlap / area2 > overlap_threshold:
        return 0
    return 1


# Given an array of squares (four-tuples, of coordinates of topleft and bottomright pixels),
# Merge those with large overlapping areas into larger shapes
def merge_squares(squares):
    uf = UnionFind(squares, are_squares_overlap)
    uf.merge_all()
    merged_squares = {}

    # Taking the simple approach: just generate a rectangle that contains all those squares
    for i in range(len(squares)):
        square = squares[i]
        group = uf.group_arr[i]

        if group not in merged_squares:
            merged_squares[group] = [float('inf'), float('inf'), -1, -1]

        merged_squares[group][0] = min(merged_squares[group][0], square[0])
        merged_squares[group][1] = min(merged_squares[group][1], square[1])
        merged_squares[group][2] = max(merged_squares[group][2], square[2])
        merged_squares[group][3] = max(merged_squares[group][3], square[3])

    return merged_squares.values()


# Merge squares through nonmaximal suppression: finding all local maxima, use squares surrounding those as
# merged squares. Once one square is used, don't use it again
def merge_squares_2(squares, im):
    im_height = len(im)
    im_width = len(im[0])
    counter = np.zeros((im_height, im_width))
    # For every square, increment counters in the corresponding region
    for square in squares:
        x1, y1, x2, y2 = square
        counter[y1:(y2+1), x1:(x2+1)] += np.ones((y2-y1+1, x2-x1+1))

    is_in_bound = lambda x, y: not (x < 0 or y < 0 or x >= im_width or y >= im_height)

    could_be = np.zeros((im_height, im_width))
    for y in range(im_height):
        for x in range(im_width):
            cur = counter[y, x]
            if (y == 0 or counter[y-1, x] <= cur) and (y == im_height - 1 or counter[y+1, x] <= cur) and \
               (x == 0 or counter[y, x-1] <= cur) and (x == im_width - 1 or counter[y, x+1] <= cur):
                #if y != 0 and x != 0 and y < im_height and x < im_width:
                #    print could_be[(y-1):(y+2), (x-1):(x+2)]
                could_be[y, x] = 1

    # "Corrupt" those "could-be"'s from those "cannot-be"'s
    frontiers = Queue()
    # Put all 0 in frontiers
    for y in range(im_height):
        for x in range(im_width):
            if could_be[y, x] == 0:
                frontiers.put((x, y))
    while not frontiers.empty():
        x, y = frontiers.get(False)
        for delta_x in range(-1, 2):
            for delta_y in range(-1, 2):
                if abs(delta_x + delta_y) != 1:
                    continue
                new_x = x + delta_x
                new_y = y + delta_y
                # Check boundaries
                if not is_in_bound(new_x, new_y) or could_be[new_y, new_x] == 0:
                    continue
                if counter[y, x] == counter[new_y, new_x]:
                    could_be[new_y, new_x] = 0
                    frontiers.put((new_x, new_y))

    # All cells with 1 in could_be are local maxima. Use these to find regions to merge
    # Find all contiguous regions. In each region, find the topleft corner and bottom right corner
    merged_boxes = []
    for y in range(im_height):
        for x in range(im_width):
            if could_be[y, x] == 0:
                continue
            frontiers = Queue()
            frontiers.put((x, y))
            box = [im_width, im_height, -1, -1]

            while not frontiers.empty():
                r_x, r_y = frontiers.get(False)
                could_be[r_y, r_x] = 0
                for delta_x in range(-1, 2):
                    for delta_y in range(-1, 2):
                        if abs(delta_x + delta_y) != 1:
                            continue
                        neighb_x = r_x + delta_x
                        neighb_y = r_y + delta_y
                        if not is_in_bound(neighb_x, neighb_y) or could_be[neighb_y, neighb_x] == 0:
                            continue
                        could_be[neighb_y, neighb_x] = 0
                        frontiers.put((neighb_x, neighb_y))
                        box[0] = min(box[0], neighb_x)
                        box[1] = min(box[1], neighb_y)
                        box[2] = max(box[2], neighb_x)
                        box[3] = max(box[3], neighb_y)

            merged_box = list(box)
            # With these coordinates, find boxes that fully contain this region
            for square in list(squares):  # Iterate in the copy of squares
                if square[0] <= box[0] and square[1] <= box[1] and square[2] >= box[2] and square[3] >= box[3]:
                    merged_box[0] = min(merged_box[0], square[0])
                    merged_box[1] = min(merged_box[1], square[1])
                    merged_box[2] = max(merged_box[2], square[2])
                    merged_box[3] = max(merged_box[3], square[3])
                    squares.remove(square)

            # TODO Maybe take the halfway between box and merged box is better?
            merged_boxes.append(merged_box)

    return merged_boxes


def jitter_regions(im, regions, radius):
    gradient_percentage = 0.5
    jittered = im.copy()
    for region in regions:
        patch = im[region[1]:(region[3]+1), region[0]:(region[2]+1), :]
        region_height = int(region[3] - region[1])
        region_width = int(region[2] - region[0])

        jitter_y = int(random.randrange(-radius, radius))
        jitter_x = int(math.floor(math.sqrt(radius ** 2 - jitter_y ** 2) * np.random.choice([-1, 1])))

        new_y1 = region[1] + jitter_y
        # Check boundaries
        if new_y1 < 0:
            new_y1 = 0
        if region_height + new_y1 >= len(im):
            new_y1 = len(im) - region_height - 1
        new_y2 = region_height + new_y1

        new_x1 = region[0] + jitter_x
        if new_x1 < 0:
            new_x1 = 0
        if region_width + new_x1 >= len(im[0]):
            new_x1 = len(im[0]) - region_width - 1
        new_x2 = region_width + new_x1

        assert(new_x1 >= 0 and new_x2 >= 0 and new_y1 >= 0 and new_y2 >= 0)
        assert(new_x1 < len(im[0]) and new_x2 < len(im[0]) and new_y1 < len(im) and new_y2 < len(im))
        assert(new_x2 - new_x1 == region[2] - region[0])
        assert(new_y2 - new_y1 == region[3] - region[1])

        center_x = region_width / 2 + new_x1
        center_y = region_height / 2 + new_y1
        farthest = math.sqrt((center_x - new_x1) ** 2 + (center_y - new_y1) ** 2)
        for y in range(new_y1, new_y2+1):
            for x in range(new_x1, new_x2+1):
                # Distance from center of patch
                dist = math.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
                patch_opacity = (1 - dist / farthest) / gradient_percentage # TODO CHANGE THE OPACITY FUNCTION for a smoother and faster
                if patch_opacity > 1:
                    patch_opacity = 1
                jittered[y, x, :] = (1 - patch_opacity) * im[y, x, :] + patch_opacity * patch[y-new_y1, x-new_x1, :]

    return jittered

clusters = []
for c in args.clusters:
    clusters.append(int(c))

# Maps cluster ID to kmeans score threshold. Patches with activation larger than threshold
# (meaning distance too large) will not be considered
thresholds = {}
thres_percentage = float(args.cluster_threshold)
for c in clusters:
    cluster_scores = [kmeans_scores[i] for i in range(n_vectors) if predicted[i] == c]
    cluster_scores.sort(reverse=True)
    thresholds[c] = cluster_scores[int(math.floor(len(cluster_scores) * thres_percentage))]
    print 'Range of all scores for cluster', c, ':', min(cluster_scores), ' - ', max(cluster_scores)


def jitter_images():
    for (dirpath, dirnames, filenames) in walk(args.input_dir):
        for filename in filenames:
            if args.filename_filter is not None:
                if not fnmatch.fnmatch(filename, args.filename_filter):
                    continue
            path = os.path.abspath(os.path.join(dirpath, filename))
            sys.stdout.write("\x1b[2K\rProcessed: %s" % path)
            sys.stdout.flush()

            name_only, ext = os.path.splitext(filename)

            load_image(path)
            dim_feature_map = len(net.blobs[layer].data[0][0])
            num_feature_maps = len(net.blobs[layer].data[0])
            im = net.transformer.deprocess('data', net.blobs['data'].data[0])
            axis = None
            if args.interactive:
                plt.imshow(im)
                axis = plt.gca()

            detected_squares = {}

            # NOTE that in a 256x56x56 layer,
            #   dat.reshape(-1,56*56)[:,0] == dat[:,0,0]
            #   dat.reshape(-1,56*56)[:,56] == dat[:,1,0]
            hypercolumns = net.blobs[layer].data[0].reshape(num_feature_maps, -1)
            # Vectorize. 20 TIMES FASTER THAN NONVECTORIZED VERSION!!!
            transposed = hypercolumns.transpose((1, 0))
            predictions = kmeans_obj.predict(transposed)
            # Vectorize: use transformed instead of .score() on each vector, 8x speedup
            transformed = kmeans_obj.transform(transposed)
            for y in range(dim_feature_map):
                for x in range(dim_feature_map):
                    i = y * dim_feature_map + x
                    prediction = predictions[i]
                    score = - transformed[i, prediction] ** 2
                    if prediction in clusters and score > thresholds[prediction]:
                        rec_field = rf.get_receptive_field(layer, x, y)

                        # HACK: if the borders touch any boundary of the image, don't use it
                        if rec_field[0] == 0 or rec_field[1] == 0 or rec_field[2] == len(im[0]) - 1 or rec_field[3] == len(im) - 1:
                            continue

                        if args.show_bounding_box:
                            axis.add_patch(Rectangle((rec_field[0], rec_field[1]),
                                rec_field[2] - rec_field[0] + 1,
                                rec_field[3] - rec_field[1] + 1,
                                fill=False, edgecolor='red'))
                        if prediction not in detected_squares:
                            detected_squares[prediction] = []
                        detected_squares[prediction].append(rec_field)

            all_squares = []
            for k in detected_squares.keys():
                all_squares += detected_squares[k]
            merged_regions = merge_squares(all_squares)

            if args.show_bounding_box:
                for region in merged_regions:
                    axis.add_patch(Rectangle((region[0], region[1]),
                        region[2] - region[0] + 1,
                        region[3] - region[1] + 1,
                        fill=False, edgecolor="blue"))

            if args.interactive and args.show_bounding_box:
                plt.show()
                plt.clf()
            elif args.show_bounding_box:
                plt.savefig(os.path.join(args.output_dir, name_only + '_detected' + ext), bbox_inches='tight', pad_inches=0)
                plt.clf()

            jittered = jitter_regions(im, merged_regions, int(args.radius))
            if args.interactive:
                plt.imshow(jittered)
                plt.show()
            else:
                save_path = os.path.join(args.output_dir, name_only + '_jittered' + ext)
                rescaled = (255.0 / jittered.max() * (jittered - jittered.min())).astype(np.uint8)
                jittered_im = Image.fromarray(rescaled)
                jittered_im.save(save_path)

start = time.time()
jitter_images()
end = time.time()
print '\nTook', end - start, 'seconds'
