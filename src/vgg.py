'''
Main file.
'''

from constants import *
import argparse
from os import walk
import os

import matplotlib

parser = argparse.ArgumentParser()
parser.add_argument('--images', default=research_root + 'images/flickr/eyes-yes/', required=False)
parser.add_argument('--layer', default='conv4_1', required=False)
parser.add_argument('--sample_fraction', default=0.3, required=False)
parser.add_argument('--n_clusters', default=32, required=False)

parser.add_argument('--center_only_path', default=None, required=False)
parser.add_argument('--center_only_neuron_x', default=None, required=False)

parser.add_argument('--gpu', default=0, required=False)

parser.add_argument('--load_layer_dump_from', default=None, required=False)
parser.add_argument('--load_classification_dump_from', default=None, required=False)

parser.add_argument('--save_layer_dump_to', default=None, required=False)
parser.add_argument('--save_classification_dump_to', default=None, required=False)

parser.add_argument('--save_plots_to', default=None, required=False)

parser.add_argument('--append_kmeans_scores', action='store_true')

args = parser.parse_args()

if args.save_plots_to is not None:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

import math
import numpy as np
import pickle

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE
import sklearn.datasets

import skimage

import get_receptive_field as rf
import caffe

# Find better way to write it to distribute more evenly
def sample(width, height, number):
    prob_true = number * 1.0 / width / height
    return np.random.rand(height, width) < prob_true

# Force non-interative mode, if saving plots
if args.save_plots_to is not None:
    plt.ioff()

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

print 'Loading images from ' + args.images
print 'Sampling ' + str(args.sample_fraction) + ' of responses from layer ' + args.layer

imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

caffe.set_device(int(args.gpu))
if mode == 'cpu':
    caffe.set_mode_cpu()
else:
    caffe.set_mode_gpu()

net = caffe.Classifier(caffe_root + 'models/vgg16/VGG_ILSVRC_16_layers_deploy.prototxt',
        caffe_root + 'models/vgg16/VGG_ILSVRC_16_layers.caffemodel',
        mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
        channel_swap=(2, 1, 0),
        raw_scale=255,
        image_dims=(224, 224))

# Dimensions: 224x224, with 3 channels. Batch size 1
# NOTE: maybe can use batching to speed up processing?
net.blobs['data'].reshape(1, 3, 224, 224)


def load_image(path, echo=True):
    net.predict([caffe.io.load_image(path)], oversample=False)
    if echo:
        print("Predicted class is #{}.".format(net.blobs['prob'].data[0].argmax()))
        top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
        print labels[top_k]


sample_mask = []

# Array of vectors
vectors = []
# Array of file paths
vec_origin_file = []
# Array of (arrays of (x, y))
vec_location = []

if args.load_layer_dump_from is not None:
    print 'Loading raw layer dump file from', args.load_layer_dump_from
    f = open(args.load_layer_dump_from)
    [args_images, args_layer, args_sample_fraction,
        args_center_path, args_center_x,
        sample_mask, vectors, vec_origin_file, vec_location] = pickle.load(f)

    args.images = args_images
    args.layer = args_layer
    args.sample_fraction = args_sample_fraction
    args.center_only_path = args_center_path
    args.center_only_neuron_x = args_center_x

    n_clusters = int(args.n_clusters)
    f.close()
    print 'Finished loading dump.'
else:
    # Loop through every image in the given directory
    for (dirpath, dirnames, filenames) in walk(args.images):
        for filename in filenames:
            path = os.path.abspath(os.path.join(dirpath, filename))
            print 'Processed', path
            load_image(path, False)

            response = net.blobs[args.layer].data[0]
            num_responses = len(response)
            height_response = len(response[0])
            width_response = len(response[0][0])

            if len(sample_mask) == 0:
                # sample_mask not initialized yet; sample new
                print str(num_responses) + ' filters of ' + str(height_response) + 'x' + str(width_response)

                sample_mask = sample(width_response, height_response,
                        float(args.sample_fraction) * width_response * height_response)

            # TODO: this could be parallelized by multiplication -- then filtering out 0 columns
            for y in range(height_response):
                for x in range(width_response):
                    if sample_mask[y][x]:
                        ## NOTE: DOUBLE CHECK IF FIRST IS Y SECOND IS X, corresponding to images
                        vectors.append(response[:, y, x].copy())
                        vec_origin_file.append(path)
                        vec_location.append((x, y))

    print 'Got', len(vectors), 'vectors randomly sampled'
    # Load the images of which only the center patches will be used
    if args.center_only_path is not None:
        print 'Loading images of which center patches will be used'
        location_to_pick = int(args.center_only_neuron_x)
        print 'Center neuron x and y coordinate:', location_to_pick

        for (dirpath, dirnames, filenames) in walk(args.center_only_path):
            for filename in filenames:
                path = os.path.abspath(os.path.join(dirpath, filename))
                print 'Processed', path
                load_image(path, False)

                response = net.blobs[args.layer].data[0]
                vectors.append(response[:, location_to_pick, location_to_pick].copy())
                vec_origin_file.append(path)
                vec_location.append((location_to_pick, location_to_pick))

    # Save data (layer) dump if parameter is specified
    if args.save_layer_dump_to is not None:
        print 'Saving layer data dump to', args.save_layer_dump_to
        f = open(args.save_layer_dump_to, 'wb')
        pickle.dump([args.images, args.layer, args.sample_fraction,
            args.center_only_path, args.center_only_neuron_x,
            sample_mask, vectors, vec_origin_file, vec_location], f)
        f.close()
        print 'Finished saving layer dump'

print 'Got', len(vectors), 'vectors in total for clustering'

if args.load_classification_dump_from is not None:
    print 'Loading classification dump from', args.load_classification_dump_from
    f = open(args.load_classification_dump_from)
    dumped = pickle.load(f)
    f.close()
    kmeans_scores = []
    if len(dumped) == 3:
        n_clusters, kmeans_obj, predicted = dumped
    elif len(dumped) == 4:
        n_clusters, kmeans_obj, predicted, kmeans_scores = dumped
    args.n_clusters = n_clusters
    n_clusters = int(n_clusters)
    print 'Finished loading classification dump'
else:
    n_clusters = int(args.n_clusters)
    n_restarts = 10
    kmeans_obj = KMeans(init='k-means++', n_clusters=n_clusters, n_init=n_restarts)
    predicted = kmeans_obj.fit_predict(vectors)

    # Precompute distances of every vector to its cluster center
    kmeans_scores = []
    for vec_i in range(len(vectors)):
        kmeans_scores.append(kmeans_obj.score(vectors[vec_i].reshape(1, -1)))

    if args.save_classification_dump_to is not None:
        print 'Saving classification dump to', args.save_classification_dump_to
        f = open(args.save_classification_dump_to, 'wb')
        pickle.dump([n_clusters, kmeans_obj, predicted, kmeans_scores], f)
        f.close()
        print 'Finished saving classification dump'

def do_tsne(data):
    tsne_model = TSNE(n_components=2, init='pca')
    trans_tsne = tsne_model.fit_transform(data)
    return tsne_model, trans_tsne


## WARNING: THIS METHOD OF SELECTING MAY NOT BE WORKING AS YOU INTENDED!!!
def plot_clusters_2d(data, labels, selected_rows):
    plt.scatter(data[selected_rows, 0], data[selected_rows, 1], c=labels[selected_rows], cmap=plt.get_cmap('Spectral'), lw=0)
    plt.show()


#plot_clusters_2d(vectors20k_tsne[np.logical_or(predicted20k == 26, predicted20k == 54), :], predicted20k[np.logical_or(predicted20k == 26, predicted20k == 54)])

##### PCA
pca = PCA(n_components=80)
#vectors_trans = pca.fit_transform(vectors)
#print pca.explained_variance_ratio_
#predicted = kmeans_obj.fit_predict(vectors_trans)


#########


# Given an id, return the patch in the input image that generated vectors[vec_id]
def get_original_patch_of_vec(vec_id):
    this_vec_location = vec_location[vec_id]
    rec_field = rf.get_receptive_field(args.layer, this_vec_location[0], this_vec_location[1])
    load_image(vec_origin_file[vec_id], False)
    im = net.transformer.deprocess('data',
        net.blobs['data'].data[0][:,rec_field[1]:(rec_field[3]+1),rec_field[0]:(rec_field[2]+1)])
    return im


def get_sparsity(data):
    numer = 0
    denom = 0
    n = len(data) * 1.0
    for r in data:
        numer = numer + r / n
        denom = denom + r**2 / n

    A = numer**2 / denom
    S = (1 - A) / (1 - 1/n)
    return A, S


def get_activations_of_cluster(cluster_i):
    count = 0
    bigsum = None
    for idx in range(len(predicted)):
        if predicted[idx] == cluster_i:
            if bigsum is None:
                bigsum = vectors[idx]
            else:
                bigsum = bigsum + vectors[idx]
            count = count + 1.0

    return bigsum, count


def plot_raw_activation(cluster_i):
    totalsum, count = get_activations_of_cluster(cluster_i)
    totalsum = totalsum / count

    plt.plot(range(0, len(totalsum)), totalsum, 'b-')
    plt.title(args.layer + ' Cluster #' + str(cluster_i) + ' (' + str(args.n_clusters) + ' total)')
    plt.xlabel('Neuron #')
    plt.ylabel('Average activation')

    A, S = get_sparsity(totalsum)
    plt.annotate(
        'Sparsity: S=' + str(S),
        xy = (0.9, 0.9), xytext = (0.9, 0.9),
        textcoords = 'axes fraction', ha = 'right', va = 'bottom')

    plt.show()


def get_top_n_in_cluster(cluster_i, n):
    scores = []

    for vec_id in range(len(vectors)):
        if predicted[vec_id] == cluster_i:
            scores.append((vec_id, kmeans_obj.score(vectors[vec_id].reshape(1, -1))))

    scores.sort(key=lambda tup: -tup[1])
    if n == -1:
        return scores
    return scores[0:n]


def get_top_in_clusters(clusters_i):
    scores = {}

    for i in clusters_i:
        scores[i] = []

    if len(kmeans_scores) == 0:
        print 'Precomputed kmeans scores do not exist'
        for vec_i in range(len(vectors)):
            if predicted[vec_i] in clusters_i:
                scores[predicted[vec_i]].append((vec_i, kmeans_obj.score(vectors[vec_i].reshape(1, -1))))
    else:
        for vec_i in range(len(vectors)):
            if predicted[vec_i] in clusters_i:
                scores[predicted[vec_i]].append((vec_i, kmeans_scores[vec_i]))

    out = []
    for i in clusters_i:
        scores[i].sort(key=lambda tup: -tup[1])
        out.append(scores[i][0][0])
    return out


# Do MDS on cluster centers, plot on 2D
def plot_clusters_embedding():
    ax = plt.gca()
    _, centers_2d = do_tsne(kmeans_obj.cluster_centers_)
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1])
    clusters_top = get_top_in_clusters(range(args.n_clusters))

    for cluster_i in range(args.n_clusters):
        top_vector = clusters_top[cluster_i]
        im = get_original_patch_of_vec(top_vector)
        imagebox = OffsetImage(im, zoom=0.5)
        ab = AnnotationBbox(imagebox, (0., 0.), xybox=(centers_2d[cluster_i, 0], centers_2d[cluster_i, 1]),
                            pad=0, frameon=False, xycoords='data', boxcoords="data")
        ax.add_artist(ab)

    plt.show()


def plot_activation(cluster_i, top_n=4):
    grid_dims = (8, 9)
    ax = plt.subplot2grid(grid_dims, (0, 0), colspan=7, rowspan=8)

    bigsum, count = get_activations_of_cluster(cluster_i)
    bigsum /= count

    # Get sorted indexes
    sorted_indexes = [i[0] for i in sorted(enumerate(bigsum), key=lambda x:x[1], reverse=True)]
    top_indexes = [sorted_indexes[x] for x in range(top_n)]
    top_responses = [bigsum[sorted_indexes[x]] for x in range(top_n)]

    bigsum.sort()
    bigsum = bigsum[::-1]
    ax.plot(range(len(bigsum)), bigsum, 'bo-')
    ax.set_title(args.layer + ' Cluster #' + str(cluster_i) + ' (' + str(args.n_clusters) + ' total)')
    ax.set_xlabel('Neuron #')
    ax.set_ylabel('Average activation')

    print 'Highest neuron responses:'
    for i in range(top_n):
        print 'Neuron #', top_indexes[i], ', mean response: ', top_responses[i]

        plt.annotate(
            'Neuron #' + str(top_indexes[i]) + '\navg: ' + str(top_responses[i]),
            xy=(i, bigsum[i]), xytext=(100 + 20 * i, -50 - 20 * i),
            textcoords='offset points', ha='right', va='bottom',
            bbox = dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops = dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Print sparsity metric
    A, S = get_sparsity(bigsum)
    plt.annotate(
        'Sparsity: S=' + str(S),
        xy=(0.9, 0.9), xytext = (0.9, 0.9),
        textcoords='axes fraction', ha = 'right', va = 'bottom')

    # Rightmost column patch plot
    print 'Looking for vectors closest to the cluster center...'
    top_vectors = get_top_n_in_cluster(cluster_i, 8)
    for i, (vec_id, score) in enumerate(top_vectors):
        ax2 = plt.subplot2grid(grid_dims, (i, 7))
        if i == 0:
            ax2.set_title('Patches\nClosest\nto Cluster\nCenter', {'fontsize': 10})
        plt.axis('off')
        ax2.imshow(get_original_patch_of_vec(vec_id))

    # Rightmost column: patches that gen highest response for each neuron
    all_vectors = np.array(vectors)
    for i, neuron_i in enumerate(top_indexes):
        responses = all_vectors[:, neuron_i]
        max_vec_id = np.argmax(responses)
        ax3 = plt.subplot2grid(grid_dims, (i, 8))
        plt.axis('off')
        ax3.set_title('Neuron #' + str(neuron_i), {'fontsize': 10})
        ax3.imshow(get_original_patch_of_vec(max_vec_id))

    plt.show()


# Plot response of neuron_i on all patches in vectors array
def plot_stimuli_response(neuron_i, inputs=vectors, top_n=10):
    grid_dims = (top_n, top_n)
    ax = plt.subplot2grid(grid_dims, (0, 0), colspan=top_n-1, rowspan=top_n)

    all_vectors = np.array(inputs)
    responses = all_vectors[:, neuron_i]
    len_responses = len(responses)

    sorted_indexes = [i[0] for i in sorted(enumerate(responses), key=lambda x:x[1], reverse=True)]
    responses.sort()
    responses = responses[::-1]
    ax.plot(range(len(all_vectors)), responses, 'bo-')
    ax.set_title(args.layer + ' Neuron #' + str(neuron_i) + ' Responses to ' + str(len_responses) + ' Stimuli')
    ax.set_xlabel('Stimulus #')
    ax.set_ylabel('Activation')

    # Include patches with top n responses
    for i in range(top_n):
        ax2 = plt.subplot2grid(grid_dims, (i, top_n - 1))
        plt.axis('off')
        if i == 0:
            ax2.set_title('Top ' + str(top_n) + '\nPatches', {'fontsize': 10})
        im = get_original_patch_of_vec(sorted_indexes[i])
        ax2.imshow(im)

    # Find number of stimuli that generate response > 0.5*MAX
    num_half_height_stimuli = 0
    for i in range(len_responses):
        if responses[i] > 0.5 * responses[0]:
            num_half_height_stimuli = num_half_height_stimuli + 1

    print '# responses greater than half height:', num_half_height_stimuli, '(', num_half_height_stimuli * 1.0 / len_responses, ')'
    print 'Half height:', 0.5 * responses[0]

    plt.show()


def do_pca_on_neuron(neuron_i, do_ica=False):
    patches = []
    half_height = np.array(vectors)[:, neuron_i].max() * 0.5
    middle_neuron_i = math.floor(len(net.blobs[args.layer].data[0][0]) / 2) + 1
    rec_field = rf.get_receptive_field(args.layer, middle_neuron_i, middle_neuron_i)
    patch_dim = int(rec_field[2] - rec_field[0] + 1)
    print 'Patch dimension is', patch_dim
    print 'Receptive field of neuron in the center hypercolumn is', rec_field

    for i in range(len(predicted)):
        if vectors[i][neuron_i] > half_height:
            im = get_original_patch_of_vec(i)
            if (len(im) == patch_dim) and (len(im[0]) == patch_dim):
                patches.append(skimage.color.rgb2gray(im)) # WARNING: TURNED INTO GRAYSCALE!!
                # ... but should try to do PCA on 30000 dimensions and see what you will get too

    print 'Found', len(patches), 'patches with activations greater than half height'
    patches = np.array(patches)
    mean_patch = patches.mean(0)

    # Subtract the mean from all data
    flattened = []
    for i in range(len(patches)):
        patches[i] -= mean_patch
        flattened.append(patches[i].flatten())

    patches_pca = FastICA(n_components=100)
    patches_trans = patches_pca.fit_transform(flattened)
    #print sum(patches_pca.explained_variance_ratio_)

    for start_id in range(0, 99, 25):
        fig = plt.figure()
        dim_plot = 5
        for fig_id in range(dim_plot * dim_plot):
            print 'Creating figure', fig_id
            fig.add_subplot(dim_plot, dim_plot, fig_id + 1)
            plt.imshow(patches_pca.components_[fig_id + start_id].reshape(100, 100))
            plt.axis('off')
        plt.show()

    # Treat the coefficients as Gaussians and sample from them
    synthesized = np.empty(10000)
    for i in range(100):
        sampled = sklearn.datasets.make_gaussian_quantiles([patches_trans[:, i].mean()], patches_trans[:, i].var(), n_samples=1, n_classes=1, n_features=1)[0][0][0]
        synthesized += sampled * patches_pca.components_[i]

    tmp = synthesized.reshape(100, 100) + mean_patch
    plt.imshow(tmp)
    plt.show()

    return mean_patch, flattened, patches_pca, patches_trans


## Plot image with response of a certain neuron, or highest among all neurons in each hypercolumn
def dye_image_with_response(path):
    # Plot a heat map to show which places have highest activation
    heatmap = np.empty([224, 224])
    load_image(path)
    layer_response = net.blobs[args.layer].data[0]

    # Find the highest response in each hypercolumn
    dim = len(layer_response[0])
    for x in range(dim):
        for y in range(dim):
            max_response = np.max(layer_response[:, y, x])
            rec_field = rf.get_receptive_field(args.layer, x, y)

            # Update the heatmap with largest activation values
            for heatmap_x in range(rec_field[0], rec_field[2] + 1):
                for heatmap_y in range(rec_field[1], rec_field[3] + 1):
                    if heatmap[heatmap_y, heatmap_x] < max_response:
                        heatmap[heatmap_y, heatmap_x] = max_response

    ax = plt.gca()
    im = net.transformer.deprocess('data', net.blobs['data'].data[0])
    ax.imshow(im)
    ax.imshow(heatmap, alpha=0.7)
    plt.show()


def dye_image_with_neuron_response(path, neuron_i):
    pass


## distance_threshold: require a vector to be smaller than distance of distance_threshold * num of vectors in cluster
##                     for it to be considered as in this cluster
def find_patches_in_cluster(cluster_i, image_path, dist_thres_percentage=1.0):
    score_thres = 0.0
    cluster_scores = get_top_n_in_cluster(cluster_i, -1) # All score values in this cluster
    cluster_scores = [score for (vec_id, score) in cluster_scores]
    # Figure out exact value of distance threshold given the percentage
    score_thres = cluster_scores[int(math.floor(len(cluster_scores) * dist_thres_percentage)) - 1]
    print 'Limiting cluster score to smaller than', score_thres

    # Given an image, find the patches in the image that have responses in the given cluster
    load_image(image_path, False)
    dim_filter = len(net.blobs[args.layer].data[0][0])

    plt.imshow(net.transformer.deprocess('data', net.blobs['data'].data[0]))
    axis = plt.gca()

    total_patches_found = 0
    patches_ignored = 0 # Number of patches that are in the cluster but ignored due to score

    for y in range(dim_filter):
        for x in range(dim_filter):
            hypercolumn = net.blobs[args.layer].data[0][:,y,x].copy().reshape(1, -1)
            prediction = kmeans_obj.predict(hypercolumn)
            if prediction == cluster_i:
                total_patches_found = total_patches_found + 1
                if kmeans_obj.score(hypercolumn) < score_thres:
                    patches_ignored = patches_ignored + 1
                else:
                    rec_field = rf.get_receptive_field(args.layer, x, y)
                    #### NOTE: VERIFY THAT YOU GOT X AND Y RIGHT AGAIN!!!
                    axis.add_patch(Rectangle((rec_field[0], rec_field[1]),
                        rec_field[2] - rec_field[0] + 1,
                        rec_field[3] - rec_field[1] + 1,
                        fill=False, edgecolor="red"))

    print 'Found', total_patches_found, 'patches in total,', patches_ignored, 'ignored due to distance'
    plt.show()

def view_nth_in_cluster(cluster_i, i):
    num_in_cluster_seen = 0
    for vec_id in range(len(vectors)):
        if predicted[vec_id] == cluster_i:
            if num_in_cluster_seen == i:
                plt.imshow(get_original_patch_of_vec(vec_id))
                plt.show()
                return
            else:
                num_in_cluster_seen = num_in_cluster_seen + 1


# View n images in the cluster_i-th cluster that are closest to the center
def view_nth_cluster(cluster_i, n):
    fig = plt.figure()
    fig_id = 1
    dim_plot = math.floor(math.sqrt(n))
    if dim_plot * dim_plot < n:
        dim_plot = dim_plot + 1

    scores = get_top_n_in_cluster(cluster_i, n)

    for (vec_id, score) in scores:
        print 'Vector #', vec_id, 'with score', score

        fig.add_subplot(dim_plot, dim_plot, fig_id)

        im = get_original_patch_of_vec(vec_id)
        plt.imshow(im)
        plt.axis('off')
        fig_id = fig_id + 1

        if fig_id > n:
            plt.show()
            return

    plt.show()

def view_n_from_clusters(from_cluster, to_cluster, n_each, save_plots=False):
    fig = plt.figure()
    fig_id = 1

    for i in range(from_cluster, to_cluster+1):
        scores = get_top_n_in_cluster(i, n_each)
        for (vec_id, score) in scores:
            print 'Vector #', vec_id, 'in cluster #', i, ', score:', score
            fig.add_subplot(to_cluster - from_cluster + 1, n_each, fig_id)
            plt.imshow(get_original_patch_of_vec(vec_id))

            fig_id = fig_id + 1

            plt.axis('off')

    if save_plots:
        plt.savefig(os.path.join(args.save_plots_to, args.layer + '_' + 'clusters' + str(from_cluster) + 'to' + str(to_cluster) + '.png'))
    else:
        plt.show()

if args.save_plots_to is not None:
    matplotlib.use('Agg')
    if not os.path.exists(args.save_plots_to):
        os.makedirs(args.save_plots_to)
    # 8 clusters each time, 16 images for each cluster
    num_rounds = n_clusters / 8
    if n_clusters % 8 != 0:
        num_rounds = num_rounds + 1

    for i in range(num_rounds):
        start = i * 8
        end = start + 7
        if end >= n_clusters:
            end = n_clusters - 1
        view_n_from_clusters(start, end, 16, True)

if args.append_kmeans_scores:
    kmeans_scores = []
    for vec_i in range(len(vectors)):
        kmeans_scores.append(kmeans_obj.score(vectors[vec_i].reshape(1, -1)))

    print 'Saving classification and kmeans scores dump to', args.save_classification_dump_to
    f = open(args.save_classification_dump_to, 'wb')
    pickle.dump([n_clusters, kmeans_obj, predicted, kmeans_scores], f)
    f.close()
print 'All tasks completed. Exiting'
