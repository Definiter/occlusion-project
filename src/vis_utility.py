from constant import *
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE
import sklearn.datasets

import math
import numpy as np
import pickle

import skimage
import caffe

# Get receptive field.
layers = [
    'data',
    'conv1_1',
    'conv1_2',
    'pool1',
    'conv2_1',
    'conv2_2',
    'pool2',
    'conv3_1',
    'conv3_2',
    'conv3_3',
    'pool3',
    'conv4_1',
    'conv4_2',
    'conv4_3',
    'pool4',
    'conv5_1',
    'conv5_2',
    'conv5_3',
    'pool5'
]

# [ pad, kernel_size ]
conv_params = {
    'conv1_1': [1, 3],
    'conv1_2': [1, 3],
    'conv2_1': [1, 3],
    'conv2_2': [1, 3],
    'conv3_1': [1, 3],
    'conv3_2': [1, 3],
    'conv3_3': [1, 3],
    'conv4_1': [1, 3],
    'conv4_2': [1, 3],
    'conv4_3': [1, 3],
    'conv5_1': [1, 3],
    'conv5_2': [1, 3],
    'conv5_3': [1, 3]
}

# [stride, kernel_size]
pool_params = {
    'pool1': [2, 2],
    'pool2': [2, 2],
    'pool3': [2, 2],
    'pool4': [2, 2],
    'pool5': [2, 2]
}

layer_dims = {
    'data': 224,
    'conv1_1': 224,
    'conv1_2': 224,
    'pool1': 112,
    'conv2_1': 112,
    'conv2_2': 112,
    'pool2': 56,
    'conv3_1': 56,
    'conv3_2': 56,
    'conv3_3': 56,
    'pool3': 28
}

dim_input = 224

prev_layers = {}
# Build dictionary that maps one layer to its prev layer
for i in range(len(layers) - 1):
    prev_layers[layers[i+1]] = layers[i]


# Returns: top-left corner x, y, and bottom-right corner x, y
# x is the column of the neuron in the 2D array representing the filter response
# y is the row
# NOTE: THERE COULD BE PROBLEM HERE. DOUBLE CHECK
def get_conv_neuron_rec_field(layer, neuron_x, neuron_y):
    last_layer_top_left = [-1, -1]
    last_layer_bottom_right = [-1, -1]

    # stride == 1 for VGG16
    params = conv_params[layer]
    pad = params[0]
    kernel_size = params[1]

    last_layer_top_left[0] = neuron_x - pad
    last_layer_top_left[1] = neuron_y - pad
    last_layer_bottom_right[0] = neuron_x - pad + kernel_size - 1
    last_layer_bottom_right[1] = neuron_y - pad + kernel_size - 1

    last_layer = prev_layers[layer]

    if last_layer.startswith('conv'):
        last_layer_tl_rec_field = get_conv_neuron_rec_field(prev_layers[layer],
                                                            last_layer_top_left[0],
                                                            last_layer_top_left[1])
        last_layer_br_rec_field = get_conv_neuron_rec_field(prev_layers[layer],
                                                            last_layer_bottom_right[0],
                                                            last_layer_bottom_right[1])
    elif last_layer.startswith('pool'):
        last_layer_tl_rec_field = get_pool_neuron_rec_field(prev_layers[layer],
                                                            last_layer_top_left[0],
                                                            last_layer_top_left[1])
        last_layer_br_rec_field = get_pool_neuron_rec_field(prev_layers[layer],
                                                            last_layer_bottom_right[0],
                                                            last_layer_bottom_right[1])
    else:
        # Data layer
        return [last_layer_top_left[0], last_layer_top_left[1],
                last_layer_bottom_right[0], last_layer_bottom_right[1]]

    return [last_layer_tl_rec_field[0], last_layer_tl_rec_field[1],
            last_layer_br_rec_field[2], last_layer_br_rec_field[3]]


# Returns: same as the above
def get_pool_neuron_rec_field(layer, neuron_x, neuron_y):
    # [x, y] of top-left neuron in the previous layer this neuron is looking at
    last_layer_top_left = [-1, -1]
    last_layer_bottom_right = [-1, -1]

    params = pool_params[layer]
    stride = params[0]
    kernel_size = params[1]

    last_layer_top_left[0] = stride * neuron_x
    last_layer_top_left[1] = stride * neuron_y
    last_layer_bottom_right[0] = stride * neuron_x + kernel_size - 1
    last_layer_bottom_right[1] = stride * neuron_y + kernel_size - 1

    last_layer_tl_rec_field = get_conv_neuron_rec_field(prev_layers[layer],
                                                        last_layer_top_left[0],
                                                        last_layer_top_left[1])
    last_layer_br_rec_field = get_conv_neuron_rec_field(prev_layers[layer],
                                                        last_layer_bottom_right[0],
                                                        last_layer_bottom_right[1])

    return [last_layer_tl_rec_field[0], last_layer_tl_rec_field[1],
            last_layer_br_rec_field[2], last_layer_br_rec_field[3]]


def cap_value(v, maximum):
    if v < 0:
        return 0
    elif v > maximum:
        return maximum
    else:
        return v


def get_receptive_field(layer, x, y, print_size=False):
    rec_field = None
    if layer.startswith('pool'):
        rec_field = get_pool_neuron_rec_field(layer, x, y)
    elif layer.startswith('conv'):
        rec_field = get_conv_neuron_rec_field(layer, x, y)

    # Must be a square
    assert(rec_field[2] - rec_field[0] == rec_field[3] - rec_field[1])

    # Cap the values at boundaries of the image
    for i in range(len(rec_field)):
        rec_field[i] = cap_value(rec_field[i], dim_input - 1)

    if print_size:
        print 'Size:', rec_field[3] - rec_field[0] + 1

    return rec_field


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
#pca = PCA(n_components=80)
#vectors_trans = pca.fit_transform(vectors)
#print pca.explained_variance_ratio_
#predicted = kmeans_obj.fit_predict(vectors_trans)


#########


# Given an id, return the patch in the input image that generated vectors[vec_id]
def get_original_patch_of_vec(vec_id, net, cluster):
    this_vec_location = cluster.vectors[vec_id].location
    rec_field = get_receptive_field(cluster.layer, this_vec_location[0], this_vec_location[1])
    net.predict([caffe.io.load_image(cluster.vectors[vec_id].origin_file)], oversample=False)
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

# Get sum of vectors in a cluster.
def get_activations_of_cluster(cluster_i, cluster):
    count = 0
    bigsum = None 
    for idx in range(len(cluster.predicted)):
        if cluster.predicted[idx] == cluster_i:
            if bigsum is None:
                bigsum = cluster.vectors[idx].data
            else:
                bigsum = bigsum + cluster.vectors[idx].data
            count = count + 1.0

    return bigsum, count

# Plot the center of a cluster.
def plot_raw_activation(cluster_i, cluster):
    totalsum, count = get_activations_of_cluster(cluster_i, cluster)
    totalsum = totalsum / count # Cluster center.

    plt.plot(range(0, len(totalsum)), totalsum, 'b-')
    plt.title(cluster.layer + ' Cluster #' + str(cluster_i) + ' (' + str(cluster.n_clusters) + ' total)')
    plt.xlabel('Neuron #')
    plt.ylabel('Average activation')

    A, S = get_sparsity(totalsum)
    plt.annotate(
        'Sparsity: S=' + str(S),
        xy = (0.9, 0.9), xytext = (0.9, 0.9),
        textcoords = 'axes fraction', ha = 'right', va = 'bottom')

    plt.show()

# Get n closest vectors in the cluster.
def get_top_n_in_cluster(cluster_i, n, cluster):
    scores = []

    for vec_id in range(len(cluster.vectors)):
        if cluster.predicted[vec_id] == cluster_i:
            scores.append((vec_id, cluster.kmeans.score(cluster.vectors[vec_id].data.reshape(1, -1))))

    scores.sort(key=lambda tup: -tup[1])
    if n == -1:
        return scores
    return scores[0:n]

# Get the closest vector of each cluster
def get_top_in_clusters(clusters_i, cluster):
    scores = {}

    for i in clusters_i:
        scores[i] = []

    if len(cluster.kmeans_scores) == 0:
        print 'Precomputed kmeans scores do not exist'
        for vec_i in range(len(cluster.vectors)):
            if cluster.predicted[vec_i] in clusters_i:
                scores[cluster.predicted[vec_i]].append((vec_i, cluster.kmeans.score(cluster.vectors[vec_i].data.reshape(1, -1))))
    else:
        for vec_i in range(len(cluster.vectors)):
            if cluster.predicted[vec_i] in clusters_i:
                scores[cluster.predicted[vec_i]].append((vec_i, cluster.kmeans_scores[vec_i]))

    out = []
    for i in clusters_i:
        scores[i].sort(key=lambda tup: -tup[1])
        out.append(scores[i][0][0])
    return out


# Do MDS on cluster centers, plot on 2D, show top vector's patch in each cluster center
def plot_clusters_embedding(cluster, vectors, net):
    ax = plt.gca()
    _, centers_2d = do_tsne(cluster.kmeans.cluster_centers_)
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1])
    clusters_top = get_top_in_clusters(range(cluster.n_clusters), cluster)

    for cluster_i in range(cluster.n_clusters):
        top_vector = clusters_top[cluster_i]
        im = get_original_patch_of_vec(top_vector, net, cluster)
        imagebox = OffsetImage(im, zoom=0.5)
        ab = AnnotationBbox(imagebox, (0., 0.), xybox=(centers_2d[cluster_i, 0], centers_2d[cluster_i, 1]),
                            pad=0, frameon=False, xycoords='data', boxcoords="data")
        ax.add_artist(ab)

    plt.show()

# Plot cluster center vector, sorting by activation
def plot_activation(cluster_i, cluster, net, top_n=4):
    grid_dims = (8, 9)
    ax = plt.subplot2grid(grid_dims, (0, 0), colspan=7, rowspan=8)

    bigsum, count = get_activations_of_cluster(cluster_i, cluster)
    bigsum /= count

    # Get sorted indexes
    sorted_indexes = [i[0] for i in sorted(enumerate(bigsum), key=lambda x:x[1], reverse=True)]
    top_indexes = [sorted_indexes[x] for x in range(top_n)]
    top_responses = [bigsum[sorted_indexes[x]] for x in range(top_n)]

    bigsum.sort()
    bigsum = bigsum[::-1]
    ax.plot(range(len(bigsum)), bigsum, 'bo-')
    ax.set_title(cluster.layer + ' Cluster #' + str(cluster_i) + ' (' + str(cluster.n_clusters) + ' total)')
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

    # Rightmost column patch plot, getting n closest vectors in the cluster
    print 'Looking for vectors closest to the cluster center...'
    top_vectors = get_top_n_in_cluster(cluster_i, 8, cluster)
    for i, (vec_id, score) in enumerate(top_vectors):
        ax2 = plt.subplot2grid(grid_dims, (i, 7))
        if i == 0:
            ax2.set_title('Patches\nClosest\nto Cluster\nCenter', {'fontsize': 10})
        plt.axis('off')
        ax2.imshow(get_original_patch_of_vec(vec_id, net, cluster))

    # Rightmost column: patches that gen highest response for each neuron
    all_vectors = np.array([v.data for v in cluster.vectors])
    for i, neuron_i in enumerate(top_indexes):
        responses = all_vectors[:, neuron_i]
        max_vec_id = np.argmax(responses)
        ax3 = plt.subplot2grid(grid_dims, (i, 8))
        plt.axis('off')
        ax3.set_title('Neuron #' + str(neuron_i), {'fontsize': 10})
        ax3.imshow(get_original_patch_of_vec(max_vec_id, net, cluster))

    plt.show()


# TODO: bookmark
# Plot response of neuron_i on all patches in vectors array
def plot_stimuli_response(neuron_i, vectors, cluster, net, top_n=10):
    grid_dims = (top_n, top_n)
    ax = plt.subplot2grid(grid_dims, (0, 0), colspan=top_n-1, rowspan=top_n)

    all_vectors = np.array([v.data for v in vectors])
    responses = all_vectors[:, neuron_i]
    len_responses = len(responses)

    sorted_indexes = [i[0] for i in sorted(enumerate(responses), key=lambda x:x[1], reverse=True)]
    responses.sort()
    responses = responses[::-1]
    ax.plot(range(len(all_vectors)), responses, 'bo-')
    ax.set_title(cluster.layer + ' Neuron #' + str(neuron_i) + ' Responses to ' + str(len_responses) + ' Stimuli')
    ax.set_xlabel('Stimulus #')
    ax.set_ylabel('Activation')

    # Include patches with top n responses
    for i in range(top_n):
        ax2 = plt.subplot2grid(grid_dims, (i, top_n - 1))
        plt.axis('off')
        if i == 0:
            ax2.set_title('Top ' + str(top_n) + '\nPatches', {'fontsize': 10})
        im = get_original_patch_of_vec(sorted_indexes[i], net, cluster)
        ax2.imshow(im)

    # Find number of stimuli that generate response > 0.5*MAX
    num_half_height_stimuli = 0
    for i in range(len_responses):
        if responses[i] > 0.5 * responses[0]:
            num_half_height_stimuli = num_half_height_stimuli + 1

    print '# responses greater than half height:', num_half_height_stimuli, '(', num_half_height_stimuli * 1.0 / len_responses, ')'
    print 'Half height:', 0.5 * responses[0]

    plt.show()


def do_pca_on_neuron(neuron_i, cluster, net, do_ica=False):
    patches = []
    half_height = np.array([v.data for v in cluster.vectors])[:, neuron_i].max() * 0.5
    middle_neuron_i = math.floor(len(net.blobs[cluster.layer].data[0][0]) / 2) + 1
    rec_field = get_receptive_field(cluster.layer, middle_neuron_i, middle_neuron_i)
    patch_dim = int(rec_field[2] - rec_field[0] + 1)
    print 'Patch dimension is', patch_dim
    print 'Receptive field of neuron in the center hypercolumn is', rec_field

    for i in range(len(cluster.predicted)):
        if cluster.vectors[i].data[neuron_i] > half_height:
            im = get_original_patch_of_vec(i, net, cluster)
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
def dye_image_with_response(path, cluster, net):
    # Plot a heat map to show which places have highest activation
    heatmap = np.empty([224, 224])
    net.predict([caffe.io.load_image(path)], oversample=False)
    layer_response = net.blobs[cluster.layer].data[0]

    # Find the highest response in each hypercolumn
    dim = len(layer_response[0])
    for x in range(dim):
        for y in range(dim):
            max_response = np.max(layer_response[:, y, x])
            rec_field = get_receptive_field(cluster.layer, x, y)

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
def find_patches_in_cluster(cluster_i, image_path, net, cluster, dist_thres_percentage=1.0):
    score_thres = 0.0
    cluster_scores = get_top_n_in_cluster(cluster_i, -1, cluster) # All score values in this cluster
    cluster_scores = [score for (vec_id, score) in cluster_scores]
    # Figure out exact value of distance threshold given the percentage
    score_thres = cluster_scores[int(math.floor(len(cluster_scores) * dist_thres_percentage)) - 1]
    print 'Limiting cluster score to smaller than', score_thres

    # Given an image, find the patches in the image that have responses in the given cluster
    net.predict([caffe.io.load_image(image_path)], oversample=False)
    dim_filter = len(net.blobs[cluster.layer].data[0][0])

    plt.imshow(net.transformer.deprocess('data', net.blobs['data'].data[0]))
    axis = plt.gca()

    total_patches_found = 0
    patches_ignored = 0 # Number of patches that are in the cluster but ignored due to score

    for y in range(dim_filter):
        for x in range(dim_filter):
            hypercolumn = net.blobs[cluster.layer].data[0][:,y,x].copy().reshape(1, -1)
            prediction = cluster.kmeans.predict(hypercolumn)
            if prediction == cluster_i:
                total_patches_found = total_patches_found + 1
                if cluster.kmeans.score(hypercolumn) < score_thres:
                    patches_ignored = patches_ignored + 1
                else:
                    rec_field = get_receptive_field(cluster.layer, x, y)
                    #### NOTE: VERIFY THAT YOU GOT X AND Y RIGHT AGAIN!!!
                    axis.add_patch(Rectangle((rec_field[0], rec_field[1]),
                        rec_field[2] - rec_field[0] + 1,
                        rec_field[3] - rec_field[1] + 1,
                        fill=False, edgecolor="red"))

    print 'Found', total_patches_found, 'patches in total,', patches_ignored, 'ignored due to distance'
    plt.show()

def view_nth_in_cluster(cluster_i, i, cluster, net):
    num_in_cluster_seen = 0
    for vec_id in range(len(cluster.vectors)):
        if cluster.predicted[vec_id] == cluster_i:
            if num_in_cluster_seen == i:
                plt.imshow(get_original_patch_of_vec(vec_id, net, cluster))
                plt.show()
                return
            else:
                num_in_cluster_seen = num_in_cluster_seen + 1


# View n images in the cluster_i-th cluster that are closest to the center
def view_nth_cluster(cluster_i, n, cluster, net):
    fig = plt.figure()
    fig_id = 1
    dim_plot = math.floor(math.sqrt(n))
    if dim_plot * dim_plot < n:
        dim_plot = dim_plot + 1

    scores = get_top_n_in_cluster(cluster_i, n, cluster)

    for (vec_id, score) in scores:
        print 'Vector #', vec_id, 'with score', score

        fig.add_subplot(dim_plot, dim_plot, fig_id)

        im = get_original_patch_of_vec(vec_id, net, cluster)
        plt.imshow(im)
        plt.axis('off')
        fig_id = fig_id + 1

        if fig_id > n:
            plt.show()
            return

    plt.show()

def view_n_from_clusters(from_cluster, to_cluster, n_each, save_plots, net, cluster):
    fig = plt.figure()
    fig_id = 1

    for i in range(from_cluster, to_cluster+1):
        scores = get_top_n_in_cluster(i, n_each, cluster)
        for (vec_id, score) in scores:
            print 'Vector #', vec_id, 'in cluster #', i, ', score:', score
            fig.add_subplot(to_cluster - from_cluster + 1, n_each, fig_id)
            plt.imshow(get_original_patch_of_vec(vec_id, net, cluster))

            fig_id = fig_id + 1

            plt.axis('off')

    if save_plots:
        p = os.path.join(research_root + 'result/' + dataset_name, cluster.layer + '_' + 'clusters' + str(from_cluster) + 'to' + str(to_cluster) + '.png')
        print p
        plt.savefig(os.path.join(research_root + 'result/' + dataset_name, cluster.layer + '_' + 'clusters' + str(from_cluster) + 'to' + str(to_cluster) + '.png'))
    else:
        plt.show()
        
        
def visualize_cluster(net, cluster):
    plt.ioff()
    plt.rcParams['figure.figsize'] = (20, 20)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    if not os.path.exists(research_root + 'result/' + dataset_name):
        os.makedirs(research_root + 'result/' + dataset_name)
    # Visualize 8 clusters each time, 16 images for each cluster
    num_rounds = cluster.n_cluster / 8
    if cluster.n_cluster % 8 != 0:
        num_rounds = num_rounds + 1
    for i in range(num_rounds):
        start = i * 8
        end = start + 7
        if end >= cluster.n_cluster:
            end = cluster.n_cluster - 1
        view_n_from_clusters(start, end, 16, True, net, cluster)

