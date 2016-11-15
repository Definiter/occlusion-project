# Set this to the path to Caffe installation on your system
caffe_root = "/home/haow3/software/caffe-rc3/python"
gpu = True

# -------------------------------------
# These settings should work by default
# DNN being visualized
# These two settings are default, and can be overriden in the act_max.py
synthesizing_root = "/home/haow3/occlusion-project/src/synthesizing/"
net_weights = synthesizing_root + "nets/caffenet/bvlc_reference_caffenet.caffemodel"
net_definition = synthesizing_root + "nets/caffenet/caffenet.prototxt"

# Generator DNN
generator_weights = synthesizing_root + "nets/upconv/fc6/generator.caffemodel"
generator_definition = synthesizing_root + "nets/upconv/fc6/generator.prototxt"

# Encoder DNN
encoder_weights = synthesizing_root + "nets/caffenet/bvlc_reference_caffenet.caffemodel"
encoder_definition = synthesizing_root + "nets/caffenet/caffenet.prototxt"
