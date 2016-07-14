#!/usr/bin/env sh
# Create the imagenet lmdb inputs.
# N.B. set the path to the imagenet train + val data dirs

TOOLS=/home/haow3/software/caffe-rc3/build/tools

if [ $# -eq 6 ]; then
    TRAIN_DATA_ROOT=$1
    TRAIN_FILELIST=$2
    VAL_DATA_ROOT=$3
    VAL_FILELIST=$4
    TRAIN_LMDB_OUT=$5
    VAL_LMDB_OUT=$6
elif [ $# -eq 3 ]; then
    TRAIN_DATA_ROOT=$1
    TRAIN_FILELIST=$2
    TRAIN_LMDB_OUT=$3
else
    echo "Unrecognized number of arguments"
    exit 1
fi

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  # WARNING!! DIFFERENCE BETWEEN 227 (input size of alexnet), 224 (input size of vgg) AND 256!!
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
    echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
    echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
        "where the ImageNet training data is stored."
    exit 1
fi

if [ $# -eq 6 ]; then
    if [ ! -d "$VAL_DATA_ROOT" ]; then
        echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
        echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
            "where the ImageNet validation data is stored."
        exit 1
    fi
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT/ \
    $TRAIN_FILELIST \
    $TRAIN_LMDB_OUT

if [ $# -eq 6 ]; then
    echo "Creating val lmdb..."

    GLOG_logtostderr=1 $TOOLS/convert_imageset \
        --resize_height=$RESIZE_HEIGHT \
        --resize_width=$RESIZE_WIDTH \
        --shuffle \
        $VAL_DATA_ROOT/ \
        $VAL_FILELIST \
        $VAL_LMDB_OUT
fi

echo "Done."
