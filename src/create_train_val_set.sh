#!/bin/bash

# Main dir for output files
main_dir=/home/haow3/occlusion-project/drive

# If validation set dir and filelist are provided, generate val set; otherwise, don't
if [ $# -eq 5 ]; then
    # Subset of imagenet used for training
    IMAGENET_TRAIN_PATH=$1
    IMAGENET_TRAIN_FILELIST=$2
    # Subset of imagenet used for validation
    IMAGENET_VAL_PATH=$3
    IMAGENET_VAL_FILELIST=$4
    # Exclude mask class images when generating training list
    FACE_CLASS_PATH=$5
elif [ $# -eq 3 ]; then
    IMAGENET_TRAIN_PATH=$1
    IMAGENET_TRAIN_FILELIST=$2
    FACE_CLASS_PATH=$3
else
    echo "Unrecognized number of arguments"
    exit 1
fi

echo "Imagenet training images path: $IMAGENET_TRAIN_PATH"
echo "Imagenet training images list path: $IMAGENET_TRAIN_FILELIST"
echo "Face images path: $FACE_CLASS_PATH"

MASK_CLASS=643

train_path_dir_name=$(basename $IMAGENET_TRAIN_PATH)
face_path_dir_name=$(basename $FACE_CLASS_PATH)

# Regex for matching face class images from imagenet images, so that they can be put back later
FACE_CLASS_FILE_REGEX=/

# First: generate list of imagenet images without mask class
TRAIN_EXCLUDED_OUT=$main_dir/train_no_$MASK_CLASS.txt
if [ -f $TRAIN_EXCLUDED_OUT ] ; then
    rm $TRAIN_EXCLUDED_OUT
fi
python ~/occlusion-project/src/filter_filelist.py -i $IMAGENET_TRAIN_FILELIST -o $TRAIN_EXCLUDED_OUT -l $MASK_CLASS
# Prefix it with folder name
sed -e "s/^/$train_path_dir_name\//" $TRAIN_EXCLUDED_OUT > $main_dir/tmp.txt
rm $TRAIN_EXCLUDED_OUT
mv $main_dir/tmp.txt $TRAIN_EXCLUDED_OUT

# Generate a list of face class images with mask class
TRAIN_FACE_OUT=$main_dir/train_face_class.txt
if [ -f $TRAIN_FACE_OUT ] ; then
    rm $TRAIN_FACE_OUT
fi
python ~/occlusion-project/src/generate_filelist.py -i $FACE_CLASS_PATH -o $TRAIN_FACE_OUT -l $MASK_CLASS
# Prefix the filelist with folder name
sed -e "s/^/$face_path_dir_name\//" $TRAIN_FACE_OUT > $main_dir/tmp.txt
rm $TRAIN_FACE_OUT
mv $main_dir/tmp.txt $TRAIN_FACE_OUT

# Combine these two files into one file of all training images
cd $main_dir
cat train_face_class.txt train_no_$MASK_CLASS.txt > train.txt

# Then, generate list of imagenet images without mask class for validation
VAL_EXCLUDED_OUT=$main_dir/val.txt
if [ -f $VAL_EXCLUDED_OUT ] ; then
    rm $VAL_EXCLUDED_OUT
fi

if [ $# -eq 5 ]; then
    python ~/occlusion-project/src/filter_filelist.py -i $IMAGENET_VAL_FILELIST -o $VAL_EXCLUDED_OUT -l $MASK_CLASS
fi

# Create a dir that contains all training images
mkdir all_train
mv $IMAGENET_TRAIN_PATH all_train/
mv $FACE_CLASS_PATH     all_train/

if [ $# -eq 5 ]; then
    # Call script to create lmdb file of training and validation sets
    ~/occlusion-project/src/create_lmdb.sh $main_dir/all_train $main_dir/train.txt $IMAGENET_VAL_PATH $main_dir/val.txt \
        $main_dir/${face_path_dir_name}_train_lmdb \
        $main_dir/${face_path_dir_name}_val_lmdb
else
    ~/occlusion-project/src/create_lmdb.sh $main_dir/all_train $main_dir/train.txt $main_dir/${face_path_dir_name}_train_lmdb
fi

# Move the training directories back to their original locations
mv all_train/$train_path_dir_name $IMAGENET_TRAIN_PATH
mv all_train/$face_path_dir_name  $FACE_CLASS_PATH
rmdir all_train

# Cleanup
rm $TRAIN_EXCLUDED_OUT
rm $TRAIN_FACE_OUT
rm $main_dir/train.txt
if [ $# -eq 5 ]; then
    rm $VAL_EXCLUDED_OUT
fi
