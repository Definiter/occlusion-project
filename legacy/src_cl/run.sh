mkdir ../result/$1
python vgg.py --images=../data/$1 --save_layer_dump_to=../result/$1/layer.dump --save_classification_dump_to=../result/$1/classification.dump --save_plots_to=../result/$1/
