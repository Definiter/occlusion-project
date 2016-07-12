'''
Generate a file list containing images in a given directory, with all the same given label
'''

import argparse
import os
from os import walk

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir', required=True)
parser.add_argument('-o', '--output_file', required=True)
parser.add_argument('-l', '--label', required=True)
args = parser.parse_args()

image_dir_prefix_len = len(args.input_dir)
if not args.input_dir.endswith('/'):
    image_dir_prefix_len += 1

in_f = open(args.output_file, 'w')
for (dirpath, dirnames, filenames) in walk(args.input_dir):
    for filename in filenames:
        path = os.path.abspath(os.path.join(dirpath, filename))
        # Take the path without the prefix
        in_f.write(path[image_dir_prefix_len:] + ' ' + args.label + '\n')

in_f.close()
