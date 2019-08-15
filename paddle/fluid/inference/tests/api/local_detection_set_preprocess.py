# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#import image_util
#from paddle.utils.image_util import *

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import numpy as np
import xml.etree.ElementTree
import os
import time
import six
import argparse
import functools
import distutils.util
import sys
from optparse import OptionParser


def preprocess(args, img):
    img_width, img_height = img.size

    img = img.resize((args.resize_w, args.resize_h), Image.ANTIALIAS)
    img = np.array(img)

    # HWC to CHW
    if len(img.shape) == 3:
        img = np.swapaxes(img, 1, 2)
        img = np.swapaxes(img, 1, 0)
    # RBG to BGR
    img = img[[2, 1, 0], :, :]
    img = img.astype('float32')
    img_mean = np.array(args.mean_value)[:, np.newaxis, np.newaxis].astype(
        'float32')
    img -= img_mean
    img *= 0.007843
    return img


def pascalvoc(args):
    data_dir = os.path.expanduser(args.data_dir)

    label_list = []
    label_fpath = os.path.join(data_dir, args.label_file)
    for line in open(label_fpath):
        label_list.append(line.strip())

    file_list_path = os.path.join(data_dir, args.file_list)
    flist = open(file_list_path)
    lines = [line.strip() for line in flist]

    image_out_path = os.path.join(data_dir, args.image_out)
    f1 = open(image_out_path, "w+b")
    f1.seek(0)
    line_len = len(lines)
    f1.write(np.array(line_len).astype('int64').tobytes())

    boxes = []
    lbls = []
    difficults = []
    object_nums = []

    for line in lines:
        image_path, label_path = line.split()
        image_path = os.path.join(data_dir, image_path)
        label_path = os.path.join(data_dir, label_path)

        im = Image.open(image_path)
        if im.mode == 'L':
            im = im.convert('RGB')
        im_width, im_height = im.size

        im = preprocess(args, im)
        np_im = np.array(im)
        f1.write(np_im.astype('float32').tobytes())

        # layout: label | xmin | ymin | xmax | ymax | difficult
        bbox_labels = []
        root = xml.etree.ElementTree.parse(label_path).getroot()

        objects = root.findall('object')
        objects_size = len(objects)
        object_nums.append(objects_size)

        for object in objects:
            bbox_sample = []
            # start from 1
            bbox_sample.append(
                float(label_list.index(object.find('name').text)))
            bbox = object.find('bndbox')
            difficult = float(object.find('difficult').text)
            bbox_sample.append(float(bbox.find('xmin').text) / im_width)
            bbox_sample.append(float(bbox.find('ymin').text) / im_height)
            bbox_sample.append(float(bbox.find('xmax').text) / im_width)
            bbox_sample.append(float(bbox.find('ymax').text) / im_height)
            bbox_sample.append(difficult)
            bbox_labels.append(bbox_sample)

        bbox_labels = np.array(bbox_labels)
        if len(bbox_labels) == 0: continue

        lbls.extend(bbox_labels[:, 0])
        boxes.extend(bbox_labels[:, 1:5])
        difficults.extend(bbox_labels[:, -1])

    f1.write(np.array(object_nums).astype('uint64').tobytes())
    f1.write(np.array(lbls).astype('int64').tobytes())
    print(np.array(lbls).astype('int64'))
    f1.write(np.array(boxes).astype('float32').tobytes())
    f1.write(np.array(difficults).astype('int64').tobytes())
    f1.close()


def parse_list(string):
    return eval(string)


def check_data_dir(string):
    if not os.path.exists(string):
        raise ValueError("Path %s do not exist." % string)
    return string


def main(args):
    parser = OptionParser(description=__doc__)
    parser.add_option(
        "-d",
        "--data_dir",
        type=str,
        default="/pascalvoc_small",
        metavar="DATA_DIR",
        action='callback',
        callback=check_data_dir,
        help="Dataset directory")
    parser.add_option(
        "-f",
        "--file_list",
        type=str,
        default="test.txt",
        metavar="FILE_LIST",
        help="A file containing the image file path and relevant annotation file path"
    )
    parser.add_option(
        "-l",
        "--label_file",
        type=str,
        default="label_list",
        help="List the all labels in the same sequence as denoted in the annotation file"
    )
    parser.add_option(
        "-i", "--image_out", type=str, default='pascalvoc_small.bin')
    parser.add_option("-c", "--resize_h", type=int, default=300)
    parser.add_option("-w", "--resize_w", type=int, default=300)
    parser.add_option(
        "-m",
        "--mean_value",
        type=str,
        default='[127.5, 127.5, 127.5]',
        action='callback',
        callback=parse_list)
    parser.add_option("-a", "--ap_version", type=str, default='11point')

    (options, _) = parser.parse_args(args)
    print(options.data_dir, options.file_list, options.label_file,
          options.resize_h, options.resize_w, options.mean_value,
          options.ap_version, options.image_out)
    pascalvoc(options)


if __name__ == "__main__":
    main(sys.argv)
