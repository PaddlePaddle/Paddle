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
from PIL import Image
import numpy as np
import xml.etree.ElementTree
import os
import time
import six

DATA_DIR = "~/data/pascalvoc/"
FILE_LIST = "test.txt"
# cd /home/li/data/pascalvoc/./VOCdevkit/VOC2007/JPEGImages/ pass

label_file = "label_list"
RESIZE_H = 300
RESIZE_W = 300
mean_value = [127.5, 127.5, 127.5]
ap_version = '11point'
IMAGE_OUT = 'pascalvoc.bin'

DATA_DIR = os.path.expanduser(DATA_DIR)


def preprocess(img):
    img_width, img_height = img.size

    img = img.resize((RESIZE_W, RESIZE_H), Image.ANTIALIAS)
    img = np.array(img)

    # HWC to CHW
    if len(img.shape) == 3:
        img = np.swapaxes(img, 1, 2)
        img = np.swapaxes(img, 1, 0)
    # RBG to BGR
    img = img[[2, 1, 0], :, :]
    img = img.astype('float32')
    img_mean = np.array(mean_value)[:, np.newaxis, np.newaxis].astype('float32')
    img -= img_mean
    img = img * 0.007843
    return img


def pascalvoc():
    label_list = []
    label_fpath = os.path.join(DATA_DIR, label_file)
    for line in open(label_fpath):
        label_list.append(line.strip())

    file_list_path = os.path.join(DATA_DIR, FILE_LIST)
    flist = open(file_list_path)
    lines = [line.strip() for line in flist]

    image_out_path = os.path.join(DATA_DIR, IMAGE_OUT)
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
        image_path = os.path.join(DATA_DIR, image_path)
        label_path = os.path.join(DATA_DIR, label_path)

        im = Image.open(image_path)
        if im.mode == 'L':
            im = im.convert('RGB')
        im_width, im_height = im.size

        im = preprocess(im)
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

    # Until here, image size is 5348160008
    # num of lods
    f1.write(np.array(object_nums).astype('uint64').tobytes())
    # num of labels
    f1.write(np.array(lbls).astype('int64').tobytes())
    print(np.array(lbls).astype('int64'))
    f1.write(np.array(boxes).astype('float32').tobytes())

    f1.write(np.array(difficults).astype('int64').tobytes())

    f1.close()


if __name__ == "__main__":
    pascalvoc()
