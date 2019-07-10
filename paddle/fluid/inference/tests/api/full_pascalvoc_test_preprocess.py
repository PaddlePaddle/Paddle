# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import os
import sys
from paddle.dataset.common import download
import tarfile
import StringIO
import hashlib
import tarfile

DATA_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
DATA_DIR = os.path.expanduser("~/.cache/paddle/dataset/pascalvoc/")
TAR_FILE = "VOCtest_06-Nov-2007.tar"
TAR_PATH = os.path.join(DATA_DIR, TAR_FILE)
RESIZE_H = 300
RESIZE_W = 300
mean_value = [127.5, 127.5, 127.5]
ap_version = '11point'
DATA_OUT = 'pascalvoc_full.bin'
DATA_OUT_PATH = os.path.join(DATA_DIR, DATA_OUT)
BIN_TARGETHASH = "f6546cadc42f5ff13178b84ed29b740b"
TAR_TARGETHASH = "b6e924de25625d8de591ea690078ad9f"
TEST_LIST_KEY = "VOCdevkit/VOC2007/ImageSets/Main/test.txt"
BIN_FULLSIZE = 5348678856


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


def print_processbar(done_percentage):
    done_filled = done_percentage * '='
    empty_filled = (100 - done_percentage) * ' '
    sys.stdout.write("\r[%s%s]%d%%" %
                     (done_filled, empty_filled, done_percentage))
    sys.stdout.flush()


def convert_pascalvoc(tar_path, data_out_path):
    print("Start converting ...\n")
    images = {}
    gt_labels = {}
    boxes = []
    lbls = []
    difficults = []
    object_nums = []

    # map label to number (index)
    label_list = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
        "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
        "tvmonitor"
    ]
    print_processbar(0)
    #read from tar file and write to bin
    tar = tarfile.open(tar_path, "r")
    f_test = tar.extractfile(TEST_LIST_KEY).read()
    lines = f_test.split('\n')
    del lines[-1]
    line_len = len(lines)
    per_percentage = line_len / 100

    f1 = open(data_out_path, "w+b")
    f1.seek(0)
    f1.write(np.array(line_len).astype('int64').tobytes())
    for tarInfo in tar:
        if tarInfo.isfile():
            tmp_filename = tarInfo.name
            name_arr = tmp_filename.split('/')
            name_prefix = name_arr[-1].split('.')[0]
            if name_arr[-2] == 'JPEGImages' and name_prefix in lines:
                images[name_prefix] = tar.extractfile(tarInfo).read()
            if name_arr[-2] == 'Annotations' and name_prefix in lines:
                gt_labels[name_prefix] = tar.extractfile(tarInfo).read()

    for line_idx, name_prefix in enumerate(lines):
        im = Image.open(StringIO.StringIO(images[name_prefix]))
        if im.mode == 'L':
            im = im.convert('RGB')
        im_width, im_height = im.size

        im = preprocess(im)
        np_im = np.array(im)
        f1.write(np_im.astype('float32').tobytes())

        # layout: label | xmin | ymin | xmax | ymax | difficult
        bbox_labels = []
        root = ET.fromstring(gt_labels[name_prefix])

        objects = root.findall('object')
        objects_size = len(objects)
        object_nums.append(objects_size)

        for object in objects:
            bbox_sample = []
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

        if line_idx % per_percentage:
            print_processbar(line_idx / per_percentage)

    f1.write(np.array(object_nums).astype('uint64').tobytes())
    f1.write(np.array(lbls).astype('int64').tobytes())
    f1.write(np.array(boxes).astype('float32').tobytes())
    f1.write(np.array(difficults).astype('int64').tobytes())
    f1.close()
    print_processbar(100)
    print("Conversion finished!\n")


def download_pascalvoc(data_url, data_dir, tar_targethash, tar_path):
    print("Downloading pascalvcoc test set...")
    download(data_url, data_dir, tar_targethash)
    if not os.path.exists(tar_path):
        print("Failed in downloading pascalvoc test set. URL %s\n" % data_url)
    else:
        tmp_hash = hashlib.md5(open(tar_path, 'rb').read()).hexdigest()
        if tmp_hash != tar_targethash:
            print("Downloaded test set is broken, removing ...\n")
        else:
            print("Downloaded successfully. Path: %s\n" % tar_path)


def run_convert():
    try_limit = 2
    retry = 0
    while not (os.path.exists(DATA_OUT_PATH) and
               os.path.getsize(DATA_OUT_PATH) == BIN_FULLSIZE and BIN_TARGETHASH
               == hashlib.md5(open(DATA_OUT_PATH, 'rb').read()).hexdigest()):
        if os.path.exists(DATA_OUT_PATH):
            sys.stderr.write(
                "The existing binary file is broken. It is being removed...\n")
            os.remove(DATA_OUT_PATH)
        if retry < try_limit:
            retry = retry + 1
        else:
            download_pascalvoc(DATA_URL, DATA_DIR, TAR_TARGETHASH, TAR_PATH)
            convert_pascalvoc(TAR_PATH, DATA_OUT_PATH)
    print("Success! \nThe binary file can be found at %s\n" % DATA_OUT_PATH)


if __name__ == "__main__":
    run_convert()
