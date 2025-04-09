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

import argparse
import hashlib
import os
import sys
import tarfile
import xml.etree.ElementTree
from io import StringIO

import numpy as np
from PIL import Image

from paddle.dataset.common import download

DATA_URL = (
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
)
DATA_DIR = os.path.expanduser("~/.cache/paddle/dataset/pascalvoc/")
TAR_FILE = "VOCtest_06-Nov-2007.tar"
TAR_PATH = os.path.join(DATA_DIR, TAR_FILE)
SIZE_FLOAT32 = 4
SIZE_INT64 = 8
RESIZE_H = 300
RESIZE_W = 300
MEAN_VALUE = [127.5, 127.5, 127.5]
AP_VERSION = '11point'
DATA_OUT = 'pascalvoc_full.bin'
DATA_OUT_PATH = os.path.join(DATA_DIR, DATA_OUT)
BIN_TARGETHASH = "f6546cadc42f5ff13178b84ed29b740b"
TAR_TARGETHASH = "b6e924de25625d8de591ea690078ad9f"
TEST_LIST_KEY = "VOCdevkit/VOC2007/ImageSets/Main/test.txt"
BIN_FULLSIZE = 5348678856


def preprocess(img):
    img_width, img_height = img.size
    img = img.resize((RESIZE_W, RESIZE_H), Image.LANCZOS)
    img = np.array(img)
    # HWC to CHW
    if len(img.shape) == 3:
        img = np.swapaxes(img, 1, 2)
        img = np.swapaxes(img, 1, 0)
    # RBG to BGR
    img = img[[2, 1, 0], :, :]
    img = img.astype('float32')
    img_mean = np.array(MEAN_VALUE)[:, np.newaxis, np.newaxis].astype('float32')
    img -= img_mean
    img = img * 0.007843
    return img


def convert_pascalvoc_local2bin(args):
    data_dir = os.path.expanduser(args.data_dir)
    label_fpath = os.path.join(data_dir, args.label_file)
    assert data_dir, 'Once set --local, user need to provide the --data_dir'
    flabel = open(label_fpath)
    label_list = [line.strip() for line in flabel]

    img_annotation_list_path = os.path.join(data_dir, args.img_annotation_list)
    flist = open(img_annotation_list_path)
    lines = [line.strip() for line in flist]

    output_file_path = os.path.join(data_dir, args.output_file)
    f1 = open(output_file_path, "w+b")
    f1.seek(0)
    image_nums = len(lines)
    f1.write(np.array(image_nums).astype('int64').tobytes())

    boxes = []
    lbls = []
    difficulties = []
    object_nums = []

    for line in lines:
        image_path, label_path = line.split()
        image_path = os.path.join(data_dir, image_path)
        label_path = os.path.join(data_dir, label_path)

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
                float(label_list.index(object.find('name').text))
            )
            bbox = object.find('bndbox')
            difficult = float(object.find('difficult').text)
            bbox_sample.append(float(bbox.find('xmin').text) / im_width)
            bbox_sample.append(float(bbox.find('ymin').text) / im_height)
            bbox_sample.append(float(bbox.find('xmax').text) / im_width)
            bbox_sample.append(float(bbox.find('ymax').text) / im_height)
            bbox_sample.append(difficult)
            bbox_labels.append(bbox_sample)

        bbox_labels = np.array(bbox_labels)
        if len(bbox_labels) == 0:
            continue

        lbls.extend(bbox_labels[:, 0])
        boxes.extend(bbox_labels[:, 1:5])
        difficulties.extend(bbox_labels[:, -1])

    f1.write(np.array(object_nums).astype('uint64').tobytes())
    f1.write(np.array(lbls).astype('int64').tobytes())
    f1.write(np.array(boxes).astype('float32').tobytes())
    f1.write(np.array(difficulties).astype('int64').tobytes())
    f1.close()

    object_nums_sum = sum(object_nums)
    # The data should be contains
    # number of images + all images data + an array that represent object numbers of each image
    # + labels of all objects in images + bboxes of all objects + difficulties of all objects
    # so the target size should be as follows:
    target_size = (
        SIZE_INT64
        + image_nums * 3 * args.resize_h * args.resize_h * SIZE_FLOAT32
        + image_nums * SIZE_INT64
        + object_nums_sum * (SIZE_INT64 + 4 * SIZE_FLOAT32 + SIZE_INT64)
    )
    if os.path.getsize(output_file_path) == target_size:
        print(
            "Success! \nThe local data output binary file can be found at: ",
            output_file_path,
        )
    else:
        print("Conversion failed!")


def print_processbar(done_percentage):
    done_filled = done_percentage * '='
    empty_filled = (100 - done_percentage) * ' '
    sys.stdout.write(f"\r[{done_filled}{empty_filled}]{done_percentage}%")
    sys.stdout.flush()


def convert_pascalvoc_tar2bin(tar_path, data_out_path):
    print("Start converting ...\n")
    images = {}
    gt_labels = {}
    boxes = []
    lbls = []
    difficulties = []
    object_nums = []

    # map label to number (index)
    label_list = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]
    print_processbar(0)
    # read from tar file and write to bin
    tar = tarfile.open(tar_path, "r")
    f_test = tar.extractfile(TEST_LIST_KEY).read()
    lines = f_test.split('\n')
    del lines[-1]
    image_nums = len(lines)
    per_percentage = image_nums / 100

    f1 = open(data_out_path, "w+b")
    f1.seek(0)
    f1.write(np.array(image_nums).astype('int64').tobytes())
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
        im = Image.open(StringIO(images[name_prefix]))
        if im.mode == 'L':
            im = im.convert('RGB')
        im_width, im_height = im.size

        im = preprocess(im)
        np_im = np.array(im)
        f1.write(np_im.astype('float32').tobytes())

        # layout: label | xmin | ymin | xmax | ymax | difficult
        bbox_labels = []
        root = xml.etree.ElementTree.fromstring(gt_labels[name_prefix])

        objects = root.findall('object')
        objects_size = len(objects)
        object_nums.append(objects_size)

        for object in objects:
            bbox_sample = []
            bbox_sample.append(
                float(label_list.index(object.find('name').text))
            )
            bbox = object.find('bndbox')
            difficult = float(object.find('difficult').text)
            bbox_sample.append(float(bbox.find('xmin').text) / im_width)
            bbox_sample.append(float(bbox.find('ymin').text) / im_height)
            bbox_sample.append(float(bbox.find('xmax').text) / im_width)
            bbox_sample.append(float(bbox.find('ymax').text) / im_height)
            bbox_sample.append(difficult)
            bbox_labels.append(bbox_sample)

        bbox_labels = np.array(bbox_labels)
        if len(bbox_labels) == 0:
            continue
        lbls.extend(bbox_labels[:, 0])
        boxes.extend(bbox_labels[:, 1:5])
        difficulties.extend(bbox_labels[:, -1])

        if line_idx % per_percentage:
            print_processbar(line_idx / per_percentage)

    # The data should be stored in binary in following sequence:
    # number of images->all images data->an array that represent object numbers in each image
    # ->labels of all objects in images->bboxes of all objects->difficulties of all objects
    f1.write(np.array(object_nums).astype('uint64').tobytes())
    f1.write(np.array(lbls).astype('int64').tobytes())
    f1.write(np.array(boxes).astype('float32').tobytes())
    f1.write(np.array(difficulties).astype('int64').tobytes())
    f1.close()
    print_processbar(100)
    print("Conversion finished!\n")


def download_pascalvoc(data_url, data_dir, tar_targethash, tar_path):
    print("Downloading pascalvcoc test set...")
    download(data_url, data_dir, tar_targethash)
    if not os.path.exists(tar_path):
        print(f"Failed in downloading pascalvoc test set. URL {data_url}\n")
    else:
        tmp_hash = hashlib.md5(open(tar_path, 'rb').read()).hexdigest()
        if tmp_hash != tar_targethash:
            print("Downloaded test set is broken, removing ...\n")
        else:
            print(f"Downloaded successfully. Path: {tar_path}\n")


def run_convert():
    try_limit = 2
    retry = 0
    while not (
        os.path.exists(DATA_OUT_PATH)
        and os.path.getsize(DATA_OUT_PATH) == BIN_FULLSIZE
        and BIN_TARGETHASH
        == hashlib.md5(open(DATA_OUT_PATH, 'rb').read()).hexdigest()
    ):
        if os.path.exists(DATA_OUT_PATH):
            sys.stderr.write(
                "The existing binary file is broken. It is being removed...\n"
            )
            os.remove(DATA_OUT_PATH)
        if retry < try_limit:
            retry = retry + 1
        else:
            download_pascalvoc(DATA_URL, DATA_DIR, TAR_TARGETHASH, TAR_PATH)
            convert_pascalvoc_tar2bin(TAR_PATH, DATA_OUT_PATH)
    print(f"Success!\nThe binary file can be found at {DATA_OUT_PATH}\n")


def main_pascalvoc_preprocess(args):
    parser = argparse.ArgumentParser(
        description="Convert the full pascalvoc val set or local data to binary file.",
        usage=None,
        add_help=True,
    )
    parser.add_argument(
        '--local',
        action="store_true",
        help="If used, user need to set --data_dir and then convert file",
    )
    parser.add_argument(
        "--data_dir", default="", type=str, help="Dataset root directory"
    )
    parser.add_argument(
        "--img_annotation_list",
        type=str,
        default="test_100.txt",
        help="A file containing the image file path and corresponding annotation file path",
    )
    parser.add_argument(
        "--label_file",
        type=str,
        default="label_list",
        help="List of object labels with same sequence as denoted in the annotation file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="pascalvoc_small.bin",
        help="File path of the output binary file",
    )
    parser.add_argument(
        "--resize_h",
        type=int,
        default=RESIZE_H,
        help="Image preprocess with resize_h",
    )
    parser.add_argument(
        "--resize_w",
        type=int,
        default=RESIZE_W,
        help="Image prerocess with resize_w",
    )
    parser.add_argument(
        "--mean_value",
        type=str,
        default=MEAN_VALUE,
        help="Image preprocess with mean_value",
    )
    parser.add_argument(
        "--ap_version",
        type=str,
        default=AP_VERSION,
        help="Image preprocess with ap_version",
    )
    args = parser.parse_args()
    if args.local:
        convert_pascalvoc_local2bin(args)
    else:
        run_convert()


if __name__ == "__main__":
    main_pascalvoc_preprocess(sys.argv)
