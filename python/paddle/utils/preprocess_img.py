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

import sys
import os
import random
import numpy as np
import PIL.Image as Image
from six.moves import cStringIO as StringIO
from . import preprocess_util
from .image_util import crop_img


def resize_image(img, target_size):
    """
    Resize an image so that the shorter edge has length target_size.
    img: the input image to be resized.
    target_size: the target resized image size.
    """
    percent = (target_size / float(min(img.size[0], img.size[1])))
    resized_size = int(round(img.size[0] * percent)),\
                   int(round(img.size[1] * percent))
    img = img.resize(resized_size, Image.ANTIALIAS)
    return img


class DiskImage:
    """
    A class of image data on disk.
    """

    def __init__(self, path, target_size):
        """
        path: path of the image.
        target_size: target resize size.
        """
        self.path = path
        self.target_size = target_size
        self.img = None
        pass

    def read_image(self):
        if self.img is None:
            print("reading: " + self.path)
            image = resize_image(Image.open(self.path), self.target_size)
            self.img = image

    def convert_to_array(self):
        self.read_image()
        np_array = np.array(self.img)
        if len(np_array.shape) == 3:
            np_array = np.swapaxes(np_array, 1, 2)
            np_array = np.swapaxes(np_array, 1, 0)
        return np_array

    def convert_to_paddle_format(self):
        """
        convert the image into the paddle batch format.
        """
        self.read_image()
        output = StringIO()
        self.img.save(output, "jpeg")
        contents = output.getvalue()
        return contents


class ImageClassificationDatasetCreater(preprocess_util.DatasetCreater):
    """
    A class to process data for image classification.
    """

    def __init__(self, data_path, target_size, color=True):
        """
        data_path: the path to store the training data and batches.
        target_size: processed image size in a batch.
        color: whether to use color images.
        """
        preprocess_util.DatasetCreater.__init__(self, data_path)
        self.target_size = target_size
        self.color = color
        self.keys = ["images", "labels"]
        self.permute_key = "labels"

    def create_meta_file(self, data):
        """
        Create a meta file for image classification.
        The meta file contains the meam image, as well as some configs.
        data: the training Dataaet.
        """
        output_path = os.path.join(self.data_path, self.batch_dir_name,
                                   self.meta_filename)
        if self.color:
            mean_img = np.zeros((3, self.target_size, self.target_size))
        else:
            mean_img = np.zeros((self.target_size, self.target_size))
        for d in data.data:
            img = d[0].convert_to_array()
            cropped_img = crop_img(img, self.target_size, self.color)
            mean_img += cropped_img
        mean_img /= len(data.data)
        mean_img = mean_img.astype('int32').flatten()
        preprocess_util.save_file({
            "data_mean": mean_img,
            "image_size": self.target_size,
            "mean_image_size": self.target_size,
            "num_classes": self.num_classes,
            "color": self.color
        }, output_path)
        pass

    def create_dataset_from_list(self, path):
        data = []
        label_set = []
        for line in open(path):
            items = line.rstrip.split()
            image_path = items[0]
            label_name = items[1]
            if not label_name in label_set:
                label_set[label_name] = len(list(label_set.keys()))
            img = DiskImage(path=image_path, target_size=self.target_size)
            label = preprocess_util.Lablel(
                label=label_set[label_name], name=label_name)
        return preprocess_util.Dataset(data, self.keys), label_set

    def create_dataset_from_dir(self, path):
        """
        Create a Dataset object for image classification.
        Each folder in the path directory corresponds to a set of images of
        this label, and the name of the folder is the name of the
        path: the path of the image dataset.
        """
        if self.from_list:
            return self.create_dataset_from_list(path)
        label_set = preprocess_util.get_label_set_from_dir(path)
        data = []
        for l_name in list(label_set.keys()):
            image_paths = preprocess_util.list_images(
                os.path.join(path, l_name))
            for p in image_paths:
                img = DiskImage(path=p, target_size=self.target_size)
                label = preprocess_util.Label(
                    label=label_set[l_name], name=l_name)
                data.append((img, label))
        random.shuffle(data)
        return preprocess_util.Dataset(data, self.keys), label_set
