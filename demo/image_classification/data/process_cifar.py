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

import numpy as np
import sys
import os
import PIL.Image as Image
"""
  Usage: python process_cifar input_dir output_dir
"""


def mkdir_not_exist(path):
    """
    Make dir if the path does not exist.
    path: the path to be created.
    """
    if not os.path.exists(path):
        os.mkdir(path)


def create_dir_structure(output_dir):
    """
    Create the directory structure for the directory.
    output_dir: the direcotry structure path.
    """
    mkdir_not_exist(os.path.join(output_dir))
    mkdir_not_exist(os.path.join(output_dir, "train"))
    mkdir_not_exist(os.path.join(output_dir, "test"))


def convert_batch(batch_path, label_set, label_map, output_dir, data_split):
    """
    Convert CIFAR batch to the structure of Paddle format.
    batch_path: the batch to be converted.
    label_set: the set of labels.
    output_dir: the output path.
    data_split: whether it is training or testing data.
    """
    data = np.load(batch_path)
    for data, label, filename in zip(data['data'], data['labels'],
                                     data['filenames']):
        data = data.reshape((3, 32, 32))
        data = np.transpose(data, (1, 2, 0))
        label = label_map[label]
        output_dir_this = os.path.join(output_dir, data_split, str(label))
        output_filename = os.path.join(output_dir_this, filename)
        if not label in label_set:
            label_set[label] = True
            mkdir_not_exist(output_dir_this)
        Image.fromarray(data).save(output_filename)


if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    num_batch = 5
    create_dir_structure(output_dir)
    label_map = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
    }
    labels = {}
    for i in range(1, num_batch + 1):
        convert_batch(
            os.path.join(input_dir, "data_batch_%d" % i), labels, label_map,
            output_dir, "train")
    convert_batch(
        os.path.join(input_dir, "test_batch"), {}, label_map, output_dir,
        "test")
