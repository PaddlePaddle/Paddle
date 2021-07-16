#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
import math
import random
import paddle
import paddle.fluid.core as core
from op_test import OpTest
from paddle.fluid import Program, program_guard


def class_center_sample_numpy(label, classes_list, num_sample):
    unique_label = np.unique(label)
    nranks = len(classes_list)
    class_interval = np.cumsum(np.insert(classes_list, 0, 0))
    pos_class_center_per_device = []
    unique_label_per_device = []

    for i in range(nranks):
        index = np.logical_and(unique_label >= class_interval[i],
                               unique_label < class_interval[i + 1])
        pos_class_center_per_device.append(unique_label[index] - class_interval[
            i])
        unique_label_per_device.append(unique_label[index])

    num_sample_per_device = []
    for pos_class_center in pos_class_center_per_device:
        num_sample_per_device.append(max(len(pos_class_center), num_sample))
    sampled_class_interval = np.cumsum(np.insert(num_sample_per_device, 0, 0))

    remapped_dict = {}
    for i in range(nranks):
        for idx, v in enumerate(unique_label_per_device[i],
                                sampled_class_interval[i]):
            remapped_dict[v] = idx

    remapped_label = []
    for l in label:
        remapped_label.append(remapped_dict[l])

    return np.array(remapped_label), np.array(pos_class_center_per_device)


class TestClassCenterSampleOp(OpTest):
    def initParams(self):
        self.op_type = "class_center_sample"
        self.batch_size = 20
        self.num_sample = 6
        self.num_classes = 10
        self.seed = 2021

    def init_dtype(self):
        self.dtype = np.int64

    def init_fix_seed(self):
        self.fix_seed = False

    def setUp(self):
        self.initParams()
        self.init_dtype()
        self.init_fix_seed()
        label = np.random.randint(
            0, self.num_classes, (self.batch_size, ), dtype=self.dtype)

        remapped_label, sampled_class_center = class_center_sample_numpy(
            label, [self.num_classes], self.num_sample)

        self.inputs = {'Label': label}
        self.outputs = {
            'RemappedLabel': remapped_label.astype(self.dtype),
            'SampledLocalClassCenter': sampled_class_center.astype(self.dtype)
        }

        self.attrs = {
            'num_classes': self.num_classes,
            'num_sample': self.num_sample,
            'seed': self.seed,
            'fix_seed': self.fix_seed,
        }

    def test_check_output(self):
        self.check_output(no_check_set=['SampledLocalClassCenter'])


class TestClassCenterSampleOpINT32(TestClassCenterSampleOp):
    def init_dtype(self):
        self.dtype = np.int32


class TestClassCenterSampleOpFixSeed(TestClassCenterSampleOp):
    def init_fix_seed(self):
        self.fix_seed = True


if __name__ == '__main__':
    unittest.main()
