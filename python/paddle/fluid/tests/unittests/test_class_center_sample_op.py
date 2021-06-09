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
import paddle
from op_test import OpTest
from paddle.distributed.fleet.utils.plsc_util import class_center_sample
from paddle.static import program_guard, Program


def common_setup(self, label, num_class, ratio=0.1, ignore_label=-1):
    self.op_type = 'class_center_sample'
    seed = 123
    np.random.seed(seed)

    unique_label = np.unique(label).tolist()
    if ignore_label in unique_label:
        unique_label.remove(ignore_label)
    num_sample = int(num_class * ratio)
    while len(unique_label) < num_sample:
        l = np.random.randint(0, num_class)
        if l not in unique_label:
            unique_label.append(l)

    unique_label.sort()
    new_label_dict = {}
    for idx, l in enumerate(unique_label):
        new_label_dict[l] = idx

    remaped_label = []
    for l in label:
        if l == ignore_label:
            remaped_label.append(l)
        else:
            remaped_label.append(new_label_dict[l])

    self.inputs = {'X': np.array(label)}
    self.attrs = {
        'num_class': num_class,
        'ratio': ratio,
        'ignore_label': ignore_label,
        'seed': seed,
    }
    self.outputs = {
        'Out': np.array(remaped_label),
        'SampledClass': np.array(unique_label)
    }


class TestClassCenterSampleOp(OpTest):
    def setUp(self):
        label = [-1, 0, -1, -1]
        num_class = 5
        ratio = 1.0
        ignore_label = -1
        common_setup(self, label, num_class, ratio, ignore_label)

    def test_check_output(self):
        self.check_output(check_dygraph=False)


if __name__ == '__main__':
    unittest.main()
