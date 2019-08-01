#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import math
import numpy as np
import unittest
from py_precise_roi_pool import PyPrRoIPool
from op_test import OpTest


class TestPRROIPoolOp(OpTest):
    def set_data(self):
        self.init_test_case()
        self.make_rois()
        self.prRoIPool = PyPrRoIPool()
        self.outs = self.prRoIPool.calculate(
            self.x, self.rois, self.output_channels, self.spatial_scale,
            self.pooled_height, self.pooled_width).astype('float32')
        self.inputs = {'X': self.x, 'ROIs': (self.rois[:, 1:5], self.rois_lod)}
        self.attrs = {
            'output_channels': self.output_channels,
            'spatial_scale': self.spatial_scale,
            'pooled_height': self.pooled_height,
            'pooled_width': self.pooled_width
        }
        self.outputs = {'Out': self.outs}

    def init_test_case(self):
        self.batch_size = 3
        self.channels = 3 * 2 * 2
        self.height = 6
        self.width = 4

        self.x_dim = [self.batch_size, self.channels, self.height, self.width]

        self.spatial_scale = 1.0 / 4.0
        self.output_channels = 3
        self.pooled_height = 2
        self.pooled_width = 2

        self.x = np.random.random(self.x_dim).astype('float32')

    def make_rois(self):
        rois = []
        self.rois_lod = [[]]
        for bno in range(self.batch_size):
            self.rois_lod[0].append(bno + 1)
            for i in range(bno + 1):
                x1 = np.random.random_integers(
                    0, self.width // self.spatial_scale - self.pooled_width)
                y1 = np.random.random_integers(
                    0, self.height // self.spatial_scale - self.pooled_height)

                x2 = np.random.random_integers(x1 + self.pooled_width,
                                               self.width // self.spatial_scale)
                y2 = np.random.random_integers(
                    y1 + self.pooled_height, self.height // self.spatial_scale)
                roi = [bno, x1, y1, x2, y2]
                rois.append(roi)
        self.rois_num = len(rois)
        self.rois = np.array(rois).astype('float32')

    def setUp(self):
        self.op_type = 'prroi_pool'
        self.set_data()

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
