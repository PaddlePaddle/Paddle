#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://w_idxw.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
import sys
import math
from op_test import OpTest

global_var = []

class TestAnchorGeneratorOp(OpTest):
    def set_data(self):
        self.init_test_params()
        self.init_test_input()
        self.init_test_output()
        self.inputs = {'Input': self.input, 'Image': self.image}

        self.attrs = {
            'anchor_sizes': self.anchor_sizes,
            'aspect_ratios': self.aspect_ratios,
            'stride': self.stride,
            'offset': self.offset,
            'variances': self.variances,
        }

        self.outputs = {'Anchors': self.out_anchors, 'Variances': self.out_var}

    def test_check_output(self):
        global global_var
        global_var = self.outputs
        self.check_output()
        print(global_var['Anchors'][0][0])

    def setUp(self):
        self.op_type = "anchor_generator"
        self.set_data()

    def init_test_params(self):
        self.layer_w = 4
        self.layer_h = 4

        self.stride = [16, 16]

        self.image_w = 16 * 4
        self.image_h = 16 * 4

        self.input_channels = 2
        self.image_channels = 3
        self.batch_size = 1

        self.anchor_sizes = [64, 128, 256, 512]
        self.aspect_ratios = [0.5, 1, 2]

        self.num_anchors = len(self.aspect_ratios) * len(self.anchor_sizes)
        self.offset = 0.5

        self.variances = [0.1, 0.1, 0.2, 0.2]

    def init_test_input(self):
        self.image = np.random.random(
            (self.batch_size, self.image_channels, self.image_w,
             self.image_h)).astype('float32')

        self.input = np.random.random(
            (self.batch_size, self.input_channels, self.layer_w,
             self.layer_h)).astype('float32')

    def init_test_output(self):
        out_dim = (self.layer_h, self.layer_w, self.num_priors, 4)
        out_anchors = np.zeros(out_dim).astype('float32')

        for h_idx in range(self.layer_h):
            for w_idx in range(self.layer_w):
                x_ctr = (w_idx * self.stride[0]) + offset * (self.stride[0] - 1)
                y_ctr = (h_idx * self.stride[1]) + offset * (self.stride[1] - 1)
                idx = 0
                for r in range(len(self.aspect_ratios)):
                    ar = self.aspect_ratios[r]
                    for s in range(len(self.anchor_sizes)):
                        anchor_size = self.anchor_sizes[s]
                        area = self.stride[0] * self.stride[1]
                        area_ratios = area / ar
                        base_w = np.round(np.sqrt(area_ratios))
                        base_h = np.round(base_w * ar)
                        scale_w = anchor_size / self.stride[0]
                        scale_h = anchor_size / self.stride[1]
                        w = scale_w * base_w
                        h = scale_h * base_h
                        out_anchors[h_idx, w_idx, idx, :] = [(x_ctr - 0.5 * (w - 1)),
                                                             (y_ctr - 0.5 * (h - 1)),
                                                             (x_ctr + 0.5 * (w - 1)),
                                                             (y_ctr + 0.5 * (h - 1))]
                        idx += 1

        # set the variance.
        out_var = np.tile(self.variances, (self.layer_h, self.layer_w,
                                           self.num_priors, 1))
        self.out_anchors = out_anchors.astype('float32')
        self.out_var = out_var.astype('float32')


if __name__ == '__main__':
    unittest.main()
