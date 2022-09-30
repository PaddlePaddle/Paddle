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


def anchor_generator_in_python(input_feat, anchor_sizes, aspect_ratios,
                               variances, stride, offset):
    num_anchors = len(aspect_ratios) * len(anchor_sizes)
    layer_h = input_feat.shape[2]
    layer_w = input_feat.shape[3]
    out_dim = (layer_h, layer_w, num_anchors, 4)
    out_anchors = np.zeros(out_dim).astype('float32')

    for h_idx in range(layer_h):
        for w_idx in range(layer_w):
            x_ctr = (w_idx * stride[0]) + offset * (stride[0] - 1)
            y_ctr = (h_idx * stride[1]) + offset * (stride[1] - 1)
            idx = 0
            for r in range(len(aspect_ratios)):
                ar = aspect_ratios[r]
                for s in range(len(anchor_sizes)):
                    anchor_size = anchor_sizes[s]
                    area = stride[0] * stride[1]
                    area_ratios = area / ar
                    base_w = np.round(np.sqrt(area_ratios))
                    base_h = np.round(base_w * ar)
                    scale_w = anchor_size / stride[0]
                    scale_h = anchor_size / stride[1]
                    w = scale_w * base_w
                    h = scale_h * base_h
                    out_anchors[h_idx, w_idx,
                                idx, :] = [(x_ctr - 0.5 * (w - 1)),
                                           (y_ctr - 0.5 * (h - 1)),
                                           (x_ctr + 0.5 * (w - 1)),
                                           (y_ctr + 0.5 * (h - 1))]
                    idx += 1

    # set the variance.
    out_var = np.tile(variances, (layer_h, layer_w, num_anchors, 1))
    out_anchors = out_anchors.astype('float32')
    out_var = out_var.astype('float32')
    return out_anchors, out_var


class TestAnchorGeneratorOp(OpTest):

    def set_data(self):
        self.init_test_params()
        self.init_test_input()
        self.init_test_output()
        self.inputs = {'Input': self.input}

        self.attrs = {
            'anchor_sizes': self.anchor_sizes,
            'aspect_ratios': self.aspect_ratios,
            'stride': self.stride,
            'offset': self.offset,
            'variances': self.variances,
        }

        self.outputs = {'Anchors': self.out_anchors, 'Variances': self.out_var}

    def test_check_output(self):
        self.check_output()

    def setUp(self):
        self.op_type = "anchor_generator"
        self.set_data()

    def init_test_params(self):
        self.batch_size = 1
        self.input_channels = 2
        self.layer_h = 2
        self.layer_w = 2

        self.anchor_sizes = [64., 128., 256., 512.]
        self.aspect_ratios = [0.5, 1., 2.]
        self.stride = [16., 16.]

        self.offset = 0.5

        self.variances = [0.1, 0.1, 0.2, 0.2]

    def init_test_input(self):
        self.input = np.random.random(
            (self.batch_size, self.input_channels, self.layer_h,
             self.layer_w)).astype('float32')

    def init_test_output(self):
        self.out_anchors, self.out_var = anchor_generator_in_python(
            self.input, self.anchor_sizes, self.aspect_ratios, self.variances,
            self.stride, self.offset)


if __name__ == '__main__':
    unittest.main()
