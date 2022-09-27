# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np
import sys
import math
from op_test import OpTest


class TestDensityPriorBoxOp(OpTest):

    def set_data(self):
        self.init_test_params()
        self.init_test_input()
        self.init_test_output()
        self.inputs = {'Input': self.input, 'Image': self.image}

        self.attrs = {
            'variances': self.variances,
            'clip': self.clip,
            'step_w': self.step_w,
            'step_h': self.step_h,
            'offset': self.offset,
            'densities': self.densities,
            'fixed_sizes': self.fixed_sizes,
            'fixed_ratios': self.fixed_ratios,
            'flatten_to_2d': self.flatten_to_2d
        }
        self.outputs = {'Boxes': self.out_boxes, 'Variances': self.out_var}

    def test_check_output(self):
        self.check_output()

    def setUp(self):
        self.op_type = "density_prior_box"
        self.set_data()

    def set_density(self):
        self.densities = [4, 2, 1]
        self.fixed_sizes = [32.0, 64.0, 128.0]
        self.fixed_ratios = [1.0]
        self.layer_w = 17
        self.layer_h = 17
        self.image_w = 533
        self.image_h = 533
        self.flatten_to_2d = False

    def init_test_params(self):
        self.set_density()

        self.step_w = float(self.image_w) / float(self.layer_w)
        self.step_h = float(self.image_h) / float(self.layer_h)

        self.input_channels = 2
        self.image_channels = 3
        self.batch_size = 10

        self.variances = [0.1, 0.1, 0.2, 0.2]
        self.variances = np.array(self.variances, dtype=np.float64).flatten()

        self.clip = True
        self.num_priors = 0
        if len(self.fixed_sizes) > 0 and len(self.densities) > 0:
            for density in self.densities:
                if len(self.fixed_ratios) > 0:
                    self.num_priors += len(self.fixed_ratios) * (pow(
                        density, 2))
        self.offset = 0.5

    def init_test_input(self):
        self.image = np.random.random(
            (self.batch_size, self.image_channels, self.image_w,
             self.image_h)).astype('float32')

        self.input = np.random.random(
            (self.batch_size, self.input_channels, self.layer_w,
             self.layer_h)).astype('float32')

    def init_test_output(self):
        out_dim = (self.layer_h, self.layer_w, self.num_priors, 4)
        out_boxes = np.zeros(out_dim).astype('float32')
        out_var = np.zeros(out_dim).astype('float32')

        step_average = int((self.step_w + self.step_h) * 0.5)
        for h in range(self.layer_h):
            for w in range(self.layer_w):
                idx = 0
                c_x = (w + self.offset) * self.step_w
                c_y = (h + self.offset) * self.step_h
                # Generate density prior boxes with fixed size
                for density, fixed_size in zip(self.densities,
                                               self.fixed_sizes):
                    if (len(self.fixed_ratios) > 0):
                        for ar in self.fixed_ratios:
                            shift = int(step_average / density)
                            box_width_ratio = fixed_size * math.sqrt(ar)
                            box_height_ratio = fixed_size / math.sqrt(ar)
                            for di in range(density):
                                for dj in range(density):
                                    c_x_temp = c_x - step_average / 2.0 + shift / 2.0 + dj * shift
                                    c_y_temp = c_y - step_average / 2.0 + shift / 2.0 + di * shift
                                    out_boxes[h, w, idx, :] = [
                                        max((c_x_temp - box_width_ratio / 2.0) /
                                            self.image_w, 0),
                                        max((c_y_temp - box_height_ratio / 2.0)
                                            / self.image_h, 0),
                                        min((c_x_temp + box_width_ratio / 2.0) /
                                            self.image_w, 1),
                                        min((c_y_temp + box_height_ratio / 2.0)
                                            / self.image_h, 1)
                                    ]
                                    idx += 1
        if self.clip:
            out_boxes = np.clip(out_boxes, 0.0, 1.0)
        out_var = np.tile(self.variances,
                          (self.layer_h, self.layer_w, self.num_priors, 1))
        self.out_boxes = out_boxes.astype('float32')
        self.out_var = out_var.astype('float32')
        if self.flatten_to_2d:
            self.out_boxes = self.out_boxes.reshape((-1, 4))
            self.out_var = self.out_var.reshape((-1, 4))


class TestDensityPriorBox(TestDensityPriorBoxOp):

    def set_density(self):
        self.densities = [3, 4]
        self.fixed_sizes = [1.0, 2.0]
        self.fixed_ratios = [1.0]
        self.layer_w = 32
        self.layer_h = 32
        self.image_w = 40
        self.image_h = 40
        self.flatten_to_2d = True


if __name__ == '__main__':
    unittest.main()
