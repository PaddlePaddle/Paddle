#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
            'min_sizes': self.min_sizes,
            'aspect_ratios': self.aspect_ratios,
            'variances': self.variances,
            'flip': self.flip,
            'clip': self.clip,
            'min_max_aspect_ratios_order': self.min_max_aspect_ratios_order,
            'step_w': self.step_w,
            'step_h': self.step_h,
            'offset': self.offset,
            'densities': self.densities,
            'fixed_sizes': self.fixed_sizes,
            'fixed_ratios': self.fixed_ratios
        }
        if len(self.max_sizes) > 0:
            self.attrs['max_sizes'] = self.max_sizes

        self.outputs = {'Boxes': self.out_boxes, 'Variances': self.out_var}

    def test_check_output(self):
        self.check_output()

    def setUp(self):
        self.op_type = "density_prior_box"
        self.set_data()

    def set_max_sizes(self):
        max_sizes = [5, 10]
        self.max_sizes = np.array(max_sizes).astype('float32').tolist()

    def set_min_max_aspect_ratios_order(self):
        self.min_max_aspect_ratios_order = False

    def set_density(self):
        self.densities = []
        self.fixed_sizes = []
        self.fixed_ratios = []

    def init_test_params(self):
        self.layer_w = 32
        self.layer_h = 32

        self.image_w = 40
        self.image_h = 40

        self.step_w = float(self.image_w) / float(self.layer_w)
        self.step_h = float(self.image_h) / float(self.layer_h)

        self.input_channels = 2
        self.image_channels = 3
        self.batch_size = 10

        self.min_sizes = [2, 4]
        self.min_sizes = np.array(self.min_sizes).astype('float32').tolist()
        self.set_max_sizes()
        self.aspect_ratios = [2.0, 3.0]
        self.flip = True
        self.set_min_max_aspect_ratios_order()
        self.real_aspect_ratios = [1, 2.0, 1.0 / 2.0, 3.0, 1.0 / 3.0]
        self.aspect_ratios = np.array(
            self.aspect_ratios, dtype=np.float).flatten()
        self.variances = [0.1, 0.1, 0.2, 0.2]
        self.variances = np.array(self.variances, dtype=np.float).flatten()

        self.set_density()

        self.clip = True

        self.num_priors = len(self.real_aspect_ratios) * len(self.min_sizes)
        if len(self.fixed_sizes) > 0:
            if len(self.densities) > 0:
                for density in self.densities:
                    if len(self.fixed_ratios) > 0:
                        self.num_priors += len(self.fixed_ratios) * (pow(
                            density, 2))
                    else:
                        self.num_priors += len(self.real_aspect_ratios) * (pow(
                            density, 2))
        if len(self.max_sizes) > 0:
            self.num_priors += len(self.max_sizes)
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
                for density, fixed_size in zip(self.densities,
                                               self.fixed_sizes):
                    box_width = box_height = fixed_size
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
                    else:
                        shift = int(fixed_size / density)
                        for di in range(density):
                            for dj in range(density):
                                c_x_temp = c_x - fixed_size / 2.0 + shift / 2.0 + dj * shift
                                c_y_temp = c_y - fixed_size / 2.0 + shift / 2.0 + di * shift
                                out_boxes[h, w, idx, :] = [
                                    max((c_x_temp - box_width / 2.0) /
                                        self.image_w, 0),
                                    max((c_y_temp - box_height / 2.0) /
                                        self.image_h, 0), min(
                                            (c_x_temp + box_width / 2.0) /
                                            self.image_w, 1), min(
                                                (c_y_temp + box_height / 2.0) /
                                                self.image_h, 1)
                                ]
                                idx += 1
                        for ar in self.real_aspect_ratios:
                            if (abs(ar - 1.) < 1e-6):
                                continue
                            shift = int(fixed_size / density)
                            box_width_ratio = fixed_size * math.sqrt(ar)
                            box_height_ratio = fixed_size / math.sqrt(ar)
                            for di in range(density):
                                for dj in range(density):
                                    c_x_temp = c_x - fixed_size / 2.0 + shift / 2.0 + dj * shift
                                    c_y_temp = c_y - fixed_size / 2.0 + shift / 2.0 + di * shift
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

                for s in range(len(self.min_sizes)):
                    min_size = self.min_sizes[s]
                    if not self.min_max_aspect_ratios_order:
                        for r in range(len(self.real_aspect_ratios)):
                            ar = self.real_aspect_ratios[r]
                            c_w = min_size * math.sqrt(ar) / 2.
                            c_h = (min_size / math.sqrt(ar)) / 2.
                            out_boxes[h, w, idx, :] = [
                                (c_x - c_w) / self.image_w, (c_y - c_h) /
                                self.image_h, (c_x + c_w) / self.image_w,
                                (c_y + c_h) / self.image_h
                            ]
                            idx += 1

                        if len(self.max_sizes) > 0:
                            max_size = self.max_sizes[s]
                            c_w = c_h = math.sqrt(min_size * max_size) / 2.
                            out_boxes[h, w, idx, :] = [
                                (c_x - c_w) / self.image_w, (c_y - c_h) /
                                self.image_h, (c_x + c_w) / self.image_w,
                                (c_y + c_h) / self.image_h
                            ]
                            idx += 1
                    else:
                        c_w = c_h = min_size / 2.
                        out_boxes[h, w, idx, :] = [(c_x - c_w) / self.image_w,
                                                   (c_y - c_h) / self.image_h,
                                                   (c_x + c_w) / self.image_w,
                                                   (c_y + c_h) / self.image_h]
                        idx += 1
                        if len(self.max_sizes) > 0:
                            max_size = self.max_sizes[s]
                            c_w = c_h = math.sqrt(min_size * max_size) / 2.
                            out_boxes[h, w, idx, :] = [
                                (c_x - c_w) / self.image_w, (c_y - c_h) /
                                self.image_h, (c_x + c_w) / self.image_w,
                                (c_y + c_h) / self.image_h
                            ]
                            idx += 1
                        for r in range(len(self.real_aspect_ratios)):
                            ar = self.real_aspect_ratios[r]
                            if abs(ar - 1.) < 1e-6:
                                continue
                            c_w = min_size * math.sqrt(ar) / 2.
                            c_h = (min_size / math.sqrt(ar)) / 2.
                            out_boxes[h, w, idx, :] = [
                                (c_x - c_w) / self.image_w, (c_y - c_h) /
                                self.image_h, (c_x + c_w) / self.image_w,
                                (c_y + c_h) / self.image_h
                            ]
                            idx += 1
        if self.clip:
            out_boxes = np.clip(out_boxes, 0.0, 1.0)
        out_var = np.tile(self.variances, (self.layer_h, self.layer_w,
                                           self.num_priors, 1))
        self.out_boxes = out_boxes.astype('float32')
        self.out_var = out_var.astype('float32')


class TestPriorBoxOpWithoutMaxSize(TestDensityPriorBoxOp):
    def set_max_sizes(self):
        self.max_sizes = []


class TestPriorBoxOpWithSpecifiedOutOrder(TestDensityPriorBoxOp):
    def set_min_max_aspect_ratios_order(self):
        self.min_max_aspect_ratios_order = True


class TestDensityPriorBoxWithDensityBox(TestDensityPriorBoxOp):
    def set_density(self):
        self.densities = [3, 4]
        self.fixed_sizes = [1.0, 2.0]
        self.fixed_ratios = [1.0 / 2.0, 1.0, 2.0]


class TestDensityPriorBoxWithDensityBoxWithoutFixedRatios(
        TestDensityPriorBoxOp):
    def set_density(self):
        self.densities = [3, 4]
        self.fixed_sizes = [1.0, 2.0]
        self.fixed_ratios = []


if __name__ == '__main__':
    unittest.main()
