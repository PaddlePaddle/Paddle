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

import math
import numpy as np
import sys
import unittest

sys.path.append("..")

import paddle

from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


class XPUTestPriorBoxOp(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'prior_box'
        self.use_dynamic_create_class = False

    class TestPriorBoxOp(XPUOpTest):

        def setUp(self):
            self.op_type = "prior_box"
            self.use_xpu = True
            self.dtype = self.in_type
            self.set_data()

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
                'offset': self.offset
            }
            if len(self.max_sizes) > 0:
                self.attrs['max_sizes'] = self.max_sizes

            self.outputs = {'Boxes': self.out_boxes, 'Variances': self.out_var}

        def test_check_output(self):
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

        def set_max_sizes(self):
            max_sizes = [5, 10]
            self.max_sizes = np.array(max_sizes).astype('float32').tolist()

        def set_min_max_aspect_ratios_order(self):
            self.min_max_aspect_ratios_order = False

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
            self.aspect_ratios = np.array(self.aspect_ratios,
                                          dtype=np.float64).flatten()
            self.variances = [0.1, 0.1, 0.2, 0.2]
            self.variances = np.array(self.variances,
                                      dtype=np.float64).flatten()

            self.clip = True
            self.num_priors = len(self.real_aspect_ratios) * len(self.min_sizes)
            if len(self.max_sizes) > 0:
                self.num_priors += len(self.max_sizes)
            self.offset = 0.5

        def init_test_input(self):
            self.image = np.random.random(
                (self.batch_size, self.image_channels, self.image_w,
                 self.image_h)).astype(self.dtype)

            self.input = np.random.random(
                (self.batch_size, self.input_channels, self.layer_w,
                 self.layer_h)).astype(self.dtype)

        def init_test_output(self):
            out_dim = (self.layer_h, self.layer_w, self.num_priors, 4)
            out_boxes = np.zeros(out_dim).astype(self.dtype)
            out_var = np.zeros(out_dim).astype(self.dtype)

            idx = 0
            for h in range(self.layer_h):
                for w in range(self.layer_w):
                    c_x = (w + self.offset) * self.step_w
                    c_y = (h + self.offset) * self.step_h
                    idx = 0
                    for s in range(len(self.min_sizes)):
                        min_size = self.min_sizes[s]
                        if not self.min_max_aspect_ratios_order:
                            # rest of priors
                            for r in range(len(self.real_aspect_ratios)):
                                ar = self.real_aspect_ratios[r]
                                c_w = min_size * math.sqrt(ar) / 2
                                c_h = (min_size / math.sqrt(ar)) / 2
                                out_boxes[h, w,
                                          idx, :] = [(c_x - c_w) / self.image_w,
                                                     (c_y - c_h) / self.image_h,
                                                     (c_x + c_w) / self.image_w,
                                                     (c_y + c_h) / self.image_h]
                                idx += 1

                            if len(self.max_sizes) > 0:
                                max_size = self.max_sizes[s]
                                # second prior: aspect_ratio = 1,
                                c_w = c_h = math.sqrt(min_size * max_size) / 2
                                out_boxes[h, w,
                                          idx, :] = [(c_x - c_w) / self.image_w,
                                                     (c_y - c_h) / self.image_h,
                                                     (c_x + c_w) / self.image_w,
                                                     (c_y + c_h) / self.image_h]
                                idx += 1
                        else:
                            c_w = c_h = min_size / 2.
                            out_boxes[h, w,
                                      idx, :] = [(c_x - c_w) / self.image_w,
                                                 (c_y - c_h) / self.image_h,
                                                 (c_x + c_w) / self.image_w,
                                                 (c_y + c_h) / self.image_h]
                            idx += 1
                            if len(self.max_sizes) > 0:
                                max_size = self.max_sizes[s]
                                # second prior: aspect_ratio = 1,
                                c_w = c_h = math.sqrt(min_size * max_size) / 2
                                out_boxes[h, w,
                                          idx, :] = [(c_x - c_w) / self.image_w,
                                                     (c_y - c_h) / self.image_h,
                                                     (c_x + c_w) / self.image_w,
                                                     (c_y + c_h) / self.image_h]
                                idx += 1

                            # rest of priors
                            for r in range(len(self.real_aspect_ratios)):
                                ar = self.real_aspect_ratios[r]
                                if abs(ar - 1.) < 1e-6:
                                    continue
                                c_w = min_size * math.sqrt(ar) / 2
                                c_h = (min_size / math.sqrt(ar)) / 2
                                out_boxes[h, w,
                                          idx, :] = [(c_x - c_w) / self.image_w,
                                                     (c_y - c_h) / self.image_h,
                                                     (c_x + c_w) / self.image_w,
                                                     (c_y + c_h) / self.image_h]
                                idx += 1

            # clip the prior's coordidate such that it is within[0, 1]
            if self.clip:
                out_boxes = np.clip(out_boxes, 0.0, 1.0)
            # set the variance.
            out_var = np.tile(self.variances,
                              (self.layer_h, self.layer_w, self.num_priors, 1))
            self.out_boxes = out_boxes.astype(self.dtype)
            self.out_var = out_var.astype(self.dtype)

    class TestPriorBoxOpWithoutMaxSize(TestPriorBoxOp):

        def set_max_sizes(self):
            self.max_sizes = []

    class TestPriorBoxOpWithSpecifiedOutOrder(TestPriorBoxOp):

        def set_min_max_aspect_ratios_order(self):
            self.min_max_aspect_ratios_order = True


support_types = get_xpu_op_support_types('prior_box')
for stype in support_types:
    create_test_class(globals(), XPUTestPriorBoxOp, stype)

if __name__ == '__main__':
    unittest.main()
