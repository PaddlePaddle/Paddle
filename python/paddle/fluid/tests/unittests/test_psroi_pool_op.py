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

import math
import numpy as np
import unittest
from op_test import OpTest


class TestPSROIPoolOp(OpTest):
    def set_data(self):
        self.init_test_case()
        self.make_rois()
        self.calc_psroi_pool()
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

        self.x = np.random.random(self.x_dim).astype('float64')

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
        self.rois = np.array(rois).astype('float64')

    def calc_psroi_pool(self):
        output_shape = (self.rois_num, self.output_channels, self.pooled_height,
                        self.pooled_width)
        out_data = np.zeros(output_shape)
        for i in range(self.rois_num):
            roi = self.rois[i]
            roi_batch_id = int(roi[0])
            roi_start_w = round(roi[1]) * self.spatial_scale
            roi_start_h = round(roi[2]) * self.spatial_scale
            roi_end_w = (round(roi[3]) + 1.) * self.spatial_scale
            roi_end_h = (round(roi[4]) + 1.) * self.spatial_scale

            roi_height = max(roi_end_h - roi_start_h, 0.1)
            roi_width = max(roi_end_w - roi_start_w, 0.1)

            bin_size_h = roi_height / float(self.pooled_height)
            bin_size_w = roi_width / float(self.pooled_width)

            x_i = self.x[roi_batch_id]

            for c in range(self.output_channels):
                for ph in range(self.pooled_height):
                    for pw in range(self.pooled_width):
                        hstart = int(
                            math.floor(float(ph) * bin_size_h + roi_start_h))
                        wstart = int(
                            math.floor(float(pw) * bin_size_w + roi_start_w))
                        hend = int(
                            math.ceil(
                                float(ph + 1) * bin_size_h + roi_start_h))
                        wend = int(
                            math.ceil(
                                float(pw + 1) * bin_size_w + roi_start_w))
                        hstart = min(max(hstart, 0), self.height)
                        hend = min(max(hend, 0), self.height)
                        wstart = min(max(wstart, 0), self.width)
                        wend = min(max(wend, 0), self.width)

                        c_in = (c * self.pooled_height + ph
                                ) * self.pooled_width + pw
                        is_empty = (hend <= hstart) or (wend <= wstart)
                        out_sum = 0.
                        for ih in range(hstart, hend):
                            for iw in range(wstart, wend):
                                out_sum += x_i[c_in, ih, iw]
                        bin_area = (hend - hstart) * (wend - wstart)
                        out_data[i, c, ph, pw] = 0. if is_empty else (
                            out_sum / float(bin_area))
        self.outs = out_data.astype('float64')

    def setUp(self):
        self.op_type = 'psroi_pool'
        self.set_data()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


if __name__ == '__main__':
    unittest.main()
