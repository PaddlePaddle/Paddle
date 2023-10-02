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
import sys
import unittest
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
from op_test import OpTest

import paddle


def _round(x):
    """In Python3 round function rounds to the nearest even number,
    we use this function to make the result always round up when the
    remainder is 0.5. See more at:
    https://stackoverflow.com/questions/33019698/how-to-properly-round-up-half-float-numbers
    """
    return Decimal(x).to_integral_value(rounding=ROUND_HALF_UP)


class TestROIPoolOp(OpTest):
    def set_data(self):
        self.init_test_case()
        self.make_rois()
        self.calc_roi_pool()

        self.inputs = {
            'X': self.x,
            'ROIs': (self.rois[:, 1:5], self.rois_lod),
            'RoisNum': self.boxes_num,
        }

        self.attrs = {
            'spatial_scale': self.spatial_scale,
            'pooled_height': self.pooled_height,
            'pooled_width': self.pooled_width,
        }

        self.outputs = {'Out': self.outs, 'Argmax': self.argmaxes}

    def init_test_case(self):
        self.batch_size = 3
        self.channels = 3
        self.height = 6
        self.width = 4

        # n, c, h, w
        self.x_dim = (self.batch_size, self.channels, self.height, self.width)

        self.spatial_scale = 1.0 / 4.0
        self.pooled_height = 2
        self.pooled_width = 2

        self.x = np.random.random(self.x_dim).astype('float64')

    def calc_roi_pool(self):
        out_data = np.zeros(
            (
                self.rois_num,
                self.channels,
                self.pooled_height,
                self.pooled_width,
            )
        )
        argmax_data = np.zeros(
            (
                self.rois_num,
                self.channels,
                self.pooled_height,
                self.pooled_width,
            )
        )

        for i in range(self.rois_num):
            roi = self.rois[i]
            roi_batch_id = int(roi[0])
            roi_start_w = int(_round(roi[1] * self.spatial_scale))
            roi_start_h = int(_round(roi[2] * self.spatial_scale))
            roi_end_w = int(_round(roi[3] * self.spatial_scale))
            roi_end_h = int(_round(roi[4] * self.spatial_scale))

            roi_height = int(max(roi_end_h - roi_start_h + 1, 1))
            roi_width = int(max(roi_end_w - roi_start_w + 1, 1))

            x_i = self.x[roi_batch_id]

            bin_size_h = float(roi_height) / float(self.pooled_height)
            bin_size_w = float(roi_width) / float(self.pooled_width)

            for c in range(self.channels):
                for ph in range(self.pooled_height):
                    for pw in range(self.pooled_width):
                        hstart = int(math.floor(ph * bin_size_h))
                        wstart = int(math.floor(pw * bin_size_w))
                        hend = int(math.ceil((ph + 1) * bin_size_h))
                        wend = int(math.ceil((pw + 1) * bin_size_w))

                        hstart = min(max(hstart + roi_start_h, 0), self.height)
                        hend = min(max(hend + roi_start_h, 0), self.height)
                        wstart = min(max(wstart + roi_start_w, 0), self.width)
                        wend = min(max(wend + roi_start_w, 0), self.width)

                        is_empty = (hend <= hstart) or (wend <= wstart)
                        if is_empty:
                            out_data[i, c, ph, pw] = 0
                        else:
                            out_data[i, c, ph, pw] = -sys.float_info.max

                        argmax_data[i, c, ph, pw] = -1

                        for h in range(hstart, hend):
                            for w in range(wstart, wend):
                                if x_i[c, h, w] > out_data[i, c, ph, pw]:
                                    out_data[i, c, ph, pw] = x_i[c, h, w]
                                    argmax_data[i, c, ph, pw] = (
                                        h * self.width + w
                                    )

        self.outs = out_data.astype('float64')
        self.argmaxes = argmax_data.astype('int64')

    def make_rois(self):
        rois = []
        self.rois_lod = [[]]
        for bno in range(self.batch_size):
            self.rois_lod[0].append(bno + 1)
            for i in range(bno + 1):
                x1 = np.random.randint(
                    0, self.width // self.spatial_scale - self.pooled_width
                )
                y1 = np.random.randint(
                    0, self.height // self.spatial_scale - self.pooled_height
                )

                x2 = np.random.randint(
                    x1 + self.pooled_width, self.width // self.spatial_scale
                )
                y2 = np.random.randint(
                    y1 + self.pooled_height, self.height // self.spatial_scale
                )

                roi = [bno, x1, y1, x2, y2]
                rois.append(roi)
        self.rois_num = len(rois)
        self.rois = np.array(rois).astype("float64")
        self.boxes_num = np.array(
            [bno + 1 for bno in range(self.batch_size)]
        ).astype('int32')

    def setUp(self):
        self.op_type = "roi_pool"
        self.python_api = lambda x, boxes, boxes_num, pooled_height, pooled_width, spatial_scale: paddle.vision.ops.roi_pool(
            x, boxes, boxes_num, (pooled_height, pooled_width), spatial_scale
        )
        self.python_out_sig = ["Out"]
        self.set_data()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestROIPoolInLodOp(TestROIPoolOp):
    def set_data(self):
        self.init_test_case()
        self.make_rois()
        self.calc_roi_pool()

        seq_len = self.rois_lod[0]

        self.inputs = {
            'X': self.x,
            'ROIs': (self.rois[:, 1:5], self.rois_lod),
            'RoisNum': np.asarray(seq_len).astype('int32'),
        }

        self.attrs = {
            'spatial_scale': self.spatial_scale,
            'pooled_height': self.pooled_height,
            'pooled_width': self.pooled_width,
        }

        self.outputs = {'Out': self.outs, 'Argmax': self.argmaxes}


if __name__ == '__main__':
    unittest.main()
