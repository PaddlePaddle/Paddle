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


class TestPRROIPoolOp(OpTest):
    def set_data(self):
        self.init_test_case()
        self.make_rois()
        self.calc_prroi_pool()
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

    def PrRoIPoolingGetData(self, data, h, w, height, width):
        overflow = (h < 0) or (w < 0) or (h >= height) or (w >= width)
        if overflow:
            return 0.0
        else:
            return data[h][w]

    def PrRoIPoolingGetCoeff(self, dh, dw):
        dw = abs(dw)
        dh = abs(dh)
        return (1.0 - dh) * (1.0 - dw)

    def PrRoIPoolingSingleCoorIntegral(self, s, t, c1, c2):
        return 0.5 * (t * t - s * s) * c2 + (t - 0.5 * t * t - s + 0.5 * s * s
                                             ) * c1

    def PrRoIPoolingInterpolation(self, data, h, w, height, width):
        retVal = 0.0
        math.floorh1 = math.floor(h)
        math.floorw1 = math.floor(w)
        retVal += self.PrRoIPoolingGetData(data, h1, w1, height,
                                           width) * self.PrRoIPoolingGetCoeff(
                                               h - float(h1), w - float(w1))
        h1 = math.floor(h) + 1
        w1 = math.floor(w)
        retVal += self.PrRoIPoolingGetData(data, h1, w1, height,
                                           width) * self.PrRoIPoolingGetCoeff(
                                               h - float(h1), w - float(w1))
        h1 = math.floor(h)
        w1 = math.floor(w) + 1
        retVal += self.PrRoIPoolingGetData(data, h1, w1, height,
                                           width) * self.PrRoIPoolingGetCoeff(
                                               h - float(h1), w - float(w1))
        h1 = math.floor(h) + 1
        w1 = math.floor(w) + 1
        retVal += self.PrRoIPoolingGetData(data, h1, w1, height,
                                           width) * self.PrRoIPoolingGetCoeff(
                                               h - float(h1), w - float(w1))
        return retVal

    def PrRoIPoolingMatCalculation(self, this_data, s_h, s_w, e_h, e_w, y0, x0,
                                   y1, x1, h0, w0):
        sum_out = 0.0
        alpha = x0 - float(s_w)
        beta = y0 - float(s_h)
        lim_alpha = x1 - float(s_w)
        lim_beta = y1 - float(s_h)
        tmp = (
            lim_alpha - 0.5 * lim_alpha * lim_alpha - alpha + 0.5 * alpha *
            alpha) * (
                lim_beta - 0.5 * lim_beta * lim_beta - beta + 0.5 * beta * beta)
        sum_out += self.PrRoIPoolingGetData(this_data, s_h, s_w, h0, w0) * tmp

        alpha = float(e_w) - x1
        lim_alpha = float(e_w) - x0
        tmp = (
            lim_alpha - 0.5 * lim_alpha * lim_alpha - alpha + 0.5 * alpha *
            alpha) * (
                lim_beta - 0.5 * lim_beta * lim_beta - beta + 0.5 * beta * beta)
        sum_out += self.PrRoIPoolingGetData(this_data, s_h, e_w, h0, w0) * tmp

        alpha = x0 - float(s_w)
        beta = float(e_h) - y1
        lim_alpha = x1 - float(s_w)
        lim_beta = float(e_h) - y0
        tmp = (
            lim_alpha - 0.5 * lim_alpha * lim_alpha - alpha + 0.5 * alpha *
            alpha) * (
                lim_beta - 0.5 * lim_beta * lim_beta - beta + 0.5 * beta * beta)
        sum_out += self.PrRoIPoolingGetData(this_data, e_h, s_w, h0, w0) * tmp

        alpha = float(e_w) - x1
        lim_alpha = float(e_w) - x0
        tmp = (
            lim_alpha - 0.5 * lim_alpha * lim_alpha - alpha + 0.5 * alpha *
            alpha) * (
                lim_beta - 0.5 * lim_beta * lim_beta - beta + 0.5 * beta * beta)
        sum_out += self.PrRoIPoolingGetData(this_data, e_h, e_w, h0, w0) * tmp

        return sum_out

    def calc_prroi_pool(self):
        output_shape = (self.rois_num, self.output_channels, self.pooled_height,
                        self.pooled_width)
        out_data = np.zeros(output_shape)
        for i in range(self.rois_num):
            roi = self.rois[i]
            roi_batch_id = int(roi[0])
            roi_start_w = roi[1] * self.spatial_scale
            roi_start_h = roi[2] * self.spatial_scale
            roi_end_w = roi[3] * self.spatial_scale
            roi_end_h = roi[4] * self.spatial_scale

            roi_width = max(roi_end_w - roi_start_w, 0.0)
            roi_height = max(roi_end_h - roi_start_h, 0.0)
            bin_size_h = roi_height / float(self.pooled_height)
            bin_size_w = roi_width / float(self.pooled_width)

            x_i = self.x[roi_batch_id]

            for c in range(self.output_channels):
                for ph in range(self.pooled_height):
                    for pw in range(self.pooled_width):
                        win_start_w = roi_start_w + bin_size_w * pw
                        win_start_h = roi_start_h + bin_size_h * ph
                        win_end_w = win_start_w + bin_size_w
                        win_end_h = win_start_h + bin_size_h

                        win_size = max(0.0, bin_size_w * bin_size_h)
                        if win_size == 0.0:
                            out_data[i, c, ph, pw] = 0.0
                        else:
                            sum_out = 0

                            s_w = math.floor(win_start_w)
                            e_w = math.ceil(win_end_w)
                            s_h = math.floor(win_start_h)
                            e_h = math.ceil(win_end_h)

                            c_in = (c * self.pooled_height + ph
                                    ) * self.pooled_width + pw

                            for w_iter in range(int(s_w), int(e_w)):
                                for h_iter in range(int(s_h), int(e_h)):
                                    sum_out += self.PrRoIPoolingMatCalculation(
                                        x_i[c], h_iter, w_iter, h_iter + 1,
                                        w_iter + 1,
                                        max(win_start_h, float(h_iter)),
                                        max(win_start_w, float(w_iter)),
                                        min(win_end_h, float(h_iter) + 1.0),
                                        min(win_end_w, float(w_iter + 1.0)),
                                        self.height, self.width)

                            out_data[i, c, ph, pw] = sum_out / win_size

        self.outs = out_data.astype('float32')

    def setUp(self):
        self.op_type = 'prroi_pool'
        self.set_data()

    def test_check_output(self):
        self.check_output()

    # def test_check_grad(self):
    #     self.check_grad(['X'], 'Out')


if __name__ == '__main__':
    unittest.main()
