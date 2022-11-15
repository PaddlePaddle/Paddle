# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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


class PyPrRoIPool:
    def __init__(self):
        pass

    def _PrRoIPoolingGetData(self, data, h, w, height, width):
        overflow = (h < 0) or (w < 0) or (h >= height) or (w >= width)
        if overflow:
            return 0.0
        else:
            return data[h][w]

    def _PrRoIPoolingMatCalculation(
        self, this_data, s_h, s_w, e_h, e_w, y0, x0, y1, x1, h0, w0
    ):
        sum_out = 0.0
        alpha = x0 - float(s_w)
        beta = y0 - float(s_h)
        lim_alpha = x1 - float(s_w)
        lim_beta = y1 - float(s_h)
        tmp = (
            lim_alpha
            - 0.5 * lim_alpha * lim_alpha
            - alpha
            + 0.5 * alpha * alpha
        ) * (lim_beta - 0.5 * lim_beta * lim_beta - beta + 0.5 * beta * beta)
        sum_out += self._PrRoIPoolingGetData(this_data, s_h, s_w, h0, w0) * tmp

        alpha = float(e_w) - x1
        lim_alpha = float(e_w) - x0
        tmp = (
            lim_alpha
            - 0.5 * lim_alpha * lim_alpha
            - alpha
            + 0.5 * alpha * alpha
        ) * (lim_beta - 0.5 * lim_beta * lim_beta - beta + 0.5 * beta * beta)
        sum_out += self._PrRoIPoolingGetData(this_data, s_h, e_w, h0, w0) * tmp

        alpha = x0 - float(s_w)
        beta = float(e_h) - y1
        lim_alpha = x1 - float(s_w)
        lim_beta = float(e_h) - y0
        tmp = (
            lim_alpha
            - 0.5 * lim_alpha * lim_alpha
            - alpha
            + 0.5 * alpha * alpha
        ) * (lim_beta - 0.5 * lim_beta * lim_beta - beta + 0.5 * beta * beta)
        sum_out += self._PrRoIPoolingGetData(this_data, e_h, s_w, h0, w0) * tmp

        alpha = float(e_w) - x1
        lim_alpha = float(e_w) - x0
        tmp = (
            lim_alpha
            - 0.5 * lim_alpha * lim_alpha
            - alpha
            + 0.5 * alpha * alpha
        ) * (lim_beta - 0.5 * lim_beta * lim_beta - beta + 0.5 * beta * beta)
        sum_out += self._PrRoIPoolingGetData(this_data, e_h, e_w, h0, w0) * tmp

        return sum_out

    def compute(
        self,
        x,
        rois,
        output_channels,
        spatial_scale=0.1,
        pooled_height=1,
        pooled_width=1,
    ):
        '''
        calculate the precise roi pooling values
        Note: This function is implements as pure python without any paddle concept involved
        :param x (array): array[N, C, H, W]
        :param rois (array): ROIs[id, x1, y1, x2, y2] (Regions of Interest) to pool over.
        :param output_channels (Integer): Expected output channels
        :param spatial_scale (float): spatial scale, default = 0.1
        :param pooled_height (Integer): Expected output height, default = 1
        :param pooled_width (Integer): Expected output width, default = 1
        :return: array[len(rois), output_channels, pooled_height, pooled_width]
        '''
        if not isinstance(output_channels, int):
            raise TypeError("output_channels must be int type")
        if not isinstance(spatial_scale, float):
            raise TypeError("spatial_scale must be float type")
        if not isinstance(pooled_height, int):
            raise TypeError("pooled_height must be int type")
        if not isinstance(pooled_width, int):
            raise TypeError("pooled_width must be int type")

        (batch_size, channels, height, width) = np.array(x).shape
        rois_num = len(rois)
        output_shape = (rois_num, output_channels, pooled_height, pooled_width)
        out_data = np.zeros(output_shape)
        for i in range(rois_num):
            roi = rois[i]
            roi_batch_id = int(roi[0])
            roi_start_w = roi[1] * spatial_scale
            roi_start_h = roi[2] * spatial_scale
            roi_end_w = roi[3] * spatial_scale
            roi_end_h = roi[4] * spatial_scale

            roi_width = max(roi_end_w - roi_start_w, 0.0)
            roi_height = max(roi_end_h - roi_start_h, 0.0)
            bin_size_h = roi_height / float(pooled_height)
            bin_size_w = roi_width / float(pooled_width)

            x_i = x[roi_batch_id]

            for c in range(output_channels):
                for ph in range(pooled_height):
                    for pw in range(pooled_width):
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

                            c_in = c
                            for w_iter in range(int(s_w), int(e_w)):
                                for h_iter in range(int(s_h), int(e_h)):
                                    sum_out += self._PrRoIPoolingMatCalculation(
                                        x_i[c_in],
                                        h_iter,
                                        w_iter,
                                        h_iter + 1,
                                        w_iter + 1,
                                        max(win_start_h, float(h_iter)),
                                        max(win_start_w, float(w_iter)),
                                        min(win_end_h, float(h_iter) + 1.0),
                                        min(win_end_w, float(w_iter + 1.0)),
                                        height,
                                        width,
                                    )

                            out_data[i, c, ph, pw] = sum_out / win_size

        return out_data
