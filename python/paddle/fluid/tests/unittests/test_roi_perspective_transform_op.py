#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUWARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
import math
import sys
import paddle.compat as cpt
from op_test import OpTest
from math import sqrt
from math import floor
from paddle import fluid


def gt_e(a, b):
    return a > b or abs(a - b) < 1e-4


def gt(a, b):
    return (a - b) > 1e-4


def lt_e(a, b):
    return a < b or abs(a - b) < 1e-4


def in_quad(x, y, roi_x, roi_y):
    # check if (x, y) is in the boundary of roi
    for i in range(4):
        xs = roi_x[i]
        ys = roi_y[i]
        xe = roi_x[(i + 1) % 4]
        ye = roi_y[(i + 1) % 4]
        if abs(ys - ye) < 1e-4:
            if abs(y - ys) < 1e-4 and abs(y - ye) < 1e-4 and gt_e(
                    x, min(xs, xe)) and lt_e(x, max(xs, xe)):
                return True
        else:
            intersec_x = (y - ys) * (xe - xs) / (ye - ys) + xs
            if abs(intersec_x - x) < 1e-4 and gt_e(y, min(ys, ye)) and lt_e(
                    y, max(ys, ye)):
                return True
    n_cross = 0
    for i in range(4):
        xs = roi_x[i]
        ys = roi_y[i]
        xe = roi_x[(i + 1) % 4]
        ye = roi_y[(i + 1) % 4]
        if abs(ys - ye) < 1e-4:
            continue
        if lt_e(y, min(ys, ye)) or gt(y, max(ys, ye)):
            continue
        intersec_x = (y - ys) * (xe - xs) / (ye - ys) + xs
        if abs(intersec_x - x) < 1e-4:
            return True
        if gt(intersec_x, x):
            n_cross += 1
    return (n_cross % 2 == 1)


def get_transform_matrix(transformed_width, transformed_height, roi_x, roi_y):
    x0 = roi_x[0]
    x1 = roi_x[1]
    x2 = roi_x[2]
    x3 = roi_x[3]
    y0 = roi_y[0]
    y1 = roi_y[1]
    y2 = roi_y[2]
    y3 = roi_y[3]

    len1 = sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1))
    len2 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
    len3 = sqrt((x2 - x3) * (x2 - x3) + (y2 - y3) * (y2 - y3))
    len4 = sqrt((x3 - x0) * (x3 - x0) + (y3 - y0) * (y3 - y0))
    estimated_height = (len2 + len4) / 2.0
    estimated_width = (len1 + len3) / 2.0

    normalized_height = max(2, transformed_height)
    normalized_width = round(estimated_width *
                             (normalized_height - 1) / estimated_height) + 1
    normalized_width = max(2, min(normalized_width, transformed_width))

    dx1 = x1 - x2
    dx2 = x3 - x2
    dx3 = x0 - x1 + x2 - x3
    dy1 = y1 - y2
    dy2 = y3 - y2
    dy3 = y0 - y1 + y2 - y3
    matrix = np.zeros([9])
    matrix[6] = (dx3 * dy2 - dx2 * dy3) / (dx1 * dy2 - dx2 * dy1 +
                                           1e-5) / (normalized_width - 1)
    matrix[7] = (dx1 * dy3 - dx3 * dy1) / (dx1 * dy2 - dx2 * dy1 +
                                           1e-5) / (normalized_height - 1)
    matrix[8] = 1

    matrix[3] = (y1 - y0 + matrix[6] *
                 (normalized_width - 1) * y1) / (normalized_width - 1)
    matrix[4] = (y3 - y0 + matrix[7] *
                 (normalized_height - 1) * y3) / (normalized_height - 1)
    matrix[5] = y0

    matrix[0] = (x1 - x0 + matrix[6] *
                 (normalized_width - 1) * x1) / (normalized_width - 1)
    matrix[1] = (x3 - x0 + matrix[7] *
                 (normalized_height - 1) * x3) / (normalized_height - 1)
    matrix[2] = x0
    return matrix


def get_source_coords(matrix, out_w, out_h):
    u = matrix[0] * out_w + matrix[1] * out_h + matrix[2]
    v = matrix[3] * out_w + matrix[4] * out_h + matrix[5]
    w = matrix[6] * out_w + matrix[7] * out_h + matrix[8]
    in_w = u / w
    in_h = v / w
    return in_w, in_h


def bilinear_interpolate(in_data, in_n, in_c, in_w, in_h):

    batch_size = in_data.shape[0]
    channels = in_data.shape[1]
    height = in_data.shape[2]
    width = in_data.shape[3]

    if gt_e(-0.5, in_w) or gt_e(in_w, width - 0.5) or gt_e(-0.5, in_h) or gt_e(
            in_h, height - 0.5):
        return 0.0

    if gt_e(0, in_w):
        in_w = 0
    if gt_e(0, in_h):
        in_h = 0

    in_w_floor = floor(in_w)
    in_h_floor = floor(in_h)

    if gt_e(in_w_floor, width - 1):
        in_w_ceil = width - 1
        in_w_floor = width - 1
        in_w = in_w_floor
    else:
        in_w_ceil = in_w_floor + 1

    if gt_e(in_h_floor, height - 1):
        in_h_ceil = height - 1
        in_h_floor = height - 1
        in_h = in_h_floor
    else:
        in_h_ceil = in_h_floor + 1

    w_floor = in_w - in_w_floor
    h_floor = in_h - in_h_floor
    w_ceil = 1 - w_floor
    h_ceil = 1 - h_floor
    v1 = in_data[in_n][in_c][int(in_h_floor)][int(in_w_floor)]
    v2 = in_data[in_n][in_c][int(in_h_ceil)][int(in_w_floor)]
    v3 = in_data[in_n][in_c][int(in_h_ceil)][int(in_w_ceil)]
    v4 = in_data[in_n][in_c][int(in_h_floor)][int(in_w_ceil)]
    w1 = w_ceil * h_ceil
    w2 = w_ceil * h_floor
    w3 = w_floor * h_floor
    w4 = w_floor * h_ceil
    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    return val


def lod_convert(lod):
    ret = [0]
    for count in lod:
        ret.append(ret[-1] + count)
    return ret


def roi_transform(in_data, rois, rois_lod, transformed_height,
                  transformed_width, spatial_scale):
    channels = in_data.shape[1]
    in_height = in_data.shape[2]
    in_width = in_data.shape[3]
    rois_num = rois.shape[0]

    roi2image = [0] * rois_num
    rois_lod = lod_convert(rois_lod[0])
    for i in range(len(rois_lod) - 1):
        for j in range(rois_lod[i], rois_lod[i + 1]):
            roi2image[j] = i

    out = np.zeros([rois_num, channels, transformed_height, transformed_width])
    mask = np.zeros([rois_num, 1, transformed_height,
                     transformed_width]).astype('int')
    matrix = np.zeros([rois_num, 9], dtype=in_data.dtype)
    for n in range(rois_num):
        roi_x = []
        roi_y = []
        for k in range(4):
            roi_x.append(rois[n][2 * k] * spatial_scale)
            roi_y.append(rois[n][2 * k + 1] * spatial_scale)
        image_id = roi2image[n]
        transform_matrix = get_transform_matrix(transformed_width,
                                                transformed_height, roi_x,
                                                roi_y)
        matrix[n] = transform_matrix
        for c in range(channels):
            for out_h in range(transformed_height):
                for out_w in range(transformed_width):
                    in_w, in_h = get_source_coords(transform_matrix, out_w,
                                                   out_h)
                    if in_quad(in_w, in_h, roi_x, roi_y) and gt(
                            in_w, -0.5) and gt(in_width - 0.5, in_w) and gt(
                                in_h, -0.5) and gt(in_height - 0.5, in_h):
                        out[n][c][out_h][out_w] = bilinear_interpolate(
                            in_data, image_id, c, in_w, in_h)
                        mask[n][0][out_h][out_w] = 1
                    else:
                        out[n][c][out_h][out_w] = 0.0
                        mask[n][0][out_h][out_w] = 0
    return out.astype("float32"), mask, matrix


class TestROIPoolOp(OpTest):

    def set_data(self):
        self.init_test_case()
        self.make_rois()

        self.inputs = {'X': self.x, 'ROIs': (self.rois, self.rois_lod)}

        self.attrs = {
            'spatial_scale': self.spatial_scale,
            'transformed_height': self.transformed_height,
            'transformed_width': self.transformed_width
        }
        out, mask, transform_matrix = roi_transform(self.x, self.rois,
                                                    self.rois_lod,
                                                    self.transformed_height,
                                                    self.transformed_width,
                                                    self.spatial_scale)
        self.outputs = {
            'Out': out,
            'Mask': mask,
            'TransformMatrix': transform_matrix
        }

    def init_test_case(self):
        self.batch_size = 2
        self.channels = 2
        self.height = 8
        self.width = 8

        # n, c, h, w
        self.x_dim = (self.batch_size, self.channels, self.height, self.width)

        self.spatial_scale = 1.0 / 2.0
        self.transformed_height = 2
        self.transformed_width = 3

        self.x = np.random.random(self.x_dim).astype('float32')

    def make_rois(self):
        rois = []
        self.rois_lod = [[]]
        for bno in range(self.batch_size):
            self.rois_lod[0].append(bno + 1)
            for i in range(bno + 1):
                x1 = np.random.randint(
                    0,
                    self.width // self.spatial_scale - self.transformed_width)
                y1 = np.random.randint(
                    0,
                    self.height // self.spatial_scale - self.transformed_height)

                x2 = np.random.randint(x1 + self.transformed_width,
                                       self.width // self.spatial_scale)
                y2 = np.random.randint(
                    0,
                    self.height // self.spatial_scale - self.transformed_height)

                x3 = np.random.randint(x1 + self.transformed_width,
                                       self.width // self.spatial_scale)
                y3 = np.random.randint(y1 + self.transformed_height,
                                       self.height // self.spatial_scale)

                x4 = np.random.randint(
                    0,
                    self.width // self.spatial_scale - self.transformed_width)
                y4 = np.random.randint(y1 + self.transformed_height,
                                       self.height // self.spatial_scale)

                roi = [x1, y1, x2, y2, x3, y3, x4, y4]
                rois.append(roi)
        self.rois_num = len(rois)
        self.rois = np.array(rois).astype("float32")

    def setUp(self):
        self.op_type = "roi_perspective_transform"
        self.set_data()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.outputs['Out2InIdx'] = np.zeros(
            [np.product(self.outputs['Out'].shape), 4]).astype("int32")
        self.outputs['Out2InWeights'] = np.zeros(
            [np.product(self.outputs['Out'].shape), 4]).astype("float32")
        self.check_grad(['X'], 'Out')

    def test_errors(self):
        x = fluid.data(name='x', shape=[100, 256, 28, 28], dtype='float32')
        rois = fluid.data(name='rois',
                          shape=[None, 8],
                          lod_level=1,
                          dtype='float32')

        x_int = fluid.data(name='x_int',
                           shape=[100, 256, 28, 28],
                           dtype='int32')
        rois_int = fluid.data(name='rois_int',
                              shape=[None, 8],
                              lod_level=1,
                              dtype='int32')
        x_tmp = [1, 2]
        rois_tmp = [1, 2]

        # type of intput and rois must be variable
        self.assertRaises(TypeError, fluid.layers.roi_perspective_transform,
                          x_tmp, rois, 7, 7)
        self.assertRaises(TypeError, fluid.layers.roi_perspective_transform, x,
                          rois_tmp, 7, 7)

        # dtype of intput and rois must be float32
        self.assertRaises(TypeError, fluid.layers.roi_perspective_transform,
                          x_int, rois, 7, 7)
        self.assertRaises(TypeError, fluid.layers.roi_perspective_transform, x,
                          rois_int, 7, 7)

        height = 7.5
        width = 7.5
        # type of transformed_height and transformed_width must be int
        self.assertRaises(TypeError, fluid.layers.roi_perspective_transform, x,
                          rois, height, 7)
        self.assertRaises(TypeError, fluid.layers.roi_perspective_transform, x,
                          rois, 7, width)

        scale = int(2)
        # type of spatial_scale must be float
        self.assertRaises(TypeError, fluid.layers.roi_perspective_transform, x,
                          rois, 7, 7, scale)


if __name__ == '__main__':
    unittest.main()
