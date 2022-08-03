# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.framework import _test_eager_guard

from op_test import OpTest


def get_broadcast_shape(shp1, shp2):
    pad_shp1, pad_shp2 = shp1, shp2
    if len(shp1) > len(shp2):
        pad_shp2 = [
            1,
        ] * (len(shp1) - len(shp2)) + shp2
    elif len(shp1) < len(shp2):
        pad_shp1 = [
            1,
        ] * (len(shp2) - len(shp1)) + shp1
    for d1, d2 in zip(pad_shp1, pad_shp2):
        if d1 != d2 and d1 != 1 and d2 != 1:
            raise ValueError
    rst = [max(d1, d2) for d1, d2 in zip(pad_shp1, pad_shp2)]
    return rst


class BroadCastInfo(object):

    def __init__(self, x_shape, e_shape):
        self.x_shape = x_shape
        self.e_shape = e_shape

        self.calculate_bcastinfo()

    def use_bcast(self):
        if len(self.x_shape) != len(self.e_shape):
            return True
        for i in range(1, len(self.x_shape)):
            if self.x_shape[i] != self.e_shape[i]:
                return True
        return False

    def calculate_bcastinfo(self):
        lhs_len = 1
        rhs_len = 1
        for i in range(1, len(self.x_shape)):
            lhs_len *= self.x_shape[i]
        for i in range(1, len(self.e_shape)):
            rhs_len *= self.e_shape[i]
        use_b = self.use_bcast()

        if use_b:
            max_ndim = max(len(self.x_shape), len(self.e_shape)) - 1
            out_len = 1
            stride_l = stride_r = 1
            lhs_offset = [0]
            rhs_offset = [0]
            for j in range(0, max_ndim):
                dl = 1 if (len(self.x_shape) - 1 - j) < 1 \
                       else self.x_shape[len(self.x_shape) - 1 - j]
                dr = 1 if (len(self.e_shape) - 1 - j) < 1 \
                       else self.e_shape[len(self.e_shape) - 1 - j]
                for i in range(1, max(dl, dr)):
                    for k in range(0, out_len):
                        lhs_offset.append(lhs_offset[k] + i *
                                          (i < dl) * stride_l)
                        rhs_offset.append(rhs_offset[k] + i *
                                          (i < dr) * stride_r)

                out_len *= max(dl, dr)
                stride_l *= dl
                stride_r *= dr
        else:
            out_len = rhs_len

        self.use_broadcast = use_b
        self.out_len = out_len
        self.lhs_len = lhs_len
        self.rhs_len = rhs_len
        if use_b:
            self.lhs_offset = lhs_offset
            self.rhs_offset = rhs_offset


def compute_graph_send_uv(inputs, attributes):
    x = inputs['x']
    y = inputs['y']
    src_index = inputs['src_index']
    dst_index = inputs['dst_index']
    compute_type = attributes['compute_type']

    gather_x = x[src_index]
    gather_y = y[dst_index]

    # Calculate forward output.
    if compute_type == "ADD":
        results = gather_x + gather_y
    elif compute_type == "MUL":
        results = gather_x * gather_y
    return results


class TestGraphSendUVOp(OpTest):

    def setUp(self):
        paddle.enable_static()
        self.op_type = "graph_send_uv"
        self.set_config()
        self.inputs = {
            'x': self.x,
            'y': self.y,
            'src_index': self.src_index,
            'dst_index': self.dst_index
        }
        self.attrs = {'compute_type': self.compute_type}
        out = compute_graph_send_uv(self.inputs, self.attrs)
        self.outputs = {'out': out}

    def test_check_output(self):
        self.check_output(check_eager=True)

    # def test_check_grad(self):
    #     self.check_grad(['x', 'y'], 'Out', check_eager=True)

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.y = np.random.random((10, 20)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'ADD'
