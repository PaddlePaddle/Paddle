# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.framework import _test_eager_guard

from op_test import OpTest


def get_broadcast_shape(shp1, shp2):
    pad_shp1, pad_shp2 = shp1, shp2
    if len(shp1) > len(shp2):
        pad_shp2 = [1, ] * (len(shp1) - len(shp2)) + shp2
    elif len(shp1) < len(shp2):
        pad_shp1 = [1, ] * (len(shp2) - len(shp1)) + shp1
    for d1, d2 in zip(pad_shp1, pad_shp2):
        if d1 != d2 and d1 != 1 and d2 != 1:
            raise ValueError
    rst = [max(d1, d2) for d1, d2 in zip(pad_shp1, pad_shp2)]
    return rst


def compute_graph_send_e_recv_for_sum(inputs, attributes):
    x = inputs['X']
    e = inputs['E']
    src_index = inputs['Src_index']
    dst_index = inputs['Dst_index']
    compute_type = attributes['compute_type']

    gather_x = x[src_index]
    out_shp = [x.shape[0], ] + get_broadcast_shape(x.shape[1:], e.shape[1:])
    results = np.zeros(out_shp, dtype=x.dtype)

    # Calculate forward output
    if compute_type == 'ADD':
        x_compute_e = gather_x + e
    elif compute_type == 'MUL':
        x_compute_e = gather_x * e
    for index, s_id in enumerate(dst_index):
        results[s_id, :] += x_compute_e[index, :]
    return results


class TestGraphSendERecvSumOp(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.op_type = "graph_send_e_recv"
        self.set_config()
        self.inputs = {
            'X': self.x,
            'E': self.e,
            'Src_index': self.src_index,
            'Dst_index': self.dst_index
        }
        self.attrs = {'compute_type': self.compute_type, 'pool_type': 'SUM'}

        out = compute_graph_send_e_recv_for_sum(self.inputs, self.attrs)

        self.outputs = {'Out': out}

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.e = np.random.random((15, 20)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'ADD'

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'E'], 'Out')


class TestSumCase1(TestGraphSendERecvSumOp):
    def set_config(self):
        self.x = np.random.random((100, 20)).astype("float64")
        self.e = np.random.random((150, 1)).astype("float64")
        index = np.random.randint(0, 100, (150, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'ADD'


class TestSumCase2(TestGraphSendERecvSumOp):
    def set_config(self):
        self.x = np.random.random((100, 1)).astype("float64")
        self.e = np.random.random((15, 20)).astype("float64")
        index = np.random.randint(0, 100, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'ADD'


class TestSumCase3(TestGraphSendERecvSumOp):
    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.e = np.random.random((15, 20)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'MUL'


class TestSumCase4(TestGraphSendERecvSumOp):
    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.e = np.random.random((150, 1)).astype("float64")
        index = np.random.randint(0, 10, (150, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'MUL'


class TestSumCase5(TestGraphSendERecvSumOp):
    def set_config(self):
        self.x = np.random.random((100, 1)).astype("float64")
        self.e = np.random.random((15, 20)).astype("float64")
        index = np.random.randint(0, 100, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'MUL'
