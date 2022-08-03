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
        self.check_output(check_eager=False)

    def test_check_grad(self):
        self.check_grad(['x', 'y'], 'out', check_eager=False)

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.y = np.random.random((10, 20)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'ADD'
