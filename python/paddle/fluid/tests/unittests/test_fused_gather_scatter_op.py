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

import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid


class TestFusedGatherScatterMaxOp(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.op_type = "fused_gather_scatter"
        x = np.random.random((10, 20)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        gather_index = index[:, 0]
        scatter_index = index[:, 1]

        self.inputs = {
            'X': x,
            'Gather_index': gather_index,
            'Scatter_index': scatter_index
        }

        self.attrs = {'pool_type': 'MAX'}

        out, self.gradient = compute_gather_scatter_for_min_max(self.inputs,
                                                                self.attrs)
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', user_defined_grads=[self.gradient])


class TestFusedGatherScatterMinOp(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.op_type = "fused_gather_scatter"
        x = np.random.random((10, 20)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        gather_index = index[:, 0]
        scatter_index = index[:, 1]

        self.inputs = {
            'X': x,
            'Gather_index': gather_index,
            'Scatter_index': scatter_index
        }

        self.attrs = {'pool_type': 'MIN'}

        out, self.gradient = compute_gather_scatter_for_min_max(self.inputs,
                                                                self.attrs)

        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', user_defined_grads=[self.gradient])


class TestFusedGatherScatterSumOp(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.op_type = "fused_gather_scatter"
        x = np.random.random((10, 20)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        gather_index = index[:, 0]
        scatter_index = index[:, 1]

        self.inputs = {
            'X': x,
            'Gather_index': gather_index,
            'Scatter_index': scatter_index
        }

        self.attrs = {'pool_type': 'SUM'}

        out, _ = compute_gather_scatter_for_sum_mean(self.inputs, self.attrs)

        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestFusedGatherScatterMeanOp(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.op_type = "fused_gather_scatter"
        x = np.random.random((10, 20)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        gather_index = index[:, 0]
        scatter_index = index[:, 1]

        self.inputs = {
            'X': x,
            'Gather_index': gather_index,
            'Scatter_index': scatter_index
        }

        self.attrs = {'pool_type': 'MEAN'}

        out, scatter_count = compute_gather_scatter_for_sum_mean(self.inputs,
                                                                 self.attrs)

        self.outputs = {'Out': out, 'Scatter_count': scatter_count}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


def compute_gather_scatter_for_sum_mean(inputs, attributes):
    x = inputs['X']
    gather_index = inputs['Gather_index']
    scatter_index = inputs['Scatter_index']

    pool_type = attributes['pool_type']

    gather_x = x[gather_index]
    target_shape = list(x.shape)
    results = np.zeros(target_shape, dtype=x.dtype)
    if pool_type == 'SUM':
        for index, s_id in enumerate(scatter_index):
            results[s_id, :] += gather_x[index, :]
    elif pool_type == 'MEAN':
        count = np.zeros(target_shape[0], dtype=np.int32)
        for index, s_id in enumerate(scatter_index):
            results[s_id, :] += gather_x[index, :]
            count[s_id] += 1
        results = results / count.reshape([-1, 1])
        results[np.isnan(results)] = 0
    else:
        raise ValueError("Invalid pool_type, only SUM, MEAN supported!")

    count = np.zeros(target_shape[0], dtype=np.int32)
    for index, s_id in enumerate(scatter_index):
        count[s_id] += 1

    return results, count


def compute_gather_scatter_for_min_max(inputs, attributes):
    x = inputs['X']
    gather_index = inputs['Gather_index']
    scatter_index = inputs['Scatter_index']

    pool_type = attributes['pool_type']

    gather_x = x[gather_index]
    target_shape = list(x.shape)
    results = np.zeros(target_shape, dtype=x.dtype)
    gradient = np.zeros_like(x)

    # Calculate forward output 
    if pool_type == "MAX":
        first_set = set()
        for index, s_id in enumerate(scatter_index):
            if s_id not in first_set:
                results[s_id, :] += gather_x[index, :]
                first_set.add(s_id)
            else:
                results[s_id, :] = np.maximum(results[s_id, :],
                                              gather_x[index, :])
    elif pool_type == "MIN":
        first_set = set()
        for index, s_id in enumerate(scatter_index):
            if s_id not in first_set:
                results[s_id, :] += gather_x[index, :]
                first_set.add(s_id)
            else:
                results[s_id, :] = np.minimum(results[s_id, :],
                                              gather_x[index, :])
    else:
        raise ValueError("Invalid pool_type, only MAX, MIN supported!")

    # Calculate backward gradient
    index_size = len(gather_index)
    for i in range(index_size):
        forward_src_idx = gather_index[i]
        forward_dst_idx = scatter_index[i]
        gradient[forward_src_idx] += 1 * (
            x[forward_src_idx] == results[forward_dst_idx])

    return results, gradient / results.size
