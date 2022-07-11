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


def compute_graph_send_e_recv_for_sum(inputs, attributes):
    x = inputs['X']
    e = inputs['E']
    src_index = inputs['Src_index']
    dst_index = inputs['Dst_index']
    compute_type = attributes['compute_type']

    gather_x = x[src_index]
    out_shp = [
        x.shape[0],
    ] + get_broadcast_shape(x.shape[1:], e.shape[1:])
    results = np.zeros(out_shp, dtype=x.dtype)

    # Calculate forward output.
    if compute_type == 'ADD':
        x_compute_e = gather_x + e
    elif compute_type == 'MUL':
        x_compute_e = gather_x * e
    for index, s_id in enumerate(dst_index):
        results[s_id, :] += x_compute_e[index, :]
    return results


def compute_graph_send_e_recv_for_mean(inputs, attributes):
    x = inputs['X']
    e = inputs['E']
    src_index = inputs['Src_index']
    dst_index = inputs['Dst_index']
    compute_type = attributes['compute_type']

    gather_x = x[src_index]
    out_shp = [
        x.shape[0],
    ] + get_broadcast_shape(x.shape[1:], e.shape[1:])
    results = np.zeros(out_shp, dtype=x.dtype)

    # Calculate forward output.
    if compute_type == 'ADD':
        x_compute_e = gather_x + e
    elif compute_type == 'MUL':
        x_compute_e = gather_x * e
    count = np.zeros(out_shp[0], dtype=np.int32)
    for index, s_id in enumerate(dst_index):
        results[s_id, :] += x_compute_e[index, :]
        count[s_id] += 1
    results = results / count.reshape([-1, 1])
    results[np.isnan(results)] = 0
    return results, count


def compute_graph_send_e_recv_for_max_min(inputs, attributes):
    x = inputs['X']
    e = inputs['E']
    src_index = inputs['Src_index']
    dst_index = inputs['Dst_index']
    compute_type = attributes['compute_type']
    pool_type = attributes['pool_type']

    gather_x = x[src_index]
    out_shp = [
        x.shape[0],
    ] + get_broadcast_shape(x.shape[1:], e.shape[1:])
    results = np.zeros(out_shp, dtype=x.dtype)

    # Calculate forward output.
    if compute_type == 'ADD':
        x_compute_e = gather_x + e
    elif compute_type == 'MUL':
        x_compute_e = gather_x * e

    first_set = set()
    if pool_type == 'MAX':
        for index, s_id in enumerate(dst_index):
            if s_id not in first_set:
                results[s_id, :] += x_compute_e[index, :]
                first_set.add(s_id)
            else:
                results[s_id, :] = np.maximum(results[s_id, :],
                                              x_compute_e[index, :])
    elif pool_type == 'MIN':
        for index, s_id in enumerate(dst_index):
            if s_id not in first_set:
                results[s_id, :] += x_compute_e[index, :]
                first_set.add(s_id)
            else:
                results[s_id, :] = np.minimum(results[s_id, :],
                                              x_compute_e[index, :])
    else:
        raise ValueError("Invalid pool_type, only MAX, MIN supported!")

    # Calculate backward gradient.
    x_gradient = np.zeros_like(x)
    e_gradient = np.zeros_like(e)
    bcast_info = BroadCastInfo(x.shape, e.shape)
    use_broadcast = bcast_info.use_broadcast
    for i in range(len(src_index)):
        forward_src_idx = src_index[i]
        forward_dst_idx = dst_index[i]
        x_off = x[forward_src_idx]
        e_off = e[i]
        out_off = results[forward_dst_idx]
        x_grad_off = x_gradient[forward_src_idx]
        e_grad_off = e_gradient[i]
        for j in range(bcast_info.out_len):
            x_add = bcast_info.lhs_offset[j] if use_broadcast else j
            e_add = bcast_info.rhs_offset[j] if use_broadcast else j
            if compute_type == 'ADD':
                if len(x_off.shape) == 1 and len(e_off.shape) == 1:
                    val = x_off[x_add] + e_off[e_add]
                    x_grad_off[x_add] += 1 * (val == out_off[j])
                    e_grad_off[e_add] += 1 * (val == out_off[j])
                else:
                    # For simplicity, we only check the situation of x_off.shape=2
                    x_add_0 = int(x_add / x_off.shape[1])
                    x_add_1 = int(x_add % x_off.shape[1])
                    e_add_0 = int(e_add / e_off.shape[1])
                    e_add_1 = int(e_add % e_off.shape[1])
                    out_add_0 = int(j / out_off.shape[1])
                    out_add_1 = int(j % out_off.shape[1])
                    val = x_off[x_add_0][x_add_1] + e_off[e_add_0][e_add_1]
                    x_grad_off[x_add_0][x_add_1] += 1 * (
                        val == out_off[out_add_0][out_add_1])
                    e_grad_off[e_add_0][e_add_1] += 1 * (
                        val == out_off[out_add_0][out_add_1])
            elif compute_type == 'MUL':
                if len(x_off.shape) == 1 and len(e_off.shape) == 1:
                    val = x_off[x_add] * e_off[e_add]
                    x_grad_off[x_add] += 1 * (val == out_off[j]) * e_off[e_add]
                    e_grad_off[e_add] += 1 * (val == out_off[j]) * x_off[x_add]
                else:
                    # For simplicity, we only check the situation of x_off.shape=2
                    x_add_0 = int(x_add / x_off.shape[1])
                    x_add_1 = int(x_add % x_off.shape[1])
                    e_add_0 = int(e_add / e_off.shape[1])
                    e_add_1 = int(e_add % e_off.shape[1])
                    out_add_0 = int(j / out_off.shape[1])
                    out_add_1 = int(j % out_off.shape[1])
                    val = x_off[x_add_0][x_add_1] * e_off[e_add_0][e_add_1]
                    x_grad_off[x_add_0][x_add_1] += 1 * (
                        val == out_off[out_add_0][out_add_1]
                    ) * e_off[e_add_0][e_add_1]
                    e_grad_off[e_add_0][e_add_1] += 1 * (
                        val == out_off[out_add_0][out_add_1]
                    ) * x_off[x_add_0][x_add_1]

    gradients = [x_gradient / results.size, e_gradient / results.size]

    return results, gradients


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
        self.x = np.random.random((10, 20)).astype("float64")
        self.e = np.random.random((15, 20)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'MUL'


class TestSumCase2(TestGraphSendERecvSumOp):

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.e = np.random.random((150, 1)).astype("float64")
        index = np.random.randint(0, 10, (150, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'ADD'


class TestSumCase3(TestGraphSendERecvSumOp):

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.e = np.random.random((150, 1)).astype("float64")
        index = np.random.randint(0, 10, (150, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'MUL'


class TestSumCase4(TestGraphSendERecvSumOp):

    def set_config(self):
        self.x = np.random.random((10, 8, 5)).astype("float64")
        self.e = np.random.random((15, 8, 1)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'ADD'


class TestSumCase5(TestGraphSendERecvSumOp):

    def set_config(self):
        self.x = np.random.random((10, 8, 5)).astype("float64")
        self.e = np.random.random((15, 8, 1)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'MUL'


class TestGraphSendERecvMeanOp(OpTest):

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
        self.attrs = {'compute_type': self.compute_type, 'pool_type': 'MEAN'}

        out, dst_count = compute_graph_send_e_recv_for_mean(
            self.inputs, self.attrs)

        self.outputs = {'Out': out, 'Dst_count': dst_count}

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


def TestMeanCase1(TestGraphSendERecvMeanOp):

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.e = np.random.random((15, 20)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'MUL'


def TestMeanCase2(TestGraphSendERecvMeanOp):

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.e = np.random.random((150, 1)).astype("float64")
        index = np.random.randint(0, 10, (150, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'SUM'


def TestMeanCase3(TestGraphSendERecvMeanOp):

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.e = np.random.random((150, 1)).astype("float64")
        index = np.random.randint(0, 10, (150, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'MUL'


def TestMeanCase4(TestGraphSendERecvMeanOp):

    def set_config(self):
        self.x = np.random.random((10, 8, 5)).astype("float64")
        self.e = np.random.random((15, 8, 1)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'SUM'


def TestMeanCase5(TestGraphSendERecvMeanOp):

    def set_config(self):
        self.x = np.random.random((10, 8, 5)).astype("float64")
        self.e = np.random.random((15, 8, 1)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'MUL'


class TestGraphSendERecvMaxOp(OpTest):

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
        self.attrs = {'compute_type': self.compute_type, 'pool_type': 'MAX'}

        out, self.gradients = compute_graph_send_e_recv_for_max_min(
            self.inputs, self.attrs)

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
        self.check_grad(['X', 'E'], 'Out', user_defined_grads=self.gradients)


class TestMaxCase1(TestGraphSendERecvMaxOp):

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.e = np.random.random((15, 20)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'MUL'


class TestMaxCase2(TestGraphSendERecvMaxOp):

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.e = np.random.random((150, 1)).astype("float64")
        index = np.random.randint(0, 10, (150, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'ADD'


class TestMaxCase3(TestGraphSendERecvMaxOp):

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.e = np.random.random((150, 1)).astype("float64")
        index = np.random.randint(0, 10, (150, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'MUL'


class TestMaxCase4(TestGraphSendERecvMaxOp):

    def set_config(self):
        self.x = np.random.random((10, 8, 5)).astype("float64")
        self.e = np.random.random((15, 8, 1)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'ADD'


class TestMaxCase5(TestGraphSendERecvMaxOp):

    def set_config(self):
        self.x = np.random.random((10, 8, 5)).astype("float64")
        self.e = np.random.random((15, 8, 1)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'MUL'


class TestGraphSendERecvMinOp(OpTest):

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
        self.attrs = {'compute_type': self.compute_type, 'pool_type': 'MIN'}

        out, self.gradients = compute_graph_send_e_recv_for_max_min(
            self.inputs, self.attrs)

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
        self.check_grad(['X', 'E'], 'Out', user_defined_grads=self.gradients)


class TestMinCase1(TestGraphSendERecvMinOp):

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.e = np.random.random((15, 20)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'MUL'


class TestMinCase2(TestGraphSendERecvMinOp):

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.e = np.random.random((150, 1)).astype("float64")
        index = np.random.randint(0, 10, (150, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'ADD'


class TestMinCase3(TestGraphSendERecvMinOp):

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.e = np.random.random((150, 1)).astype("float64")
        index = np.random.randint(0, 10, (150, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'MUL'


class TestMinCase4(TestGraphSendERecvMinOp):

    def set_config(self):
        self.x = np.random.random((10, 8, 5)).astype("float64")
        self.e = np.random.random((15, 8, 1)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'ADD'


class TestMinCase5(TestGraphSendERecvMinOp):

    def set_config(self):
        self.x = np.random.random((10, 8, 5)).astype("float64")
        self.e = np.random.random((15, 8, 1)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.compute_type = 'MUL'
