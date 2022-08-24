# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The DGL team.

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

    def __init__(self, x_shape, y_shape):
        self.x_shape = x_shape
        self.y_shape = y_shape

        self.calculate_bcastinfo()

    def use_bcast(self):
        if len(self.x_shape) != len(self.y_shape):
            return True
        for i in range(1, len(self.x_shape)):
            if self.x_shape[i] != self.y_shape[i]:
                return True
        return False

    def calculate_bcastinfo(self):
        lhs_len = 1
        rhs_len = 1
        for i in range(1, len(self.x_shape)):
            lhs_len *= self.x_shape[i]
        for i in range(1, len(self.y_shape)):
            rhs_len *= self.y_shape[i]
        use_b = self.use_bcast()

        if use_b:
            max_ndim = max(len(self.x_shape), len(self.y_shape)) - 1
            out_len = 1
            stride_l = stride_r = 1
            lhs_offset = [0]
            rhs_offset = [0]
            for j in range(0, max_ndim):
                dl = 1 if (len(self.x_shape) - 1 - j) < 1 \
                       else self.x_shape[len(self.x_shape) - 1 - j]
                dr = 1 if (len(self.y_shape) - 1 - j) < 1 \
                       else self.y_shape[len(self.y_shape) - 1 - j]
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


def compute_graph_send_ue_recv_for_sum(inputs, attributes):
    x = inputs['X']
    y = inputs['Y']
    src_index = inputs['Src_index']
    dst_index = inputs['Dst_index']
    message_op = attributes['message_op']

    gather_x = x[src_index]
    out_shp = [
        x.shape[0],
    ] + get_broadcast_shape(x.shape[1:], y.shape[1:])
    results = np.zeros(out_shp, dtype=x.dtype)

    # Calculate forward output.
    if message_op == 'ADD':
        x_compute_y = gather_x + y
    elif message_op == 'MUL':
        x_compute_y = gather_x * y
    for index, s_id in enumerate(dst_index):
        results[s_id, :] += x_compute_y[index, :]
    return results


def compute_graph_send_ue_recv_for_mean(inputs, attributes):
    x = inputs['X']
    y = inputs['Y']
    src_index = inputs['Src_index']
    dst_index = inputs['Dst_index']
    message_op = attributes['message_op']

    gather_x = x[src_index]
    out_shp = [
        x.shape[0],
    ] + get_broadcast_shape(x.shape[1:], y.shape[1:])
    results = np.zeros(out_shp, dtype=x.dtype)

    # Calculate forward output.
    if message_op == 'ADD':
        x_compute_y = gather_x + y
    elif message_op == 'MUL':
        x_compute_y = gather_x * y
    count = np.zeros(out_shp[0], dtype=np.int32)
    for index, s_id in enumerate(dst_index):
        results[s_id, :] += x_compute_y[index, :]
        count[s_id] += 1
    count_shape = [out_shp[0]]
    count_shape.extend([1] * len(out_shp[1:]))
    results = results / count.reshape(count_shape)
    results[np.isnan(results)] = 0
    return results, count


def compute_graph_send_ue_recv_for_max_min(inputs, attributes):
    x = inputs['X']
    y = inputs['Y']
    src_index = inputs['Src_index']
    dst_index = inputs['Dst_index']
    message_op = attributes['message_op']
    reduce_op = attributes['reduce_op']

    gather_x = x[src_index]
    out_shp = [
        x.shape[0],
    ] + get_broadcast_shape(x.shape[1:], y.shape[1:])
    results = np.zeros(out_shp, dtype=x.dtype)

    # Calculate forward output.
    if message_op == 'ADD':
        x_compute_y = gather_x + y
    elif message_op == 'MUL':
        x_compute_y = gather_x * y

    first_set = set()
    if reduce_op == 'MAX':
        for index, s_id in enumerate(dst_index):
            if s_id not in first_set:
                results[s_id, :] += x_compute_y[index, :]
                first_set.add(s_id)
            else:
                results[s_id, :] = np.maximum(results[s_id, :],
                                              x_compute_y[index, :])
    elif reduce_op == 'MIN':
        for index, s_id in enumerate(dst_index):
            if s_id not in first_set:
                results[s_id, :] += x_compute_y[index, :]
                first_set.add(s_id)
            else:
                results[s_id, :] = np.minimum(results[s_id, :],
                                              x_compute_y[index, :])
    else:
        raise ValueError("Invalid reduce_op, only MAX, MIN supported!")

    # Calculate backward gradient.
    x_gradient = np.zeros_like(x)
    y_gradient = np.zeros_like(y)
    bcast_info = BroadCastInfo(x.shape, y.shape)
    use_broadcast = bcast_info.use_broadcast
    for i in range(len(src_index)):
        forward_src_idx = src_index[i]
        forward_dst_idx = dst_index[i]
        x_off = x[forward_src_idx]
        y_off = y[i]
        out_off = results[forward_dst_idx]
        x_grad_off = x_gradient[forward_src_idx]
        y_grad_off = y_gradient[i]
        for j in range(bcast_info.out_len):
            x_add = bcast_info.lhs_offset[j] if use_broadcast else j
            y_add = bcast_info.rhs_offset[j] if use_broadcast else j
            if message_op == 'ADD':
                if len(x_off.shape) == 1 and len(y_off.shape) == 1:
                    val = x_off[x_add] + y_off[y_add]
                    x_grad_off[x_add] += 1 * (val == out_off[j])
                    y_grad_off[y_add] += 1 * (val == out_off[j])
                else:
                    # For simplicity, we only check the situation of x_off.shape=2
                    x_add_0 = int(x_add / x_off.shape[1])
                    x_add_1 = int(x_add % x_off.shape[1])
                    y_add_0 = int(y_add / y_off.shape[1])
                    y_add_1 = int(y_add % y_off.shape[1])
                    out_add_0 = int(j / out_off.shape[1])
                    out_add_1 = int(j % out_off.shape[1])
                    val = x_off[x_add_0][x_add_1] + y_off[y_add_0][y_add_1]
                    x_grad_off[x_add_0][x_add_1] += 1 * (
                        val == out_off[out_add_0][out_add_1])
                    y_grad_off[y_add_0][y_add_1] += 1 * (
                        val == out_off[out_add_0][out_add_1])
            elif message_op == 'MUL':
                if len(x_off.shape) == 1 and len(y_off.shape) == 1:
                    val = x_off[x_add] * y_off[y_add]
                    x_grad_off[x_add] += 1 * (val == out_off[j]) * y_off[y_add]
                    y_grad_off[y_add] += 1 * (val == out_off[j]) * x_off[x_add]
                else:
                    # For simplicity, we only check the situation of x_off.shape=2
                    x_add_0 = int(x_add / x_off.shape[1])
                    x_add_1 = int(x_add % x_off.shape[1])
                    y_add_0 = int(y_add / y_off.shape[1])
                    y_add_1 = int(y_add % y_off.shape[1])
                    out_add_0 = int(j / out_off.shape[1])
                    out_add_1 = int(j % out_off.shape[1])
                    val = x_off[x_add_0][x_add_1] * y_off[y_add_0][y_add_1]
                    x_grad_off[x_add_0][x_add_1] += 1 * (
                        val == out_off[out_add_0][out_add_1]
                    ) * y_off[y_add_0][y_add_1]
                    y_grad_off[y_add_0][y_add_1] += 1 * (
                        val == out_off[out_add_0][out_add_1]
                    ) * x_off[x_add_0][x_add_1]

    gradients = [x_gradient / results.size, y_gradient / results.size]

    return results, gradients


def graph_send_ue_recv_wrapper(x,
                               y,
                               src_index,
                               dst_index,
                               message_op="add",
                               reduce_op="sum",
                               out_size=None,
                               name=None):
    return paddle.geometric.send_ue_recv(x, y, src_index, dst_index,
                                         message_op.lower(), reduce_op.lower(),
                                         out_size, name)


class TestGraphSendUERecvSumOp(OpTest):

    def setUp(self):
        paddle.enable_static()
        self.python_api = graph_send_ue_recv_wrapper
        self.python_out_sig = ["Out"]
        self.op_type = "graph_send_ue_recv"
        self.set_config()
        self.inputs = {
            'X': self.x,
            'Y': self.y,
            'Src_index': self.src_index,
            'Dst_index': self.dst_index
        }
        self.attrs = {'message_op': self.message_op, 'reduce_op': 'SUM'}

        out = compute_graph_send_ue_recv_for_sum(self.inputs, self.attrs)

        self.outputs = {'Out': out}

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.y = np.random.random((15, 20)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'ADD'

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(['X', 'Y'], 'Out', check_eager=True)


class TestSumCase1(TestGraphSendUERecvSumOp):

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.y = np.random.random((15, 20)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'MUL'


class TestSumCase2(TestGraphSendUERecvSumOp):

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.y = np.random.random((150, 1)).astype("float64")
        index = np.random.randint(0, 10, (150, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'ADD'


class TestSumCase3(TestGraphSendUERecvSumOp):

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.y = np.random.random((150, 1)).astype("float64")
        index = np.random.randint(0, 10, (150, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'MUL'


class TestSumCase4(TestGraphSendUERecvSumOp):

    def set_config(self):
        self.x = np.random.random((10, 8, 5)).astype("float64")
        self.y = np.random.random((15, 8, 1)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'ADD'


class TestSumCase5(TestGraphSendUERecvSumOp):

    def set_config(self):
        self.x = np.random.random((10, 8, 5)).astype("float64")
        self.y = np.random.random((15, 8, 1)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'MUL'


class TestSumCase6(TestGraphSendUERecvSumOp):

    def set_config(self):
        self.x = np.random.random((100, 1)).astype("float64")
        self.y = np.random.random((15, 20)).astype("float64")
        index = np.random.randint(0, 100, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'ADD'


class TestSumCase7(TestGraphSendUERecvSumOp):

    def set_config(self):
        self.x = np.random.random((100, 1)).astype("float64")
        self.y = np.random.random((15, 20)).astype("float64")
        index = np.random.randint(0, 100, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'MUL'


class TestGraphSendUERecvMeanOp(OpTest):

    def setUp(self):
        paddle.enable_static()
        self.python_api = graph_send_ue_recv_wrapper
        self.python_out_sig = ["Out"]
        self.op_type = "graph_send_ue_recv"
        self.set_config()
        self.inputs = {
            'X': self.x,
            'Y': self.y,
            'Src_index': self.src_index,
            'Dst_index': self.dst_index
        }
        self.attrs = {'message_op': self.message_op, 'reduce_op': 'MEAN'}

        out, dst_count = compute_graph_send_ue_recv_for_mean(
            self.inputs, self.attrs)

        self.outputs = {'Out': out, 'Dst_count': dst_count}

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.y = np.random.random((15, 20)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'ADD'

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(['X', 'Y'], 'Out', check_eager=True)


class TestMeanCase1(TestGraphSendUERecvMeanOp):

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.y = np.random.random((15, 20)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'MUL'


class TestMeanCase2(TestGraphSendUERecvMeanOp):

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.y = np.random.random((150, 1)).astype("float64")
        index = np.random.randint(0, 10, (150, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'ADD'


class TestMeanCase3(TestGraphSendUERecvMeanOp):

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.y = np.random.random((150, 1)).astype("float64")
        index = np.random.randint(0, 10, (150, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'MUL'


class TestMeanCase4(TestGraphSendUERecvMeanOp):

    def set_config(self):
        self.x = np.random.random((10, 8, 5)).astype("float64")
        self.y = np.random.random((15, 8, 1)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'ADD'


class TestMeanCase5(TestGraphSendUERecvMeanOp):

    def set_config(self):
        self.x = np.random.random((10, 8, 5)).astype("float64")
        self.y = np.random.random((15, 8, 1)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'MUL'


class TestMeanCase6(TestGraphSendUERecvMeanOp):

    def set_config(self):
        self.x = np.random.random((100, 1)).astype("float64")
        self.y = np.random.random((15, 20)).astype("float64")
        index = np.random.randint(0, 100, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'ADD'


class TestMeanCase7(TestGraphSendUERecvMeanOp):

    def set_config(self):
        self.x = np.random.random((100, 1)).astype("float64")
        self.y = np.random.random((15, 20)).astype("float64")
        index = np.random.randint(0, 100, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'MUL'


class TestGraphSendUERecvMaxOp(OpTest):

    def setUp(self):
        paddle.enable_static()
        self.python_api = graph_send_ue_recv_wrapper
        self.python_out_sig = ["Out"]
        self.op_type = "graph_send_ue_recv"
        self.set_config()
        self.inputs = {
            'X': self.x,
            'Y': self.y,
            'Src_index': self.src_index,
            'Dst_index': self.dst_index
        }
        self.attrs = {'message_op': self.message_op, 'reduce_op': 'MAX'}

        out, self.gradients = compute_graph_send_ue_recv_for_max_min(
            self.inputs, self.attrs)

        self.outputs = {'Out': out}

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.y = np.random.random((15, 20)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'ADD'

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(['X', 'Y'],
                        'Out',
                        user_defined_grads=self.gradients,
                        check_eager=True)


class TestMaxCase1(TestGraphSendUERecvMaxOp):

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.y = np.random.random((15, 20)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'MUL'


class TestMaxCase2(TestGraphSendUERecvMaxOp):

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.y = np.random.random((150, 1)).astype("float64")
        index = np.random.randint(0, 10, (150, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'ADD'


class TestMaxCase3(TestGraphSendUERecvMaxOp):

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.y = np.random.random((150, 1)).astype("float64")
        index = np.random.randint(0, 10, (150, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'MUL'


class TestMaxCase4(TestGraphSendUERecvMaxOp):

    def set_config(self):
        self.x = np.random.random((10, 8, 5)).astype("float64")
        self.y = np.random.random((15, 8, 1)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'ADD'


class TestMaxCase5(TestGraphSendUERecvMaxOp):

    def set_config(self):
        self.x = np.random.random((10, 8, 5)).astype("float64")
        self.y = np.random.random((15, 8, 1)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'MUL'


class TestMaxCase6(TestGraphSendUERecvMaxOp):

    def set_config(self):
        self.x = np.random.random((100, 1)).astype("float64")
        self.y = np.random.random((15, 20)).astype("float64")
        index = np.random.randint(0, 100, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'ADD'


class TestMaxCase7(TestGraphSendUERecvMaxOp):

    def set_config(self):
        self.x = np.random.random((100, 1)).astype("float64")
        self.y = np.random.random((15, 20)).astype("float64")
        index = np.random.randint(0, 100, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'MUL'


class TestGraphSendUERecvMinOp(OpTest):

    def setUp(self):
        paddle.enable_static()
        self.python_api = graph_send_ue_recv_wrapper
        self.python_out_sig = ["Out"]
        self.op_type = "graph_send_ue_recv"
        self.set_config()
        self.inputs = {
            'X': self.x,
            'Y': self.y,
            'Src_index': self.src_index,
            'Dst_index': self.dst_index
        }
        self.attrs = {'message_op': self.message_op, 'reduce_op': 'MIN'}

        out, self.gradients = compute_graph_send_ue_recv_for_max_min(
            self.inputs, self.attrs)

        self.outputs = {'Out': out}

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.y = np.random.random((15, 20)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'ADD'

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(['X', 'Y'],
                        'Out',
                        user_defined_grads=self.gradients,
                        check_eager=True)


class TestMinCase1(TestGraphSendUERecvMinOp):

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.y = np.random.random((15, 20)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'MUL'


class TestMinCase2(TestGraphSendUERecvMinOp):

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.y = np.random.random((150, 1)).astype("float64")
        index = np.random.randint(0, 10, (150, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'ADD'


class TestMinCase3(TestGraphSendUERecvMinOp):

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.y = np.random.random((150, 1)).astype("float64")
        index = np.random.randint(0, 10, (150, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'MUL'


class TestMinCase4(TestGraphSendUERecvMinOp):

    def set_config(self):
        self.x = np.random.random((10, 8, 5)).astype("float64")
        self.y = np.random.random((15, 8, 1)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'ADD'


class TestMinCase5(TestGraphSendUERecvMinOp):

    def set_config(self):
        self.x = np.random.random((10, 8, 5)).astype("float64")
        self.y = np.random.random((15, 8, 1)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'MUL'


class TestMinCase6(TestGraphSendUERecvMinOp):

    def set_config(self):
        self.x = np.random.random((100, 1)).astype("float64")
        self.y = np.random.random((15, 20)).astype("float64")
        index = np.random.randint(0, 100, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'ADD'


class TestMinCase7(TestGraphSendUERecvMinOp):

    def set_config(self):
        self.x = np.random.random((100, 1)).astype("float64")
        self.y = np.random.random((15, 20)).astype("float64")
        index = np.random.randint(0, 100, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'MUL'


class API_GeometricSendUERecvTest(unittest.TestCase):

    def test_compute_all_with_sum(self):
        paddle.disable_static()
        x = paddle.to_tensor(np.array([[0, 2, 3], [1, 4, 5], [2, 6, 7]]),
                             dtype="float32")
        y = paddle.ones(shape=[4, 1], dtype="float32")
        src_index = paddle.to_tensor(np.array([0, 1, 2, 0]), dtype="int32")
        dst_index = paddle.to_tensor(np.array([1, 2, 1, 0]), dtype="int32")

        res_add = paddle.geometric.send_ue_recv(x, y, src_index, dst_index,
                                                "add", "sum")
        res_sub = paddle.geometric.send_ue_recv(x, y, src_index, dst_index,
                                                "sub", "sum")
        res_mul = paddle.geometric.send_ue_recv(x, y, src_index, dst_index,
                                                "mul", "sum")
        res_div = paddle.geometric.send_ue_recv(x, y, src_index, dst_index,
                                                "div", "sum")
        res = [res_add, res_sub, res_mul, res_div]

        np_add = np.array([[1, 3, 4], [4, 10, 12], [2, 5, 6]], dtype="float32")
        np_sub = np.array([[-1, 1, 2], [0, 6, 8], [0, 3, 4]], dtype="float32")
        np_mul = np.array([[0, 2, 3], [2, 8, 10], [1, 4, 5]], dtype="float32")
        np_div = np.array([[0, 2, 3], [2, 8, 10], [1, 4, 5]], dtype="float32")

        for np_res, paddle_res in zip([np_add, np_sub, np_mul, np_div], res):
            np.testing.assert_allclose(
                np_res,
                paddle_res,
                rtol=1e-05,
                atol=1e-06,
                err_msg='two value is                {}\n{}, check diff!'.
                format(np_res, paddle_res))

    def test_compute_all_with_mean(self):
        paddle.disable_static()
        x = paddle.to_tensor(np.array([[0, 2, 3], [1, 4, 5], [2, 6, 7]]),
                             dtype="float32")
        y = paddle.ones(shape=[4, 1], dtype="float32")
        src_index = paddle.to_tensor(np.array([0, 1, 2, 0]), dtype="int32")
        dst_index = paddle.to_tensor(np.array([1, 2, 1, 0]), dtype="int32")

        res_add = paddle.geometric.send_ue_recv(x, y, src_index, dst_index,
                                                "add", "mean")
        res_sub = paddle.geometric.send_ue_recv(x, y, src_index, dst_index,
                                                "sub", "mean")
        res_mul = paddle.geometric.send_ue_recv(x, y, src_index, dst_index,
                                                "mul", "mean")
        res_div = paddle.geometric.send_ue_recv(x, y, src_index, dst_index,
                                                "div", "mean")
        res = [res_add, res_sub, res_mul, res_div]

        np_add = np.array([[1, 3, 4], [2, 5, 6], [2, 5, 6]], dtype="float32")
        np_sub = np.array([[-1, 1, 2], [0, 3, 4], [0, 3, 4]], dtype="float32")
        np_mul = np.array([[0, 2, 3], [1, 4, 5], [1, 4, 5]], dtype="float32")
        np_div = np.array([[0, 2, 3], [1, 4, 5], [1, 4, 5]], dtype="float32")

        for np_res, paddle_res in zip([np_add, np_sub, np_mul, np_div], res):
            np.testing.assert_allclose(
                np_res,
                paddle_res,
                rtol=1e-05,
                atol=1e-06,
                err_msg='two value is                {}\n{}, check diff!'.
                format(np_res, paddle_res))

    def test_compute_all_with_max(self):
        paddle.disable_static()
        x = paddle.to_tensor(np.array([[0, 2, 3], [1, 4, 5], [2, 6, 7]]),
                             dtype="float32")
        y = paddle.ones(shape=[4, 1], dtype="float32")
        src_index = paddle.to_tensor(np.array([0, 1, 2, 0]), dtype="int32")
        dst_index = paddle.to_tensor(np.array([1, 2, 1, 0]), dtype="int32")

        res_add = paddle.geometric.send_ue_recv(x, y, src_index, dst_index,
                                                "add", "max")
        res_sub = paddle.geometric.send_ue_recv(x, y, src_index, dst_index,
                                                "sub", "max")
        res_mul = paddle.geometric.send_ue_recv(x, y, src_index, dst_index,
                                                "mul", "max")
        res_div = paddle.geometric.send_ue_recv(x, y, src_index, dst_index,
                                                "div", "max")
        res = [res_add, res_sub, res_mul, res_div]

        np_add = np.array([[1, 3, 4], [3, 7, 8], [2, 5, 6]], dtype="float32")
        np_sub = np.array([[-1, 1, 2], [1, 5, 6], [0, 3, 4]], dtype="float32")
        np_mul = np.array([[0, 2, 3], [2, 6, 7], [1, 4, 5]], dtype="float32")
        np_div = np.array([[0, 2, 3], [2, 6, 7], [1, 4, 5]], dtype="float32")

        np.testing.assert_allclose(np_sub, res_sub, rtol=1e-05, atol=1e-06)
        for np_res, paddle_res in zip([np_add, np_sub, np_mul, np_div], res):
            np.testing.assert_allclose(
                np_res,
                paddle_res,
                rtol=1e-05,
                atol=1e-06,
                err_msg='two value is                {}\n{}, check diff!'.
                format(np_res, paddle_res))

    def test_compute_all_with_max_fp16(self):
        paddle.disable_static()
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                x = paddle.to_tensor(np.array([[0, 2, 3], [1, 4, 5], [2, 6,
                                                                      7]]),
                                     dtype="float16")
                y = paddle.ones(shape=[4, 1], dtype="float16")
                src_index = paddle.to_tensor(np.array([0, 1, 2, 0]),
                                             dtype="int32")
                dst_index = paddle.to_tensor(np.array([1, 2, 1, 0]),
                                             dtype="int32")

                res_add = paddle.geometric.send_ue_recv(x, y, src_index,
                                                        dst_index, "add", "max")
                res_sub = paddle.geometric.send_ue_recv(x, y, src_index,
                                                        dst_index, "sub", "max")
                res_mul = paddle.geometric.send_ue_recv(x, y, src_index,
                                                        dst_index, "mul", "max")
                res_div = paddle.geometric.send_ue_recv(x, y, src_index,
                                                        dst_index, "div", "max")
                res = [res_add, res_sub, res_mul, res_div]

                np_add = np.array([[1, 3, 4], [3, 7, 8], [2, 5, 6]],
                                  dtype="float16")
                np_sub = np.array([[-1, 1, 2], [1, 5, 6], [0, 3, 4]],
                                  dtype="float16")
                np_mul = np.array([[0, 2, 3], [2, 6, 7], [1, 4, 5]],
                                  dtype="float16")
                np_div = np.array([[0, 2, 3], [2, 6, 7], [1, 4, 5]],
                                  dtype="float16")

                np.testing.assert_allclose(np_sub,
                                           res_sub,
                                           rtol=1e-05,
                                           atol=1e-06)
                for np_res, paddle_res in zip([np_add, np_sub, np_mul, np_div],
                                              res):
                    np.testing.assert_allclose(
                        np_res,
                        paddle_res,
                        rtol=1e-05,
                        atol=1e-06,
                        err_msg=
                        'two value is                        {}\n{}, check diff!'
                        .format(np_res, paddle_res))

    def test_compute_all_with_min(self):
        paddle.disable_static()
        x = paddle.to_tensor(np.array([[0, 2, 3], [1, 4, 5], [2, 6, 7]]),
                             dtype="float32")
        y = paddle.ones(shape=[4, 1], dtype="float32")
        src_index = paddle.to_tensor(np.array([0, 1, 2, 0]), dtype="int32")
        dst_index = paddle.to_tensor(np.array([1, 2, 1, 0]), dtype="int32")

        res_add = paddle.geometric.send_ue_recv(x, y, src_index, dst_index,
                                                "add", "min")
        res_sub = paddle.geometric.send_ue_recv(x, y, src_index, dst_index,
                                                "sub", "min")
        res_mul = paddle.geometric.send_ue_recv(x, y, src_index, dst_index,
                                                "mul", "min")
        res_div = paddle.geometric.send_ue_recv(x, y, src_index, dst_index,
                                                "div", "min")
        res = [res_add, res_sub, res_mul, res_div]

        np_add = np.array([[1, 3, 4], [1, 3, 4], [2, 5, 6]], dtype="float32")
        np_sub = np.array([[-1, 1, 2], [-1, 1, 2], [0, 3, 4]], dtype="float32")
        np_mul = np.array([[0, 2, 3], [0, 2, 3], [1, 4, 5]], dtype="float32")
        np_div = np.array([[0, 2, 3], [0, 2, 3], [1, 4, 5]], dtype="float32")

        for np_res, paddle_res in zip([np_add, np_sub, np_mul, np_div], res):
            np.testing.assert_allclose(
                np_res,
                paddle_res,
                rtol=1e-05,
                atol=1e-06,
                err_msg='two value is                {}\n{}, check diff!'.
                format(np_res, paddle_res))

    def test_compute_all_with_min_fp16(self):
        paddle.disable_static()
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                x = paddle.to_tensor(np.array([[0, 2, 3], [1, 4, 5], [2, 6,
                                                                      7]]),
                                     dtype="float16")
                y = paddle.ones(shape=[4, 1], dtype="float16")
                src_index = paddle.to_tensor(np.array([0, 1, 2, 0]),
                                             dtype="int32")
                dst_index = paddle.to_tensor(np.array([1, 2, 1, 0]),
                                             dtype="int32")
                res_add = paddle.geometric.send_ue_recv(x, y, src_index,
                                                        dst_index, "add", "min")
                res_sub = paddle.geometric.send_ue_recv(x, y, src_index,
                                                        dst_index, "sub", "min")
                res_mul = paddle.geometric.send_ue_recv(x, y, src_index,
                                                        dst_index, "mul", "min")
                res_div = paddle.geometric.send_ue_recv(x, y, src_index,
                                                        dst_index, "div", "min")
                res = [res_add, res_sub, res_mul, res_div]

                np_add = np.array([[1, 3, 4], [1, 3, 4], [2, 5, 6]],
                                  dtype="float16")
                np_sub = np.array([[-1, 1, 2], [-1, 1, 2], [0, 3, 4]],
                                  dtype="float16")
                np_mul = np.array([[0, 2, 3], [0, 2, 3], [1, 4, 5]],
                                  dtype="float16")
                np_div = np.array([[0, 2, 3], [0, 2, 3], [1, 4, 5]],
                                  dtype="float16")

                for np_res, paddle_res in zip([np_add, np_sub, np_mul, np_div],
                                              res):
                    np.testing.assert_allclose(
                        np_res,
                        paddle_res,
                        rtol=1e-05,
                        atol=1e-06,
                        err_msg=
                        'two value is                        {}\n{}, check diff!'
                        .format(np_res, paddle_res))

    def test_reshape_lhs_rhs(self):
        paddle.disable_static()
        x = paddle.to_tensor(np.array([[0, 2, 3], [1, 4, 5], [2, 6, 7]]),
                             dtype="float32")
        x = x.reshape(shape=[3, 3, 1])
        y = paddle.ones([4, 1], dtype="float32")
        src_index = paddle.to_tensor(np.array([0, 1, 2, 0]), dtype="int32")
        dst_index = paddle.to_tensor(np.array([1, 2, 1, 0]), dtype="int32")
        res_add = paddle.geometric.send_ue_recv(x, y, src_index, dst_index,
                                                "add", "min")
        np_add = np.array([[1, 3, 4], [1, 3, 4], [2, 5, 6]],
                          dtype="float16").reshape([3, 3, 1])
        np.testing.assert_allclose(
            np_add,
            res_add,
            rtol=1e-05,
            atol=1e-06,
            err_msg='two value is                        {}\n{}, check diff!'.
            format(np_add, res_add))

    def test_out_size_tensor_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name="x", shape=[3, 3], dtype="float32")
            y = paddle.static.data(name="y", shape=[3], dtype="float32")
            src_index = paddle.static.data(name="src", shape=[3], dtype="int32")
            dst_index = paddle.static.data(name="dst", shape=[3], dtype="int32")
            out_size = paddle.static.data(name="out_size",
                                          shape=[1],
                                          dtype="int32")

            res_sum = paddle.geometric.send_ue_recv(x, y, src_index, dst_index,
                                                    "mul", "sum", out_size)

            exe = paddle.static.Executor(paddle.CPUPlace())
            data1 = np.array([[0, 2, 3], [1, 4, 5], [2, 6, 6]], dtype="float32")
            data2 = np.array([1, 2, 3], dtype="float32")
            data3 = np.array([0, 0, 1], dtype="int32")
            data4 = np.array([0, 1, 1], dtype="int32")
            data5 = np.array([2], dtype="int32")

            np_sum = np.array([[0, 2, 3], [3, 16, 21]], dtype="float32")

            ret = exe.run(feed={
                'x': data1,
                'y': data2,
                'src': data3,
                'dst': data4,
                'out_size': data5,
            },
                          fetch_list=[res_sum])
        np.testing.assert_allclose(
            np_sum,
            ret[0],
            rtol=1e-05,
            atol=1e-06,
            err_msg='two value is                        {}\n{}, check diff!'.
            format(np_sum, ret[0]))

    def test_api_eager_dygraph(self):
        with _test_eager_guard():
            self.test_compute_all_with_sum()
            self.test_compute_all_with_mean()
            self.test_compute_all_with_max()
            self.test_compute_all_with_max_fp16()
            self.test_compute_all_with_min()
            self.test_compute_all_with_min_fp16()
            self.test_reshape_lhs_rhs()


if __name__ == "__main__":
    unittest.main()
