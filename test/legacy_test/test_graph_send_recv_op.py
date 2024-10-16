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
from op_test import OpTest

import paddle


def graph_send_recv_wrapper(
    x, src_index, dst_index, reduce_op="sum", out_size=None, name=None
):
    return paddle.geometric.send_u_recv(
        x, src_index, dst_index, reduce_op.lower(), out_size, name
    )


class TestGraphSendRecvMaxOp(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.python_api = graph_send_recv_wrapper
        self.python_out_sig = ["Out"]
        self.op_type = "graph_send_recv"
        x = np.random.random((10, 20)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        src_index = index[:, 0]
        dst_index = index[:, 1]

        self.inputs = {'X': x, 'Src_index': src_index, 'Dst_index': dst_index}

        self.attrs = {'reduce_op': 'MAX'}

        out, self.gradient = compute_graph_send_recv_for_min_max(
            self.inputs, self.attrs
        )
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', user_defined_grads=[self.gradient], check_pir=True
        )


class TestGraphSendRecvMinOp(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.python_api = graph_send_recv_wrapper
        self.python_out_sig = ["Out"]
        self.op_type = "graph_send_recv"
        x = np.random.random((10, 20)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        src_index = index[:, 0]
        dst_index = index[:, 1]

        self.inputs = {'X': x, 'Src_index': src_index, 'Dst_index': dst_index}

        self.attrs = {'reduce_op': 'MIN'}

        out, self.gradient = compute_graph_send_recv_for_min_max(
            self.inputs, self.attrs
        )

        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', user_defined_grads=[self.gradient], check_pir=True
        )


class TestGraphSendRecvSumOp(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.python_api = graph_send_recv_wrapper
        self.python_out_sig = ["Out"]
        self.op_type = "graph_send_recv"
        x = np.random.random((10, 20)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        src_index = index[:, 0]
        dst_index = index[:, 1]

        self.inputs = {'X': x, 'Src_index': src_index, 'Dst_index': dst_index}

        self.attrs = {'reduce_op': 'SUM'}

        out, _ = compute_graph_send_recv_for_sum_mean(self.inputs, self.attrs)

        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True)


class TestGraphSendRecvMeanOp(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.python_api = graph_send_recv_wrapper
        self.python_out_sig = ["Out"]
        self.op_type = "graph_send_recv"
        x = np.random.random((10, 20)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        src_index = index[:, 0]
        dst_index = index[:, 1]

        self.inputs = {'X': x, 'Src_index': src_index, 'Dst_index': dst_index}

        self.attrs = {'reduce_op': 'MEAN'}

        out, dst_count = compute_graph_send_recv_for_sum_mean(
            self.inputs, self.attrs
        )

        self.outputs = {'Out': out, 'Dst_count': dst_count}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True)


def compute_graph_send_recv_for_sum_mean(inputs, attributes):
    x = inputs['X']
    src_index = inputs['Src_index']
    dst_index = inputs['Dst_index']

    reduce_op = attributes['reduce_op']

    gather_x = x[src_index]
    target_shape = list(x.shape)
    results = np.zeros(target_shape, dtype=x.dtype)
    if reduce_op == 'SUM':
        for index, s_id in enumerate(dst_index):
            results[s_id, :] += gather_x[index, :]
    elif reduce_op == 'MEAN':
        count = np.zeros(target_shape[0], dtype=np.int32)
        for index, s_id in enumerate(dst_index):
            results[s_id, :] += gather_x[index, :]
            count[s_id] += 1
        results = results / count.reshape([-1, 1])
        results[np.isnan(results)] = 0
    else:
        raise ValueError("Invalid reduce_op, only SUM, MEAN supported!")

    count = np.zeros(target_shape[0], dtype=np.int32)
    for index, s_id in enumerate(dst_index):
        count[s_id] += 1

    return results, count


def compute_graph_send_recv_for_min_max(inputs, attributes):
    x = inputs['X']
    src_index = inputs['Src_index']
    dst_index = inputs['Dst_index']

    reduce_op = attributes['reduce_op']

    gather_x = x[src_index]
    target_shape = list(x.shape)
    results = np.zeros(target_shape, dtype=x.dtype)
    gradient = np.zeros_like(x)

    # Calculate forward output
    if reduce_op == "MAX":
        first_set = set()
        for index, s_id in enumerate(dst_index):
            if s_id not in first_set:
                results[s_id, :] += gather_x[index, :]
                first_set.add(s_id)
            else:
                results[s_id, :] = np.maximum(
                    results[s_id, :], gather_x[index, :]
                )
    elif reduce_op == "MIN":
        first_set = set()
        for index, s_id in enumerate(dst_index):
            if s_id not in first_set:
                results[s_id, :] += gather_x[index, :]
                first_set.add(s_id)
            else:
                results[s_id, :] = np.minimum(
                    results[s_id, :], gather_x[index, :]
                )
    else:
        raise ValueError("Invalid reduce_op, only MAX, MIN supported!")

    # Calculate backward gradient
    index_size = len(src_index)
    for i in range(index_size):
        forward_src_idx = src_index[i]
        forward_dst_idx = dst_index[i]
        gradient[forward_src_idx] += 1 * (
            x[forward_src_idx] == results[forward_dst_idx]
        )

    return results, gradient / results.size


class API_GraphSendRecvOpTest(unittest.TestCase):

    def test_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name="x", shape=[3, 3], dtype="float32")
            src_index = paddle.static.data(name="src", shape=[4], dtype="int32")
            dst_index = paddle.static.data(name="dst", shape=[4], dtype="int32")

            res_sum = paddle.incubate.graph_send_recv(
                x, src_index, dst_index, "sum"
            )
            res_mean = paddle.incubate.graph_send_recv(
                x, src_index, dst_index, "mean"
            )
            res_max = paddle.incubate.graph_send_recv(
                x, src_index, dst_index, "max"
            )
            res_min = paddle.incubate.graph_send_recv(
                x, src_index, dst_index, "min"
            )

            exe = paddle.static.Executor(paddle.CPUPlace())
            data1 = np.array([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype='float32')
            data2 = np.array([0, 1, 2, 0], dtype="int32")
            data3 = np.array([1, 2, 1, 0], dtype="int32")

            np_sum = np.array(
                [[0, 2, 3], [2, 8, 10], [1, 4, 5]], dtype="float32"
            )
            np_mean = np.array(
                [[0, 2, 3], [1, 4, 5], [1, 4, 5]], dtype="float32"
            )
            np_max = np.array(
                [[0, 2, 3], [2, 6, 7], [1, 4, 5]], dtype="float32"
            )
            np_min = np.array(
                [[0, 2, 3], [0, 2, 3], [1, 4, 5]], dtype="float32"
            )

            ret = exe.run(
                feed={'x': data1, 'src': data2, 'dst': data3},
                fetch_list=[res_sum, res_mean, res_max, res_min],
            )

        for np_res, ret_res in zip([np_sum, np_mean, np_max, np_min], ret):
            np.testing.assert_allclose(np_res, ret_res, rtol=1e-05, atol=1e-06)

    def test_dygraph(self):
        paddle.disable_static()
        x = paddle.to_tensor(
            np.array([[0, 2, 3], [1, 4, 5], [2, 6, 7]]), dtype="float32"
        )
        src_index = paddle.to_tensor(np.array([0, 1, 2, 0]), dtype="int32")
        dst_index = paddle.to_tensor(np.array([1, 2, 1, 0]), dtype="int32")
        res_sum = paddle.incubate.graph_send_recv(
            x, src_index, dst_index, "sum"
        )
        res_mean = paddle.incubate.graph_send_recv(
            x, src_index, dst_index, "mean"
        )
        res_max = paddle.incubate.graph_send_recv(
            x, src_index, dst_index, "max"
        )
        res_min = paddle.incubate.graph_send_recv(
            x, src_index, dst_index, "min"
        )

        np_sum = np.array([[0, 2, 3], [2, 8, 10], [1, 4, 5]], dtype="float32")
        np_mean = np.array([[0, 2, 3], [1, 4, 5], [1, 4, 5]], dtype="float32")
        np_max = np.array([[0, 2, 3], [2, 6, 7], [1, 4, 5]], dtype="float32")
        np_min = np.array([[0, 2, 3], [0, 2, 3], [1, 4, 5]], dtype="float32")

        ret = [res_sum, res_mean, res_max, res_min]

        for np_res, ret_res in zip([np_sum, np_mean, np_max, np_min], ret):
            np.testing.assert_allclose(np_res, ret_res, rtol=1e-05, atol=1e-06)

    def test_int32_input(self):
        paddle.disable_static()
        x = paddle.to_tensor(
            np.array([[0, 2, 3], [1, 4, 5], [2, 6, 6]]), dtype="int32"
        )
        src_index = paddle.to_tensor(np.array([0, 1, 2, 0, 1]), dtype="int32")
        dst_index = paddle.to_tensor(np.array([1, 2, 1, 0, 1]), dtype="int32")
        res_sum = paddle.incubate.graph_send_recv(
            x, src_index, dst_index, "sum"
        )
        res_mean = paddle.incubate.graph_send_recv(
            x, src_index, dst_index, "mean"
        )
        res_max = paddle.incubate.graph_send_recv(
            x, src_index, dst_index, "max"
        )
        res_min = paddle.incubate.graph_send_recv(
            x, src_index, dst_index, "min"
        )

        np_sum = np.array([[0, 2, 3], [3, 12, 14], [1, 4, 5]], dtype="int32")
        np_mean = np.array([[0, 2, 3], [1, 4, 4], [1, 4, 5]], dtype="int32")
        np_max = np.array([[0, 2, 3], [2, 6, 6], [1, 4, 5]], dtype="int32")
        np_min = np.array([[0, 2, 3], [0, 2, 3], [1, 4, 5]], dtype="int32")

        ret = [res_sum, res_mean, res_max, res_min]

        for np_res, ret_res in zip([np_sum, np_mean, np_max, np_min], ret):
            np.testing.assert_allclose(np_res, ret_res, rtol=1e-05, atol=1e-06)

    def test_set_outsize_gpu(self):
        paddle.disable_static()
        x = paddle.to_tensor(
            np.array([[0, 2, 3], [1, 4, 5], [2, 6, 6]]), dtype="float32"
        )
        src_index = paddle.to_tensor(np.array([0, 0, 1]), dtype="int32")
        dst_index = paddle.to_tensor(np.array([0, 1, 1]), dtype="int32")
        res = paddle.incubate.graph_send_recv(x, src_index, dst_index, "sum")
        out_size = paddle.max(dst_index) + 1
        res_set_outsize = paddle.incubate.graph_send_recv(
            x, src_index, dst_index, "sum", out_size
        )

        np_res = np.array([[0, 2, 3], [1, 6, 8], [0, 0, 0]], dtype="float32")
        np_res_set_outsize = np.array([[0, 2, 3], [1, 6, 8]], dtype="float32")

        np.testing.assert_allclose(np_res, res, rtol=1e-05, atol=1e-06)
        np.testing.assert_allclose(
            np_res_set_outsize, res_set_outsize, rtol=1e-05, atol=1e-06
        )

    def test_out_size_tensor_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name="x", shape=[3, 3], dtype="float32")
            src_index = paddle.static.data(name="src", shape=[3], dtype="int32")
            dst_index = paddle.static.data(name="dst", shape=[3], dtype="int32")
            out_size = paddle.static.data(
                name="out_size", shape=[1], dtype="int32"
            )

            res_sum = paddle.incubate.graph_send_recv(
                x, src_index, dst_index, "sum", out_size
            )

            exe = paddle.static.Executor(paddle.CPUPlace())
            data1 = np.array([[0, 2, 3], [1, 4, 5], [2, 6, 6]], dtype='float32')
            data2 = np.array([0, 0, 1], dtype="int32")
            data3 = np.array([0, 1, 1], dtype="int32")
            data4 = np.array([2], dtype="int32")

            np_sum = np.array([[0, 2, 3], [1, 6, 8]], dtype="float32")

            ret = exe.run(
                feed={
                    'x': data1,
                    'src': data2,
                    'dst': data3,
                    'out_size': data4,
                },
                fetch_list=[res_sum],
            )
        np.testing.assert_allclose(np_sum, ret[0], rtol=1e-05, atol=1e-06)


class API_GeometricSendURecvTest(unittest.TestCase):

    def test_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name="x", shape=[3, 3], dtype="float32")
            src_index = paddle.static.data(name="src", shape=[4], dtype="int32")
            dst_index = paddle.static.data(name="dst", shape=[4], dtype="int32")

            res_sum = paddle.geometric.send_u_recv(
                x, src_index, dst_index, "sum"
            )
            res_mean = paddle.geometric.send_u_recv(
                x, src_index, dst_index, "mean"
            )
            res_max = paddle.geometric.send_u_recv(
                x, src_index, dst_index, "max"
            )
            res_min = paddle.geometric.send_u_recv(
                x, src_index, dst_index, "min"
            )

            exe = paddle.static.Executor(paddle.CPUPlace())
            data1 = np.array([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype='float32')
            data2 = np.array([0, 1, 2, 0], dtype="int32")
            data3 = np.array([1, 2, 1, 0], dtype="int32")

            np_sum = np.array(
                [[0, 2, 3], [2, 8, 10], [1, 4, 5]], dtype="float32"
            )
            np_mean = np.array(
                [[0, 2, 3], [1, 4, 5], [1, 4, 5]], dtype="float32"
            )
            np_max = np.array(
                [[0, 2, 3], [2, 6, 7], [1, 4, 5]], dtype="float32"
            )
            np_min = np.array(
                [[0, 2, 3], [0, 2, 3], [1, 4, 5]], dtype="float32"
            )

            ret = exe.run(
                feed={'x': data1, 'src': data2, 'dst': data3},
                fetch_list=[res_sum, res_mean, res_max, res_min],
            )

        for np_res, ret_res in zip([np_sum, np_mean, np_max, np_min], ret):
            np.testing.assert_allclose(np_res, ret_res, rtol=1e-05, atol=1e-06)

    def test_dygraph(self):
        paddle.disable_static()
        x = paddle.to_tensor(
            np.array([[0, 2, 3], [1, 4, 5], [2, 6, 7]]), dtype="float32"
        )
        src_index = paddle.to_tensor(np.array([0, 1, 2, 0]), dtype="int32")
        dst_index = paddle.to_tensor(np.array([1, 2, 1, 0]), dtype="int32")
        res_sum = paddle.geometric.send_u_recv(x, src_index, dst_index, "sum")
        res_mean = paddle.geometric.send_u_recv(x, src_index, dst_index, "mean")
        res_max = paddle.geometric.send_u_recv(x, src_index, dst_index, "max")
        res_min = paddle.geometric.send_u_recv(x, src_index, dst_index, "min")

        np_sum = np.array([[0, 2, 3], [2, 8, 10], [1, 4, 5]], dtype="float32")
        np_mean = np.array([[0, 2, 3], [1, 4, 5], [1, 4, 5]], dtype="float32")
        np_max = np.array([[0, 2, 3], [2, 6, 7], [1, 4, 5]], dtype="float32")
        np_min = np.array([[0, 2, 3], [0, 2, 3], [1, 4, 5]], dtype="float32")

        ret = [res_sum, res_mean, res_max, res_min]

        for np_res, ret_res in zip([np_sum, np_mean, np_max, np_min], ret):
            np.testing.assert_allclose(np_res, ret_res, rtol=1e-05, atol=1e-06)

    def test_int32_input(self):
        paddle.disable_static()
        x = paddle.to_tensor(
            np.array([[0, 2, 3], [1, 4, 5], [2, 6, 6]]), dtype="int32"
        )
        src_index = paddle.to_tensor(np.array([0, 1, 2, 0, 1]), dtype="int32")
        dst_index = paddle.to_tensor(np.array([1, 2, 1, 0, 1]), dtype="int32")
        res_sum = paddle.geometric.send_u_recv(x, src_index, dst_index, "sum")
        res_mean = paddle.geometric.send_u_recv(x, src_index, dst_index, "mean")
        res_max = paddle.geometric.send_u_recv(x, src_index, dst_index, "max")
        res_min = paddle.geometric.send_u_recv(x, src_index, dst_index, "min")

        np_sum = np.array([[0, 2, 3], [3, 12, 14], [1, 4, 5]], dtype="int32")
        np_mean = np.array([[0, 2, 3], [1, 4, 4], [1, 4, 5]], dtype="int32")
        np_max = np.array([[0, 2, 3], [2, 6, 6], [1, 4, 5]], dtype="int32")
        np_min = np.array([[0, 2, 3], [0, 2, 3], [1, 4, 5]], dtype="int32")

        ret = [res_sum, res_mean, res_max, res_min]

        for np_res, ret_res in zip([np_sum, np_mean, np_max, np_min], ret):
            np.testing.assert_allclose(np_res, ret_res, rtol=1e-05, atol=1e-06)

    def test_set_outsize_gpu(self):
        paddle.disable_static()
        x = paddle.to_tensor(
            np.array([[0, 2, 3], [1, 4, 5], [2, 6, 6]]), dtype="float32"
        )
        src_index = paddle.to_tensor(np.array([0, 0, 1]), dtype="int32")
        dst_index = paddle.to_tensor(np.array([0, 1, 1]), dtype="int32")
        res = paddle.geometric.send_u_recv(x, src_index, dst_index, "sum")
        out_size = paddle.max(dst_index) + 1
        res_set_outsize = paddle.geometric.send_u_recv(
            x, src_index, dst_index, "sum", out_size
        )

        np_res = np.array([[0, 2, 3], [1, 6, 8], [0, 0, 0]], dtype="float32")
        np_res_set_outsize = np.array([[0, 2, 3], [1, 6, 8]], dtype="float32")

        np.testing.assert_allclose(np_res, res, rtol=1e-05, atol=1e-06)
        np.testing.assert_allclose(
            np_res_set_outsize, res_set_outsize, rtol=1e-05, atol=1e-06
        )

    def test_out_size_tensor_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name="x", shape=[3, 3], dtype="float32")
            src_index = paddle.static.data(name="src", shape=[3], dtype="int32")
            dst_index = paddle.static.data(name="dst", shape=[3], dtype="int32")
            out_size = paddle.static.data(
                name="out_size", shape=[1], dtype="int32"
            )

            res_sum = paddle.geometric.send_u_recv(
                x, src_index, dst_index, "sum", out_size
            )

            exe = paddle.static.Executor(paddle.CPUPlace())
            data1 = np.array([[0, 2, 3], [1, 4, 5], [2, 6, 6]], dtype='float32')
            data2 = np.array([0, 0, 1], dtype="int32")
            data3 = np.array([0, 1, 1], dtype="int32")
            data4 = np.array([2], dtype="int32")

            np_sum = np.array([[0, 2, 3], [1, 6, 8]], dtype="float32")

            ret = exe.run(
                feed={
                    'x': data1,
                    'src': data2,
                    'dst': data3,
                    'out_size': data4,
                },
                fetch_list=[res_sum],
            )
        np.testing.assert_allclose(np_sum, ret[0], rtol=1e-05, atol=1e-06)


if __name__ == '__main__':
    unittest.main()
