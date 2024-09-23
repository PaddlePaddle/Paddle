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
from op_test import OpTest

import paddle


def compute_graph_send_uv(inputs, attributes):
    x = inputs['x']
    y = inputs['y']
    src_index = inputs['src_index']
    dst_index = inputs['dst_index']
    message_op = attributes['message_op']

    gather_x = x[src_index]
    gather_y = y[dst_index]

    # Calculate forward output.
    if message_op == "ADD":
        results = gather_x + gather_y
    elif message_op == "MUL":
        results = gather_x * gather_y

    return results


def graph_send_uv_wrapper(x, y, src_index, dst_index, message_op="add"):
    return paddle.geometric.send_uv(
        x, y, src_index, dst_index, message_op.lower()
    )


class TestGraphSendUVOp(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.python_api = graph_send_uv_wrapper
        self.python_out_sig = ['out']
        self.op_type = "graph_send_uv"
        self.set_config()
        self.inputs = {
            'x': self.x,
            'y': self.y,
            'src_index': self.src_index,
            'dst_index': self.dst_index,
        }
        self.attrs = {'message_op': self.message_op}
        out = compute_graph_send_uv(self.inputs, self.attrs)
        self.outputs = {'out': out}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['x', 'y'], 'out', check_pir=True)

    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.y = np.random.random((10, 20)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'ADD'


class TestCase1(TestGraphSendUVOp):
    def set_config(self):
        self.x = np.random.random((10, 20)).astype("float64")
        self.y = np.random.random((10, 20)).astype("float64")
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'MUL'


class TestCase2(TestGraphSendUVOp):
    def set_config(self):
        self.x = np.random.random((100, 1)).astype("float64")
        self.y = np.random.random((100, 20)).astype("float64")
        index = np.random.randint(0, 100, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'ADD'


class TestCase3(TestGraphSendUVOp):
    def set_config(self):
        self.x = np.random.random((100, 20)).astype("float64")
        self.y = np.random.random((100, 1)).astype("float64")
        index = np.random.randint(0, 100, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'ADD'


class TestCase4(TestGraphSendUVOp):
    def set_config(self):
        self.x = np.random.random((100, 1)).astype("float64")
        self.y = np.random.random((100, 20)).astype("float64")
        index = np.random.randint(0, 100, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'MUL'


class TestCase5(TestGraphSendUVOp):
    def set_config(self):
        self.x = np.random.random((100, 20)).astype("float64")
        self.y = np.random.random((100, 1)).astype("float64")
        index = np.random.randint(0, 100, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'MUL'


class TestCase6(TestGraphSendUVOp):
    def set_config(self):
        self.x = np.random.random((10, 10, 1)).astype("float64")
        self.y = np.random.random((10, 10, 10))
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'ADD'


class TestCase7(TestGraphSendUVOp):
    def set_config(self):
        self.x = np.random.random((10, 10, 1)).astype("float64")
        self.y = np.random.random((10, 10, 10))
        index = np.random.randint(0, 10, (15, 2)).astype(np.int64)
        self.src_index = index[:, 0]
        self.dst_index = index[:, 1]
        self.message_op = 'MUL'


class API_GeometricSendUVTest(unittest.TestCase):
    def test_compute_all_dygraph(self):
        paddle.disable_static()
        x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")
        y = paddle.to_tensor([[1, 1, 2], [2, 3, 4], [4, 5, 6]], dtype="float32")
        src_index = paddle.to_tensor(np.array([0, 1, 2, 0]), dtype="int32")
        dst_index = paddle.to_tensor(np.array([1, 2, 1, 0]), dtype="int32")

        res_add = paddle.geometric.send_uv(
            x, y, src_index, dst_index, message_op="add"
        )
        res_sub = paddle.geometric.send_uv(
            x, y, src_index, dst_index, message_op="sub"
        )
        res_mul = paddle.geometric.send_uv(
            x, y, src_index, dst_index, message_op="mul"
        )
        res_div = paddle.geometric.send_uv(
            x, y, src_index, dst_index, message_op="div"
        )
        res = [res_add, res_sub, res_mul, res_div]

        np_add = np.array(
            [[2, 5, 7], [5, 9, 11], [4, 9, 11], [1, 3, 5]], dtype="float32"
        )
        np_sub = np.array(
            [[-2, -1, -1], [-3, -1, -1], [0, 3, 3], [-1, 1, 1]], dtype="float32"
        )
        np_mul = np.array(
            [[0, 6, 12], [4, 20, 30], [4, 18, 28], [0, 2, 6]], dtype="float32"
        )
        np_div = np.array(
            [[0, 2 / 3, 0.75], [0.25, 0.8, 5 / 6], [1, 2, 7 / 4], [0, 2, 1.5]],
            dtype="float32",
        )

        for np_res, paddle_res in zip([np_add, np_sub, np_mul, np_div], res):
            np.testing.assert_allclose(
                np_res,
                paddle_res,
                rtol=1e-05,
                atol=1e-06,
                err_msg=f'two value is                {np_res}\n{paddle_res}, check diff!',
            )

    def test_compute_all_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name="x", shape=[3, 3], dtype="float32")
            y = paddle.static.data(name="y", shape=[3, 3], dtype="float32")
            src_index = paddle.static.data(name="src", shape=[4], dtype="int32")
            dst_index = paddle.static.data(name="dst", shape=[4], dtype="int32")
            res_add = paddle.geometric.send_uv(
                x, y, src_index, dst_index, message_op="add"
            )
            res_sub = paddle.geometric.send_uv(
                x, y, src_index, dst_index, message_op="sub"
            )
            res_mul = paddle.geometric.send_uv(
                x, y, src_index, dst_index, message_op="mul"
            )
            res_div = paddle.geometric.send_uv(
                x, y, src_index, dst_index, message_op="div"
            )

            exe = paddle.static.Executor(paddle.CPUPlace())
            data1 = np.array([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")
            data2 = np.array([[1, 1, 2], [2, 3, 4], [4, 5, 6]], dtype="float32")
            data3 = np.array([0, 1, 2, 0], dtype="int32")
            data4 = np.array([1, 2, 1, 0], dtype="int32")

            np_add = np.array(
                [[2, 5, 7], [5, 9, 11], [4, 9, 11], [1, 3, 5]], dtype="float32"
            )
            np_sub = np.array(
                [[-2, -1, -1], [-3, -1, -1], [0, 3, 3], [-1, 1, 1]],
                dtype="float32",
            )
            np_mul = np.array(
                [[0, 6, 12], [4, 20, 30], [4, 18, 28], [0, 2, 6]],
                dtype="float32",
            )
            np_div = np.array(
                [
                    [0, 2 / 3, 0.75],
                    [0.25, 0.8, 5 / 6],
                    [1, 2, 7 / 4],
                    [0, 2, 1.5],
                ],
                dtype="float32",
            )

            ret = exe.run(
                feed={
                    'x': data1,
                    'y': data2,
                    'src': data3,
                    'dst': data4,
                },
                fetch_list=[res_add, res_sub, res_mul, res_div],
            )
            for np_res, paddle_res in zip(
                [np_add, np_sub, np_mul, np_div], ret
            ):
                np.testing.assert_allclose(
                    np_res,
                    paddle_res,
                    rtol=1e-05,
                    atol=1e-06,
                    err_msg=f'two value is                    {np_res}\n{paddle_res}, check diff!',
                )


if __name__ == "__main__":
    unittest.main()
