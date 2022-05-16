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

import paddle
import numpy as np
import unittest
from op_test import OpTest


def test_static_layer(place,
                      input_np,
                      label_np,
                      reduction='mean',):
    paddle.enable_static()
    prog = paddle.static.Program()
    startup_prog = paddle.static.Program()
    with paddle.static.program_guard(prog, startup_prog):
        input = paddle.static.data(
            name='input', shape=input_np.shape, dtype='float64')
        label = paddle.static.data(
            name='label', shape=label_np.shape, dtype='float64')
        sm_loss = paddle.nn.loss.SoftMarginLoss(reduction=reduction)
        res = sm_loss(input, label)
        exe = paddle.static.Executor(place)
        static_result = exe.run(prog,
                                feed={"input": input_np,
                                      "label": label_np},
                                fetch_list=[res])
    return static_result


def test_static_functional(place,
                           input_np,
                           label_np,
                           reduction='mean',):
    paddle.enable_static()
    prog = paddle.static.Program()
    startup_prog = paddle.static.Program()
    with paddle.static.program_guard(prog, startup_prog):
        input = paddle.static.data(
            name='input', shape=input_np.shape, dtype='float64')
        label = paddle.static.data(
            name='label', shape=label_np.shape, dtype='float64')

        res = paddle.nn.functional.soft_margin_loss(
            input, label, reduction=reduction)
        exe = paddle.static.Executor(place)
        static_result = exe.run(prog,
                                feed={"input": input_np,
                                      "label": label_np},
                                fetch_list=[res])
    return static_result


def test_dygraph_layer(place,
                       input_np,
                       label_np,
                       reduction='mean',):
    paddle.disable_static()
    sm_loss = paddle.nn.loss.SoftMarginLoss(reduction=reduction)
    dy_res = sm_loss(paddle.to_tensor(input_np), paddle.to_tensor(label_np))
    dy_result = dy_res.numpy()
    paddle.enable_static()
    return dy_result


def test_dygraph_functional(place,
                            input_np,
                            label_np,
                            reduction='mean',):
    paddle.disable_static()
    input = paddle.to_tensor(input_np)
    label = paddle.to_tensor(label_np)

    dy_res = paddle.nn.functional.soft_margin_loss(
        input, label, reduction=reduction)
    dy_result = dy_res.numpy()
    paddle.enable_static()
    return dy_result


def calc_softmarginloss(input_np, label_np, reduction='mean',):

    expected = np.log(1+np.exp(-label_np * input_np))
    # expected = np.mean(expected, axis=-1)

    if reduction == 'mean':
        expected = np.mean(expected)
    elif reduction == 'sum':
        expected = np.sum(expected)
    else:
        expected = expected

    return expected


class TestSoftMarginLoss(unittest.TestCase):
    def test_SoftMarginLoss(self):
        input_np = np.random.uniform(0.1, 0.8, size=(10, 10)).astype(np.float64)
        label_np = np.random.randint(0, 2, size=(10, 10)).astype(np.float64)
        label_np[label_np==0]=-1
        places = ['cpu']
        if paddle.device.is_compiled_with_cuda():
            places.append('gpu')
        reductions = ['sum', 'mean', 'none']
        for place in places:
            for reduction in reductions:
                static_result = test_static_layer(place, input_np, label_np,
                                                  reduction)
                dy_result = test_dygraph_layer(place, input_np, label_np,
                                               reduction)
                expected = calc_softmarginloss(input_np, label_np, reduction)
                self.assertTrue(np.allclose(static_result, expected))
                self.assertTrue(np.allclose(static_result, dy_result))
                self.assertTrue(np.allclose(dy_result, expected))
                static_functional = test_static_functional(place, input_np,
                                                           label_np, reduction)
                dy_functional = test_dygraph_functional(place, input_np,
                                                        label_np, reduction)
                self.assertTrue(np.allclose(static_functional, expected))
                self.assertTrue(np.allclose(static_functional, dy_functional))
                self.assertTrue(np.allclose(dy_functional, expected))

    def test_SoftMarginLoss_error(self):
        paddle.disable_static()
        self.assertRaises(
            ValueError,
            paddle.nn.loss.SoftMarginLoss,
            reduction="unsupport reduction")
        input = paddle.to_tensor([[0.1, 0.3]], dtype='float32')
        label = paddle.to_tensor([[-1.0, 1.0]], dtype='float32')
        self.assertRaises(
            ValueError,
            paddle.nn.functional.soft_margin_loss,
            input=input,
            label=label,
            reduction="unsupport reduction")
        paddle.enable_static()


def soft_margin_loss(input, label):
    return np.log(1+np.exp(-label * input))


class TestSoftMarginLossOp(OpTest):
    def setUp(self):
        self.init_test_case()
        self.op_type = "soft_margin_loss"
        input_np = np.random.uniform(0.1, 0.8, self.shape).astype("float64")
        label_np = np.random.randint(0, 2, self.shape).astype("float64")
        label_np[label_np==0]=-1
        output_np = soft_margin_loss(input_np, label_np)

        self.inputs = {'X': input_np, 'Label': label_np}
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')

    def init_test_case(self):
        self.shape = [10, 10]


class TestSoftMarginLossOpCase1(OpTest):
    def init_test_cast(self):
        self.shape = [2, 3, 4, 5]


class TestSoftMarginLossOpCase2(OpTest):
    def init_test_cast(self):
        self.shape = [2, 3, 20]


if __name__ == "__main__":
    unittest.main()
