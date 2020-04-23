# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import numpy as np
import unittest
from op_test import OpTest


class TestBCELoss(unittest.TestCase):
    def test_BCELoss(self):
        input_np = np.random.random(size=(20, 30)).astype(np.float64)
        label_np = np.random.random(size=(20, 30)).astype(np.float64)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        reductions = ['sum', 'mean', 'none']
        for place in places:
            for red in reductions:
                with fluid.program_guard(prog, startup_prog):
                    input = fluid.data(
                        name='input', shape=[None, 30], dtype='float64')
                    label = fluid.data(
                        name='label', shape=[None, 30], dtype='float64')
                    bce_loss = paddle.nn.loss.BCELoss(reduction=red)
                    res = bce_loss(input, label)

                    exe = fluid.Executor(place)
                    static_result = exe.run(
                        prog,
                        feed={"input": input_np,
                              "label": label_np},
                        fetch_list=[res])

                with fluid.dygraph.guard():
                    bce_loss = paddle.nn.loss.BCELoss(reduction=red)
                    dy_res = bce_loss(
                        fluid.dygraph.to_variable(input_np),
                        fluid.dygraph.to_variable(label_np))
                    dy_result = dy_res.numpy()

                expected = -1 * (label_np * np.log(input_np) +
                                 (1. - label_np) * np.log(1. - input_np))
                if red == 'mean':
                    expected = np.mean(expected)
                elif red == 'sum':
                    expected = np.sum(expected)
                else:
                    expected = expected
                self.assertTrue(np.allclose(static_result, expected))
                self.assertTrue(np.allclose(static_result, dy_result))
                self.assertTrue(np.allclose(dy_result, expected))

    def test_BCELoss_weight(self):
        input_np = np.random.random(size=(2, 3, 4, 10)).astype(np.float64)
        label_np = np.random.random(size=(2, 3, 4, 10)).astype(np.float64)
        weight_np = np.random.random(size=(3, 4, 10)).astype(np.float64)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(
                name='input', shape=[None, 3, 4, 10], dtype='float64')
            label = fluid.data(
                name='label', shape=[None, 3, 4, 10], dtype='float64')
            weight = fluid.data(
                name='weight', shape=[3, 4, 10], dtype='float64')
            bce_loss = paddle.nn.loss.BCELoss(weight=weight)
            res = bce_loss(input, label)

            exe = fluid.Executor(place)
            static_result = exe.run(prog,
                                    feed={
                                        "input": input_np,
                                        "label": label_np,
                                        "weight": weight_np
                                    },
                                    fetch_list=[res])

        with fluid.dygraph.guard():
            bce_loss = paddle.nn.loss.BCELoss(
                weight=fluid.dygraph.to_variable(weight_np))
            dy_res = bce_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_result = dy_res.numpy()

        expected = np.mean(-1 * weight_np *
                           (label_np * np.log(input_np) +
                            (1. - label_np) * np.log(1. - input_np)))
        self.assertTrue(np.allclose(static_result, expected))
        self.assertTrue(np.allclose(static_result, dy_result))
        self.assertTrue(np.allclose(dy_result, expected))


def bce_loss(input, label):
    return -1 * (label * np.log(input) + (1. - label) * np.log(1. - input))


class TestBceLossOp(OpTest):
    def setUp(self):
        self.init_test_case()
        self.op_type = "bce_loss"
        input_np = np.random.uniform(0.1, 0.8, self.shape).astype("float64")
        label_np = np.random.randint(0, 2, self.shape).astype("float64")
        output_np = bce_loss(input_np, label_np)

        self.inputs = {'X': input_np, 'Label': label_np}
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')

    def init_test_case(self):
        self.shape = [10, 10]


class TestBceLossOpCase1(OpTest):
    def init_test_cast(self):
        self.shape = [2, 3, 4, 5]


class TestBceLossOpCase2(OpTest):
    def init_test_cast(self):
        self.shape = [2, 3, 20]


if __name__ == "__main__":
    unittest.main()
