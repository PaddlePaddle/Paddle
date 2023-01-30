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

<<<<<<< HEAD
=======
from __future__ import print_function, division

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import paddle
import paddle.fluid as fluid
import numpy as np
import unittest
import sys

sys.path.append("..")
from op_test import OpTest

paddle.enable_static()


<<<<<<< HEAD
def test_static_layer(
    place, input_np, label_np, reduction='mean', weight_np=None
):
    prog = paddle.static.Program()
    startup_prog = paddle.static.Program()
    with paddle.static.program_guard(prog, startup_prog):
        input = paddle.fluid.data(
            name='input', shape=input_np.shape, dtype='float32'
        )
        label = paddle.fluid.data(
            name='label', shape=label_np.shape, dtype='float32'
        )
        if weight_np is not None:
            weight = paddle.fluid.data(
                name='weight', shape=weight_np.shape, dtype='float32'
            )
            bce_loss = paddle.nn.loss.BCELoss(
                weight=weight, reduction=reduction
            )
=======
def test_static_layer(place,
                      input_np,
                      label_np,
                      reduction='mean',
                      weight_np=None):
    prog = paddle.static.Program()
    startup_prog = paddle.static.Program()
    with paddle.static.program_guard(prog, startup_prog):
        input = paddle.fluid.data(name='input',
                                  shape=input_np.shape,
                                  dtype='float32')
        label = paddle.fluid.data(name='label',
                                  shape=label_np.shape,
                                  dtype='float32')
        if weight_np is not None:
            weight = paddle.fluid.data(name='weight',
                                       shape=weight_np.shape,
                                       dtype='float32')
            bce_loss = paddle.nn.loss.BCELoss(weight=weight,
                                              reduction=reduction)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        else:
            bce_loss = paddle.nn.loss.BCELoss(reduction=reduction)
        res = bce_loss(input, label)
        exe = paddle.static.Executor(place)
<<<<<<< HEAD
        static_result = exe.run(
            prog,
            feed={"input": input_np, "label": label_np}
            if weight_np is None
            else {"input": input_np, "label": label_np, "weight": weight_np},
            fetch_list=[res],
        )
    return static_result[0]


def test_static_functional(
    place, input_np, label_np, reduction='mean', weight_np=None
):
    prog = paddle.static.Program()
    startup_prog = paddle.static.Program()
    with paddle.static.program_guard(prog, startup_prog):
        input = paddle.fluid.data(
            name='input', shape=input_np.shape, dtype='float32'
        )
        label = paddle.fluid.data(
            name='label', shape=label_np.shape, dtype='float32'
        )
        if weight_np is not None:
            weight = paddle.fluid.data(
                name='weight', shape=weight_np.shape, dtype='float32'
            )
            res = paddle.nn.functional.binary_cross_entropy(
                input, label, weight=weight, reduction=reduction
            )
        else:
            res = paddle.nn.functional.binary_cross_entropy(
                input, label, reduction=reduction
            )
        exe = paddle.static.Executor(place)
        static_result = exe.run(
            prog,
            feed={"input": input_np, "label": label_np}
            if weight_np is None
            else {"input": input_np, "label": label_np, "weight": weight_np},
            fetch_list=[res],
        )
    return static_result[0]


def test_dygraph_layer(
    place, input_np, label_np, reduction='mean', weight_np=None
):
=======
        static_result = exe.run(prog,
                                feed={
                                    "input": input_np,
                                    "label": label_np
                                } if weight_np is None else {
                                    "input": input_np,
                                    "label": label_np,
                                    "weight": weight_np
                                },
                                fetch_list=[res])
    return static_result[0]


def test_static_functional(place,
                           input_np,
                           label_np,
                           reduction='mean',
                           weight_np=None):
    prog = paddle.static.Program()
    startup_prog = paddle.static.Program()
    with paddle.static.program_guard(prog, startup_prog):
        input = paddle.fluid.data(name='input',
                                  shape=input_np.shape,
                                  dtype='float32')
        label = paddle.fluid.data(name='label',
                                  shape=label_np.shape,
                                  dtype='float32')
        if weight_np is not None:
            weight = paddle.fluid.data(name='weight',
                                       shape=weight_np.shape,
                                       dtype='float32')
            res = paddle.nn.functional.binary_cross_entropy(input,
                                                            label,
                                                            weight=weight,
                                                            reduction=reduction)
        else:
            res = paddle.nn.functional.binary_cross_entropy(input,
                                                            label,
                                                            reduction=reduction)
        exe = paddle.static.Executor(place)
        static_result = exe.run(prog,
                                feed={
                                    "input": input_np,
                                    "label": label_np
                                } if weight_np is None else {
                                    "input": input_np,
                                    "label": label_np,
                                    "weight": weight_np
                                },
                                fetch_list=[res])
    return static_result[0]


def test_dygraph_layer(place,
                       input_np,
                       label_np,
                       reduction='mean',
                       weight_np=None):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    paddle.disable_static(place)
    if weight_np is not None:
        weight = paddle.to_tensor(weight_np)
        bce_loss = paddle.nn.loss.BCELoss(weight=weight, reduction=reduction)
    else:
        bce_loss = paddle.nn.loss.BCELoss(reduction=reduction)
    dy_res = bce_loss(paddle.to_tensor(input_np), paddle.to_tensor(label_np))
    dy_result = dy_res.numpy()
    paddle.enable_static()
    return dy_result


<<<<<<< HEAD
def test_dygraph_functional(
    place, input_np, label_np, reduction='mean', weight_np=None
):
=======
def test_dygraph_functional(place,
                            input_np,
                            label_np,
                            reduction='mean',
                            weight_np=None):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    paddle.disable_static(place)
    input = paddle.to_tensor(input_np)
    label = paddle.to_tensor(label_np)

    if weight_np is not None:
        weight = paddle.to_tensor(weight_np)
<<<<<<< HEAD
        dy_res = paddle.nn.functional.binary_cross_entropy(
            input, label, weight=weight, reduction=reduction
        )
    else:
        dy_res = paddle.nn.functional.binary_cross_entropy(
            input, label, reduction=reduction
        )
=======
        dy_res = paddle.nn.functional.binary_cross_entropy(input,
                                                           label,
                                                           weight=weight,
                                                           reduction=reduction)
    else:
        dy_res = paddle.nn.functional.binary_cross_entropy(input,
                                                           label,
                                                           reduction=reduction)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    dy_result = dy_res.numpy()
    paddle.enable_static()
    return dy_result


def calc_bceloss(input_np, label_np, reduction='mean', weight_np=None):
    if weight_np is None:
<<<<<<< HEAD
        expected = -1 * (
            label_np * np.log(input_np)
            + (1.0 - label_np) * np.log(1.0 - input_np)
        )
    else:
        expected = (
            -1
            * weight_np
            * (
                label_np * np.log(input_np)
                + (1.0 - label_np) * np.log(1.0 - input_np)
            )
        )
=======
        expected = -1 * (label_np * np.log(input_np) +
                         (1. - label_np) * np.log(1. - input_np))
    else:
        expected = -1 * weight_np * (label_np * np.log(input_np) +
                                     (1. - label_np) * np.log(1. - input_np))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    if reduction == 'mean':
        expected = np.mean(expected)
    elif reduction == 'sum':
        expected = np.sum(expected)
    else:
        expected = expected

    return expected


class TestBCELoss(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_BCELoss(self):
        input_np = np.random.uniform(0.1, 0.8, size=(20, 30)).astype(np.float32)
        label_np = np.random.randint(0, 2, size=(20, 30)).astype(np.float32)
        places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_npu():
            places.append(fluid.NPUPlace(0))
        reductions = ['sum', 'mean', 'none']
        for place in places:
            for reduction in reductions:
<<<<<<< HEAD
                static_result = test_static_layer(
                    place, input_np, label_np, reduction
                )
                dy_result = test_dygraph_layer(
                    place, input_np, label_np, reduction
                )
=======
                static_result = test_static_layer(place, input_np, label_np,
                                                  reduction)
                dy_result = test_dygraph_layer(place, input_np, label_np,
                                               reduction)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                expected = calc_bceloss(input_np, label_np, reduction)
                np.testing.assert_allclose(static_result, expected, rtol=1e-6)
                np.testing.assert_allclose(static_result, dy_result)
                np.testing.assert_allclose(dy_result, expected, rtol=1e-6)
                static_functional = test_static_functional(
<<<<<<< HEAD
                    place, input_np, label_np, reduction
                )
                dy_functional = test_dygraph_functional(
                    place, input_np, label_np, reduction
                )
                np.testing.assert_allclose(
                    static_functional, expected, rtol=1e-6
                )
=======
                    place, input_np, label_np, reduction)
                dy_functional = test_dygraph_functional(place, input_np,
                                                        label_np, reduction)
                np.testing.assert_allclose(static_functional,
                                           expected,
                                           rtol=1e-6)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                np.testing.assert_allclose(static_functional, dy_functional)
                np.testing.assert_allclose(dy_functional, expected, rtol=1e-6)

    def test_BCELoss_weight(self):
<<<<<<< HEAD
        input_np = np.random.uniform(0.1, 0.8, size=(2, 3, 4, 10)).astype(
            np.float32
        )
        label_np = np.random.randint(0, 2, size=(2, 3, 4, 10)).astype(
            np.float32
        )
        weight_np = np.random.random(size=(3, 4, 10)).astype(np.float32)
        place = (
            fluid.NPUPlace(0)
            if fluid.core.is_compiled_with_npu()
            else fluid.CPUPlace()
        )
        for reduction in ['sum', 'mean', 'none']:
            static_result = test_static_layer(
                place, input_np, label_np, reduction, weight_np=weight_np
            )
            dy_result = test_dygraph_layer(
                place, input_np, label_np, reduction, weight_np=weight_np
            )
            expected = calc_bceloss(
                input_np, label_np, reduction, weight_np=weight_np
            )
            np.testing.assert_allclose(static_result, expected, rtol=1e-6)
            np.testing.assert_allclose(static_result, dy_result, rtol=1e-6)
            np.testing.assert_allclose(dy_result, expected, rtol=1e-6)
            static_functional = test_static_functional(
                place, input_np, label_np, reduction, weight_np=weight_np
            )
            dy_functional = test_dygraph_functional(
                place, input_np, label_np, reduction, weight_np=weight_np
            )
            np.testing.assert_allclose(static_functional, expected, rtol=1e-6)
            np.testing.assert_allclose(
                static_functional, dy_functional, rtol=1e-6
            )
=======
        input_np = np.random.uniform(0.1, 0.8,
                                     size=(2, 3, 4, 10)).astype(np.float32)
        label_np = np.random.randint(0, 2,
                                     size=(2, 3, 4, 10)).astype(np.float32)
        weight_np = np.random.random(size=(3, 4, 10)).astype(np.float32)
        place = fluid.NPUPlace(
            0) if fluid.core.is_compiled_with_npu() else fluid.CPUPlace()
        for reduction in ['sum', 'mean', 'none']:
            static_result = test_static_layer(place,
                                              input_np,
                                              label_np,
                                              reduction,
                                              weight_np=weight_np)
            dy_result = test_dygraph_layer(place,
                                           input_np,
                                           label_np,
                                           reduction,
                                           weight_np=weight_np)
            expected = calc_bceloss(input_np,
                                    label_np,
                                    reduction,
                                    weight_np=weight_np)
            np.testing.assert_allclose(static_result, expected, rtol=1e-6)
            np.testing.assert_allclose(static_result, dy_result, rtol=1e-6)
            np.testing.assert_allclose(dy_result, expected, rtol=1e-6)
            static_functional = test_static_functional(place,
                                                       input_np,
                                                       label_np,
                                                       reduction,
                                                       weight_np=weight_np)
            dy_functional = test_dygraph_functional(place,
                                                    input_np,
                                                    label_np,
                                                    reduction,
                                                    weight_np=weight_np)
            np.testing.assert_allclose(static_functional, expected, rtol=1e-6)
            np.testing.assert_allclose(static_functional,
                                       dy_functional,
                                       rtol=1e-6)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            np.testing.assert_allclose(dy_functional, expected, rtol=1e-6)

    def test_BCELoss_error(self):
        paddle.disable_static(paddle.NPUPlace(0))
<<<<<<< HEAD
        self.assertRaises(
            ValueError, paddle.nn.loss.BCELoss, reduction="unsupport reduction"
        )
        input = paddle.to_tensor([[0.1, 0.3]], dtype='float32')
        label = paddle.to_tensor([[0.0, 1.0]], dtype='float32')
        self.assertRaises(
            ValueError,
            paddle.nn.functional.binary_cross_entropy,
            input=input,
            label=label,
            reduction="unsupport reduction",
        )
=======
        self.assertRaises(ValueError,
                          paddle.nn.loss.BCELoss,
                          reduction="unsupport reduction")
        input = paddle.to_tensor([[0.1, 0.3]], dtype='float32')
        label = paddle.to_tensor([[0.0, 1.0]], dtype='float32')
        self.assertRaises(ValueError,
                          paddle.nn.functional.binary_cross_entropy,
                          input=input,
                          label=label,
                          reduction="unsupport reduction")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        paddle.enable_static()


def bce_loss(input, label):
<<<<<<< HEAD
    return -1 * (label * np.log(input) + (1.0 - label) * np.log(1.0 - input))


class TestBceLossOp(OpTest):
=======
    return -1 * (label * np.log(input) + (1. - label) * np.log(1. - input))


class TestBceLossOp(OpTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.set_npu()
        self.init_test_case()
        self.op_type = "bce_loss"
        input_np = np.random.uniform(0.1, 0.8, self.shape).astype("float32")
        label_np = np.random.randint(0, 2, self.shape).astype("float32")
        output_np = bce_loss(input_np, label_np)

        self.inputs = {'X': input_np, 'Label': label_np}
        self.outputs = {'Out': output_np}

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ['X'], 'Out')

    def init_test_case(self):
        self.shape = [10, 10]


class TestBceLossOpCase1(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_cast(self):
        self.shape = [2, 3, 4, 5]


class TestBceLossOpCase2(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_cast(self):
        self.shape = [2, 3, 20]


if __name__ == "__main__":
    unittest.main()
