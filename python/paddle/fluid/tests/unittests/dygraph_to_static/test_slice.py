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

from __future__ import print_function

import unittest
import numpy as np

import paddle

SEED = 2020
np.random.seed(SEED)
prog_trans = paddle.jit.ProgramTranslator()


@paddle.jit.to_static
def test_slice_without_control_flow(x):
    # Python slice will not be transformed.
    x = paddle.to_tensor(x)
    a = [x]
    a[0] = paddle.full(shape=[2], fill_value=2, dtype="float32")
    return a[0]


@paddle.jit.to_static
def test_slice_in_if(x):
    x = paddle.to_tensor(x)
    a = []
    if x.numpy()[0] > 0:
        a.append(x)
    else:
        a.append(paddle.full(shape=[1, 2], fill_value=9, dtype="int32"))

    if x.numpy()[0] > 0:
        a[0] = x

    a[0] = x + 1
    out = a[0]
    return out


@paddle.jit.to_static
def test_slice_in_while_loop(x, iter_num=3):
    x = paddle.to_tensor(x)
    iter_num_var = paddle.full(shape=[1], fill_value=iter_num, dtype="int32")
    a = []
    i = 0

    while i < iter_num_var:
        a.append(x)
        i += 1

    i = 0
    while i < iter_num_var.numpy()[0]:
        a[i] = paddle.full(shape=[2], fill_value=2, dtype="float32")
        i += 1
    out = a[0:iter_num]
    return out[0]


@paddle.jit.to_static
def test_slice_in_for_loop(x, iter_num=3):
    x = paddle.to_tensor(x)
    a = []
    # Use `paddle.full` so that static analysis can analyze the type of iter_num is Tensor
    iter_num = paddle.full(
        shape=[1], fill_value=iter_num, dtype="int32"
    )  # TODO(liym27): Delete it if the type of parameter iter_num can be resolved

    for i in range(iter_num):
        a.append(x)

    for i in range(iter_num):
        a[i] = x
    out = a[2]
    return out


@paddle.jit.to_static
def test_set_value(x):
    x = paddle.to_tensor(x)
    x[0] = paddle.full(shape=[1], fill_value=2, dtype="float32")
    x[1:2, 0:1] = 10
    return x


class LayerWithSetValue(paddle.nn.Layer):
    def __init__(self, input_dim, hidden):
        super(LayerWithSetValue, self).__init__()
        self.linear = paddle.nn.Linear(input_dim, hidden)

    @paddle.jit.to_static
    def forward(self, x):
        x = self.linear(x)
        x[0] = 1
        return x


class TestSliceWithoutControlFlow(unittest.TestCase):
    def setUp(self):
        self.init_input()
        self.place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda(
        ) else paddle.CPUPlace()
        self.init_dygraph_func()
        paddle.disable_static()

    def init_input(self):
        self.input = np.random.random((3)).astype('int32')

    def init_dygraph_func(self):
        self.dygraph_func = test_slice_without_control_flow

    def run_dygraph_mode(self):
        return self._run(to_static=False)

    def _run(self, to_static):
        prog_trans.enable(to_static)
        res = self.dygraph_func(self.input)
        return res.numpy()

    def run_static_mode(self):
        return self._run(to_static=True)

    def test_transformed_static_result(self):
        static_res = self.run_static_mode()
        dygraph_res = self.run_dygraph_mode()
        self.assertTrue(
            np.allclose(dygraph_res, static_res),
            msg='dygraph_res is {}\nstatic_res is {}'.format(dygraph_res,
                                                             static_res))


class TestSliceInIf(TestSliceWithoutControlFlow):
    def init_dygraph_func(self):
        self.dygraph_func = test_slice_in_if


class TestSliceInWhileLoop(TestSliceWithoutControlFlow):
    def init_dygraph_func(self):
        self.dygraph_func = test_slice_in_while_loop


class TestSliceInForLoop(TestSliceWithoutControlFlow):
    def init_dygraph_func(self):
        self.dygraph_func = test_slice_in_for_loop


class TestSetValue(TestSliceWithoutControlFlow):
    def init_input(self):
        self.input = np.full([3, 4, 5], 5).astype('float32')

    def init_dygraph_func(self):
        self.dygraph_func = test_set_value


class TestSetValueWithLayerAndSave(unittest.TestCase):
    def test_set_value_with_save(self):
        prog_trans.enable(True)
        model = LayerWithSetValue(input_dim=10, hidden=1)
        x = paddle.full(shape=[5, 10], fill_value=5.0, dtype="float32")
        paddle.jit.save(
            layer=model,
            path="./layer_use_set_value",
            input_spec=[x],
            output_spec=None)


if __name__ == '__main__':
    unittest.main()
