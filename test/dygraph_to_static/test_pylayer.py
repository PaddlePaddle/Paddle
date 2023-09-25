# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

"""Tests for PyLayer of Dynamic-to-Static.
Only test simple cases here."""

import unittest

import numpy as np

import paddle
from paddle.autograd.py_layer import PyLayer

SEED = 2023
np.random.seed(SEED)


def compare_result(dygraph_res, static_res, rtol=1e-5, atol=0):
    np.testing.assert_allclose(
        dygraph_res.detach().numpy(),
        static_res.detach().numpy(),
        rtol=rtol,
        atol=atol,
        err_msg='dygraph result is {}\nstatic_result is {}'.format(
            dygraph_res, static_res
        ),
    )


class scaled_layer_1(PyLayer):
    @staticmethod
    def forward(ctx, x):
        y = x * 3
        return y

    @staticmethod
    def backward(ctx, dy):
        dx = paddle.sin(dy)
        return dx


class scaled_layer_2(PyLayer):
    @staticmethod
    def forward(ctx, x1, x2):
        y = x1 * x2
        return y

    @staticmethod
    def backward(ctx, dy):
        dx1 = paddle.sin(dy)
        dx2 = paddle.cos(dy)
        return dx1, dx2


class cus_tanh_1(PyLayer):
    @staticmethod
    def forward(ctx, x):
        y = paddle.tanh(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        (y,) = ctx.saved_tensor()
        grad = dy * (1 - paddle.square(y))
        return grad


class nested_layer(PyLayer):
    @staticmethod
    def forward(ctx, x1, x2):
        y = cus_tanh_1.apply(x1)
        ctx.save_for_backward(y)
        ret = y + x2
        return ret

    @staticmethod
    def backward(ctx, dy):
        (y,) = ctx.saved_tensor()
        grad1 = scaled_layer_1.apply(dy)
        grad2 = dy - paddle.square(y)
        return grad1, grad2


class SimpleNet_1(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(4, 8)

    @paddle.jit.to_static
    def forward(self, data):
        hidden = self.linear(data)
        z = cus_tanh_1.apply(hidden)
        return z


class SimpleNetInplace(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    @paddle.jit.to_static
    def forward(self, data):
        data = data**2
        z = paddle.tanh(data)
        z = cus_tanh_1.apply(z)
        return z


class TestPyLayerBase(unittest.TestCase):
    def setUp(self):
        self.place = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        self.to_static = False

    def _run(self, *input_args, **input_kwargs):
        assert getattr(
            self, "dygraph_func", None
        ), "Please setting `self.dygraph_func` before calling `self._run`"

        paddle.jit.enable_to_static(self.to_static)
        paddle.set_device(self.place)
        result = self.dygraph_func(*input_args, **input_kwargs)
        result.mean().backward()
        return result

    def _run_dygraph(self, *args, **kwargs):
        self.to_static = False
        return self._run(*args, **kwargs)

    def _run_static(self, *args, **kwargs):
        self.to_static = True
        return self._run(*args, **kwargs)

    # TODO(MarioLulab): In the future, this will be supported: not only `paddle.Tensor`
    # but also non-Tensor objects will be included in the argument list.
    def _run_and_compare(self, *args, **kwargs):
        # Step1. Clone args and kwargs to avoid dygraph and static overwriting with each other
        dygraph_inp_args = []
        static_inp_args = []
        for v in args:
            assert isinstance(
                v, paddle.Tensor
            ), "Only Support `paddle.Tensor` now"
            stop_gradient = v.stop_gradient
            # detach from the compute graph to turn `dygraph_inp_args` and `static_inp_args` into leaf nodes
            v = v.detach()
            dygraph_inp_args.append(v.clone())
            static_inp_args.append(v.clone())
            if not stop_gradient:
                dygraph_inp_args[-1].stop_gradient = False
                static_inp_args[-1].stop_gradient = False

        dygraph_inp_kwargs = {}
        static_inp_kwargs = {}
        for k, v in kwargs.items():
            stop_gradient = v.stop_gradient
            assert isinstance(
                v, paddle.Tensor
            ), "Only Support `paddle.Tensor` now"
            # detach from the compute graph to turn `dygraph_inp_kwargs` and `static_inp_kwargs` into leaf nodes
            v = v.detach()
            dygraph_inp_kwargs[k] = v.clone()
            static_inp_kwargs[k] = v.clone()
            if not stop_gradient:
                dygraph_inp_kwargs[k].stop_gradient = False
                static_inp_kwargs[k].stop_gradient = False

        # Step2. Run the dygraph and the static seperately
        dygraph_res = self._run_dygraph(*dygraph_inp_args, **dygraph_inp_kwargs)
        static_res = self._run_static(*static_inp_args, **static_inp_kwargs)

        # Step3. Compare forward result between dygraph and static
        if not isinstance(dygraph_res, tuple):
            dygraph_res = (dygraph_res,)
        if not isinstance(static_res, tuple):
            static_res = (static_res,)

        for d, s in zip(dygraph_res, static_res):
            compare_result(d, s)

        # Step4. Compare grad between dygraph and static
        for i in range(len(dygraph_inp_args)):
            self.assertEqual(
                dygraph_inp_args[i].stop_gradient,
                static_inp_args[i].stop_gradient,
            )
            if dygraph_inp_args[i].stop_gradient:
                continue

            compare_result(dygraph_inp_args[i].grad, static_inp_args[i].grad)

        for key in dygraph_inp_kwargs.keys():
            self.assertEqual(
                dygraph_inp_kwargs[key].stop_gradient,
                static_inp_kwargs[key].stop_gradient,
            )
            if dygraph_inp_kwargs[key].stop_gradient:
                continue

            compare_result(
                dygraph_inp_kwargs[key].grad, static_inp_kwargs[key].grad
            )


class TestPyLayerWithoutContext(TestPyLayerBase):
    def test_single_in_single_out(self):
        @paddle.jit.to_static
        def test_func(x):
            y = scaled_layer_1.apply(x)
            return y

        self.dygraph_func = test_func

        input1 = paddle.randn([2, 3]).astype("float32")
        input1.stop_gradient = False

        self._run_and_compare(input1)

    def test_multi_in_single_out(self):
        @paddle.jit.to_static
        def test_func(x1, x2):
            y = scaled_layer_2.apply(x1, x2)
            return y

        self.dygraph_func = test_func

        input1 = paddle.randn([2, 3]).astype("float32")
        input2 = paddle.randn([2, 3]).astype("float32")
        input1.stop_gradient = False
        input2.stop_gradient = False

        self._run_and_compare(input1, input2)


class TestPyLayerWithContext(TestPyLayerBase):
    def test_single_in_single_out(self):
        @paddle.jit.to_static
        def test_func(x):
            y = cus_tanh_1.apply(x)
            return y

        self.dygraph_func = test_func

        input1 = paddle.randn([2, 3]).astype("float32")
        input1.stop_gradient = False

        self._run_and_compare(input1)

    def test_nested_pylayer(self):
        @paddle.jit.to_static
        def test_func(x1, x2):
            y = nested_layer.apply(x1, x2)
            return y

        self.dygraph_func = test_func

        input1 = paddle.randn([2, 3]).astype("float32")
        input2 = paddle.randn([2, 3]).astype("float32")
        input1.stop_gradient = False
        input2.stop_gradient = False

        self._run_and_compare(input1, input2)


class TestPyLayerInsideNet(TestPyLayerBase):
    def test_single_in_single_out(self):
        simple_net = SimpleNet_1()
        self.dygraph_func = simple_net

        input1 = paddle.randn([3, 4]).astype("float32")
        input1.stop_gradient = False
        self._run_and_compare(input1)

    def test_inplace(self):
        simple_net = SimpleNetInplace()
        self.dygraph_func = simple_net

        input1 = paddle.randn([3, 4]).astype("float32")
        input1.stop_gradient = False
        self._run_and_compare(input1)


if __name__ == "__main__":
    unittest.main()
