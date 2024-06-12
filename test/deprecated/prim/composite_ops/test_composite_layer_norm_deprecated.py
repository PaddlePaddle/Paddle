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
from prim.composite_ops.utils import SUB_TOLERANCE

import paddle
from paddle import _C_ops
from paddle.base import core, framework
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_mode
from paddle.incubate.autograd import primapi
from paddle.nn import LayerNorm


def generate_data(shape1, shape2, shape3, dtype="float32"):
    np.random.seed(200)
    np_data1 = np.random.random(shape1).astype(dtype)
    np_data2 = np.random.random(shape2).astype(dtype)
    np_data3 = np.random.random(shape3).astype(dtype)
    return np_data1, np_data2, np_data3


def layer_norm_wrapper(
    x, normalized_shape, weight=None, bias=None, epsilon=1e-05, name=None
):
    input_shape = list(x.shape)
    input_ndim = len(input_shape)

    normalized_ndim = len(normalized_shape)
    begin_norm_axis = input_ndim - normalized_ndim
    if (
        input_ndim < normalized_ndim
        or input_shape[begin_norm_axis:] != normalized_shape
    ):
        str_normalized_shape = str(normalized_shape)
        raise ValueError(
            'Given normalized_shape is '
            + str_normalized_shape
            + ', expected input with shape [*, '
            + str_normalized_shape[1:]
            + ', but got input shape '
            + str(input_shape)
        )

    if in_dynamic_mode():
        return _C_ops.layer_norm(x, weight, bias, epsilon, begin_norm_axis)

    else:
        inputs = {}
        inputs['X'] = [x]
        if weight:
            inputs['Scale'] = [weight]
        if bias:
            inputs['Bias'] = [bias]
        attrs = {"epsilon": epsilon, "begin_norm_axis": begin_norm_axis}

        # create output
        helper = LayerHelper('layer_norm', **locals())
        from paddle.base.data_feeder import convert_dtype

        param_dtype = (
            x.dtype if convert_dtype(x.dtype) != 'float16' else 'float32'
        )
        mean_out = helper.create_variable_for_type_inference(
            dtype=param_dtype, stop_gradient=True
        )
        variance_out = helper.create_variable_for_type_inference(
            dtype=param_dtype, stop_gradient=True
        )
        layer_norm_out = helper.create_variable_for_type_inference(x.dtype)

        helper.append_op(
            type="layer_norm",
            inputs=inputs,
            outputs={
                "Y": layer_norm_out,
                "Mean": mean_out,
                "Variance": variance_out,
            },
            attrs={"epsilon": epsilon, "begin_norm_axis": begin_norm_axis},
        )

        return layer_norm_out, mean_out, variance_out


class Attr:
    def __init__(self) -> None:
        self.dtype = None
        self.n_shape = None
        self.shape1 = None
        self.shape2 = None
        self.shape3 = None

    def set_dtype(self, dtype) -> None:
        self.dtype = dtype

    def set_shape(self, n_shape, shape1=[], shape2=[], shape3=[]) -> None:
        self.n_shape = n_shape
        self.shape1 = shape1
        self.shape2 = shape2
        self.shape3 = shape3

    def get_rtol(self, flag):
        rtol = SUB_TOLERANCE[self.dtype][flag].get("rtol")
        return rtol

    def get_atol(self, flag):
        atol = SUB_TOLERANCE[self.dtype][flag].get("atol")
        return atol


attrs = Attr()


def fn(x, norm_shape, w, b):
    return layer_norm_wrapper(x, norm_shape, w, b)


def expect_forward(x, norm_shape, w, b):
    return fn(x, norm_shape, w, b)


class TestCompositelayer_norm(unittest.TestCase):
    def setUp(self):
        self.dtypes = ["float32", "float64"]
        self.n_shape = [[4], [64, 128], [64]]
        self.shape1s = [[3, 4], [64, 64, 128], [128, 64, 64]]
        self.shape2s = [[4], [64 * 128], [64]]
        self.shape3s = [[4], [64 * 128], [64]]

    def cal_composite(self, inputs, norm_shape, weight, bias):
        paddle.enable_static()
        core._set_prim_forward_enabled(True)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(
                'x', shape=inputs.shape, dtype=str(inputs.dtype)
            )
            w = paddle.static.data(
                'w', shape=weight.shape, dtype=str(weight.dtype)
            )
            b = paddle.static.data('b', shape=bias.shape, dtype=str(bias.dtype))
            out, mean, var = fn(x, norm_shape, w, b)

            blocks = main_program.blocks

            fwd_ops = [op.type for op in blocks[0].ops]
            # Ensure that layer_norm in original block
            self.assertTrue('layer_norm' in fwd_ops)

            primapi.to_prim(blocks)

            fwd_ops_new = [op.type for op in blocks[0].ops]
            # Ensure that layer_norm is splitted into small ops
            self.assertTrue('layer_norm' not in fwd_ops_new)

        exe = paddle.static.Executor()
        exe.run(startup_program)
        res = exe.run(
            main_program,
            feed={
                'x': inputs,
                'w': weight,
                'b': bias,
            },
            fetch_list=[out, mean, var],
        )
        paddle.disable_static()
        core._set_prim_forward_enabled(False)
        return res

    def cal2_composite(self, inputs, norm_shape, weight, bias):
        paddle.enable_static()
        core._set_prim_forward_enabled(True)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(
                'x', shape=inputs.shape, dtype=str(inputs.dtype)
            )

            out, mean, var = fn(x, norm_shape, weight, bias)

            blocks = main_program.blocks

            fwd_ops = [op.type for op in blocks[0].ops]
            # Ensure that layer_norm in original block
            self.assertTrue('layer_norm' in fwd_ops)

            primapi.to_prim(blocks)

            fwd_ops_new = [op.type for op in blocks[0].ops]
            # Ensure that layer_norm is splitted into small ops
            self.assertTrue('layer_norm' not in fwd_ops_new)

        exe = paddle.static.Executor()
        exe.run(startup_program)
        res = exe.run(
            main_program,
            feed={
                'x': inputs,
            },
            fetch_list=[out, mean, var],
        )
        paddle.disable_static()
        core._set_prim_forward_enabled(False)
        return res

    def compare_forward(self):
        x, w, b = generate_data(
            attrs.shape1, attrs.shape2, attrs.shape3, attrs.dtype
        )
        n_shape = attrs.n_shape
        x_p = paddle.to_tensor(x)
        w_p = paddle.to_tensor(w)
        b_p = paddle.to_tensor(b)

        expect = expect_forward(x_p, n_shape, w_p, b_p)
        actual, _a_mean, _a_var = self.cal_composite(x, n_shape, w, b)

        assert expect.numpy().dtype == actual.dtype
        np.testing.assert_allclose(
            expect.numpy(),
            actual,
            rtol=attrs.get_rtol("forward"),
            atol=attrs.get_atol("forward"),
        )

        expect_2 = expect_forward(x_p, n_shape, None, None)
        actual_2, _a_mean_2, _a_var_2 = self.cal2_composite(
            x, n_shape, None, None
        )
        assert expect_2.numpy().dtype == actual_2.dtype
        np.testing.assert_allclose(
            expect_2.numpy(),
            actual_2,
            rtol=attrs.get_rtol("forward"),
            atol=attrs.get_atol("forward"),
        )

    def test_forward(self):
        for j in self.dtypes:
            if paddle.device.get_device() == "cpu" and j == "float16":
                print("need pass this case")
                continue
            for t in range(0, len(self.shape1s)):
                attrs.set_dtype(j)
                attrs.set_shape(
                    self.n_shape[t],
                    self.shape1s[t],
                    self.shape2s[t],
                    self.shape3s[t],
                )
                self.compare_forward()


def apply_to_static(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(net, build_strategy=False, full_graph=True)


class PrimeNet(paddle.nn.Layer):
    def __init__(self, n_shape):
        super().__init__()
        self.ln = LayerNorm(n_shape)

    def forward(self, x):
        out = self.ln(x)
        return out


class TestPrimForwardAndBackward(unittest.TestCase):
    """
    Test PrimeNet with @to_static + prim forward + prim backward + cinn v.s Dygraph
    """

    def setUp(self):
        paddle.seed(2022)
        self.n_shape = [[4], [64, 128], [64]]
        self.shape1s = [[3, 4], [64, 64, 128], [128, 64, 64]]

    def train(self, use_prim):
        self.x = paddle.randn(attrs.shape1, dtype="float32")
        self.x.stop_gradient = False
        core._set_prim_all_enabled(use_prim)
        paddle.seed(2022)
        net = PrimeNet(attrs.n_shape)
        sgd = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=net.parameters()
        )

        net = paddle.amp.decorate(models=net, level='O2')

        net = apply_to_static(net, False)
        with paddle.amp.auto_cast(level='O2'):
            out = net(self.x)
            loss = paddle.mean(out)
            loss.backward()
            sgd.step()
            sgd.clear_grad()
            return loss

    def compare_forward(self):
        if not isinstance(framework._current_expected_place(), core.CPUPlace):
            expected = self.train(False)
            actual = self.train(True)
            np.testing.assert_allclose(
                expected,
                actual,
                rtol=1e-3,
                atol=1e-3,
            )

    def test_forward(self):
        for t in range(0, len(self.shape1s)):
            attrs.set_shape(
                self.n_shape[t],
                self.shape1s[t],
            )
            self.compare_forward()


if __name__ == '__main__':
    unittest.main()
