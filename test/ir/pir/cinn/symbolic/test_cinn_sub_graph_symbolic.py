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
import sys
from os.path import dirname

sys.path.append(dirname(dirname(__file__)))

import unittest

import numpy as np
import utils

import paddle
from paddle.static import InputSpec


def get_sym_shape_str_for_op(net, input_spec, op_name):
    forward_program = net.forward.get_concrete_program(*input_spec)[
        1
    ].infer_program.forward_program
    all_sym_shape_str = []
    for op in forward_program.global_block().ops:
        if op.name() == op_name:
            all_sym_shape_str.append(op.attrs()['sym_shape_str'])

    return all_sym_shape_str


def exp_sub(x):
    y = paddle.exp(x)
    z = y - x
    return z


def reshape(x):
    y = paddle.exp(x)
    z = y - x
    i = paddle.shape(x)[0]
    j = paddle.shape(y)[1]
    out = paddle.reshape(z, shape=[i, j])
    return out


def broadcast(x, y):
    z = x + y
    return z


class CINNSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fn = exp_sub

    def forward(self, x):
        out = self.fn(x)
        return out


class CINNReshapeSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fn = reshape

    def forward(self, x):
        out = self.fn(x)
        return out


class CINNBroadcastSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fn = broadcast

    def forward(self, x, y):
        out = self.fn(x, y)
        return out


class TestCinnSubGraphBase(unittest.TestCase):
    """
    Test Pir API + @to_static + CINN.
    """

    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.shape = [64, 128]
        self.x = paddle.randn(self.shape, dtype="float32")
        self.x.stop_gradient = False

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)
        utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: 1})

    def test_eval_symbolic(self):
        pass


class TestCinnExpSubGraph(TestCinnSubGraphBase):
    def eval_symbolic(self, use_cinn):
        net = CINNSubGraphNet()
        input_spec = [InputSpec(shape=[None, 128], dtype='float32')]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval_symbolic(self):
        cinn_out = self.eval_symbolic(use_cinn=True)
        dy_out = self.eval_symbolic(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


class TestCinnDyShapeBase(TestCinnSubGraphBase):
    def prepare_data(self):
        self.shape = [4, 256]
        self.x = paddle.randn(self.shape, dtype="float32")
        self.x.stop_gradient = False

    def eval_symbolic(self, use_cinn):
        paddle.seed(2022)
        net = CINNReshapeSubGraphNet()
        input_spec = [InputSpec(shape=[None, 256], dtype='float32')]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval_symbolic(self):
        cinn_out = self.eval_symbolic(use_cinn=True)
        dy_out = self.eval_symbolic(use_cinn=False)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


class TestCinnDyShapeBC(TestCinnSubGraphBase):
    def prepare_data(self):
        self.x_shape = [2, 4, 1]
        self.x = paddle.randn(self.x_shape, dtype="float32")
        self.x.stop_gradient = False

        self.y_shape = [4, 5]
        self.y = paddle.randn(self.y_shape, dtype="float32")
        self.y.stop_gradient = False

    def eval_symbolic(self, use_cinn):
        paddle.seed(2022)
        net = CINNBroadcastSubGraphNet()
        input_spec = [
            InputSpec(shape=[None, None, None], dtype='float32'),
            InputSpec(shape=[None, None], dtype='float32'),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()
        out = net(self.x, self.y)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval_symbolic(self):
        # cinn_out = self.eval_symbolic(use_cinn=True)
        dy_out = self.eval_symbolic(use_cinn=False)
        # np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


class LlamaRMSNorm(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.hidden_size = 4096
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = 1e-6

    def forward(self, hidden_states):
        # 1. variance = hidden_states.pow(2).mean(-1, keepdim=True)

        axis_rst = -1
        # 1.1 decomp pow -> elementwise_pow
        pow_tensor = paddle.full([1], 2, hidden_states.dtype)
        pow_rst = paddle.pow(hidden_states, pow_tensor)

        # 1.2 decomp mean -> sum & div
        sum_rst = paddle.sum(pow_rst, [axis_rst], keepdim=True)
        shape_rst = paddle.shape(sum_rst)
        div_by = paddle.full(shape_rst, hidden_states.shape[axis_rst])
        variance = paddle.divide(sum_rst, div_by)

        # 2. hidden_states = (paddle.rsqrt(variance + self.variance_epsilon) * hidden_states)

        # 2.1 decomp variance + self.variance_epsilon -> full + scale
        scale_tensor = paddle.full([1], 1.0)
        scale_rst = paddle.scale(variance, scale_tensor, self.variance_epsilon)

        # 2.2 decomp rsqrt -> pow(-0.5)
        rsqrt_tensor = paddle.full([1], -0.5)
        rsqrt_rst = paddle.pow(scale_rst, rsqrt_tensor)

        hidden_states = rsqrt_rst * hidden_states

        return hidden_states * self.weight


class TestCinnDyShapeRMSNorm(TestCinnSubGraphBase):
    def prepare_data(self):
        self.hidden_states_shape = [1, 300, 4096]
        self.hidden_states = paddle.randn(
            self.hidden_states_shape, dtype="float32"
        )
        self.hidden_states.stop_gradient = False
        self.expected_output_sym_shape = 'shape[S0, S1, 4096]'

    def eval_symbolic(self, use_cinn):
        paddle.seed(2022)
        net = LlamaRMSNorm()
        input_spec = [
            InputSpec(shape=[None, None, 4096], dtype='float32'),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()

        sym_shape_str_list = get_sym_shape_str_for_op(
            net, input_spec, 'builtin.shadow_output'
        )
        np.testing.assert_equal(len(sym_shape_str_list), 1)
        np.testing.assert_equal(
            sym_shape_str_list[0].find(self.expected_output_sym_shape),
            0,
            'output shape is not expected!',
        )

        out = net(self.hidden_states)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)

        return out

    def test_eval_symbolic(self):
        # cinn_out = self.eval_symbolic(use_cinn=True)
        dy_out = self.eval_symbolic(use_cinn=False)
        # np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


def unsqueeze_composite(x, axis):
    """define composite rule of op unsqueeze"""
    """using reshape to implement unsqueeze op"""
    x_shape = list(x.shape)
    axis_list = list(axis)
    for i in axis_list:
        if i < 0:
            i += len(x_shape) + 1
        x_shape = (
            x_shape[:i]
            + [
                1,
            ]
            + x_shape[i:]
        )
    out = paddle.reshape(x, x_shape)
    return out


class LlamaRepeatKV(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.n_rep = 4

    def forward(self, hidden_states):
        batch, slen, num_key_value_heads, head_dim = hidden_states.shape
        rst_unsqueeze = unsqueeze_composite(hidden_states, [-2])
        rst_tile = rst_unsqueeze.tile([1, 1, 1, self.n_rep, 1])
        out = rst_tile.reshape(
            [batch, slen, num_key_value_heads * self.n_rep, head_dim]
        )

        return out


class TestCinnDyShapeRepeatKV(TestCinnSubGraphBase):
    def prepare_data(self):
        self.hidden_states_shape = [1, 2048, 8, 96]
        self.hidden_states = paddle.randn(
            self.hidden_states_shape, dtype="float32"
        )
        self.hidden_states.stop_gradient = False
        self.expected_output_sym_shape = 'shape[S0, S1, 32, 96]'

    def eval_symbolic(self, use_cinn):
        paddle.seed(2022)
        net = LlamaRepeatKV()
        input_spec = [
            InputSpec(shape=[None, None, 8, 96], dtype='float32'),
        ]
        net = utils.apply_to_static(net, use_cinn, input_spec)
        net.eval()

        sym_shape_str_list = get_sym_shape_str_for_op(
            net, input_spec, 'builtin.shadow_output'
        )
        np.testing.assert_equal(len(sym_shape_str_list), 1)
        np.testing.assert_equal(
            sym_shape_str_list[0].find(self.expected_output_sym_shape),
            0,
            'output shape is not expected!',
        )

        out = net(self.hidden_states)
        if use_cinn:
            self.check_jit_kernel_info(net.forward)
        return out

    def test_eval_symbolic(self):
        # cinn_out = self.eval_symbolic(use_cinn=True)
        dy_out = self.eval_symbolic(use_cinn=False)
        # np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
