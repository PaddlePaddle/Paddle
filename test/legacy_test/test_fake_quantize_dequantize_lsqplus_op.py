# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import itertools
import math
import unittest

import numpy as np
from op_test import OpTest

import paddle
from paddle import _C_ops
from paddle.nn.quant.lsqplus import fake_quantize_dequantize_lsqplus


# rounding to nearest ties away from zero
def round_c_single_element(val):
    dtype = type(val)
    if val >= 0:
        return dtype(np.floor(val + 0.5))
    return dtype(np.ceil(val - 0.5))


round_c = np.vectorize(round_c_single_element)


# rounding to nearest ties to even
def round_with_ties_to_even(x):
    x_low = math.floor(x)
    x_up = math.ceil(x)
    dlow = x - x_low
    dup = x_up - x

    if dlow == dup:
        if math.fmod(x_low, 2.0) == 0:
            x = x_low
        else:
            x = x_up
    else:
        if dlow < dup:
            x = x_low
        else:
            x = x_up
    return x


round_t = np.vectorize(round_with_ties_to_even)


def inverse(scale):
    one = np.array(1.0).astype(scale.dtype)
    eps6 = np.array(1e-6).astype(scale.dtype)
    eps30 = np.array(1e-30).astype(scale.dtype)
    inv_scale = one / (scale + eps6) if scale < eps30 else one / scale
    return inv_scale


def get_compute_type(dtype):
    assert dtype in [np.float16, np.float32, np.float64]
    if dtype == np.float16:
        return np.float32
    return dtype


def fake_quantize_dequantize_lsqplus_wrapper(
    x, alpha, beta, g_scale, bit_length, is_sign, round_type
):
    return _C_ops.fake_quantize_dequantize_lsqplus(
        x, alpha, beta, g_scale, bit_length, is_sign, round_type
    )


class TestFakeQuantizeDequantizeLsqplusOp(OpTest):
    def setUp(self):
        self.op_type = 'fake_quantize_dequantize_lsqplus'
        self.attrs = {
            'bit_length': 8,
        }
        self.python_api = fake_quantize_dequantize_lsqplus_wrapper

    def _cal_backward(
        self, out_grad, x, alpha, beta, g, Qn, Qp, round_type, dtype
    ):
        compute_type = get_compute_type(dtype)
        x = x.astype(compute_type)
        alpha = alpha.astype(compute_type)
        beta = beta.astype(compute_type)
        g = g.astype(compute_type)
        out_grad = out_grad.astype(compute_type)

        round_fn = round_t if round_type == 'TiesToEven' else round_c
        q_x = (x - beta) / alpha
        lower_flag = (q_x < Qn).astype(dtype)
        upper_flag = (q_x > Qp).astype(dtype)
        middle_flag = 1.0 - lower_flag - upper_flag
        grad_alpha = np.sum(
            (
                lower_flag * Qn
                + upper_flag * Qp
                + middle_flag * round_fn(q_x)
                - middle_flag * q_x
            )
            * out_grad
            * g
        )
        grad_beta = np.sum((lower_flag + upper_flag) * out_grad * g)
        grad_x = middle_flag * out_grad

        grad_alpha = grad_alpha.astype(dtype)
        grad_beta = grad_beta.astype(dtype)
        grad_x = grad_x.astype(dtype)

        return grad_alpha.reshape([1]), grad_beta.reshape([1]), grad_x

    def _fake_quantize_dequantize_lsqplus(
        self,
        dtype,
        input_shape,
        distributions,
        round_type='TiesAwayFromZero',
        is_sign=False,
    ):
        input_data = distributions[0](input_shape).astype(dtype)
        alpha = distributions[1]([1]).astype(dtype)
        beta = distributions[2]([1]).astype(dtype)
        g_scale = distributions[3]([1]).astype(dtype)

        # prepare for forward stage
        if is_sign:
            Qn, Qp = (
                -(2 ** (self.attrs['bit_length'] - 1)),
                2 ** (self.attrs['bit_length'] - 1) - 1,
            )
        else:
            Qn, Qp = 0, 2 ** (self.attrs['bit_length']) - 1

        if round_type == 'TiesToEven':
            # round then clip
            round_out = round_t((input_data - beta) / alpha)
            output_data = np.clip(round_out, Qn, Qp) * alpha + beta
            self.attrs['round_type'] = 0
        else:
            # clip then round
            data = np.clip((input_data - beta) / alpha, Qn, Qp)
            output_data = round_c(data) * alpha + beta
            self.attrs['round_type'] = 1

        self.attrs['is_sign'] = is_sign
        self.attrs['round_type'] = 0 if round_type == 'TiesToEven' else 1
        self.inputs = {
            'x': input_data,
            'alpha': alpha,
            'beta': beta,
            'g_scale': g_scale,
        }
        self.outputs = {'out': output_data}
        self.dtype = dtype

        # check forward stage
        self.check_output()
        # check backward stage
        out_grad = np.random.random(input_data.shape).astype(dtype)
        # get input gradient
        test_gradients = self._cal_backward(
            out_grad,
            input_data,
            alpha,
            beta,
            g_scale,
            Qn,
            Qp,
            round_type,
            dtype,
        )
        # use 'gradient' to verify gradient from actual operator(its gradient generated by out_grad)
        self._test_grad(
            ['alpha', 'beta', 'x'],
            test_gradients,
            [out_grad],
            {'g_scale'},
        )

    def _test_grad(self, name, gradient, out_grads, no_grad_set=set()):
        self.check_grad(
            name,
            'out',
            user_defined_grads=gradient,
            user_defined_grad_outputs=out_grads,
            no_grad_set=no_grad_set,
        )

    def test_fake_quantize_dequantize(self):
        distributions = [
            np.random.random,
            np.random.random,
            np.random.random,
            np.random.random,
        ]
        dtype_options = [np.float32, np.float16, np.float64]
        input_shape_options = [
            (20, 15, 6, 6),
            (20, 15, 6, 6),
            (30, 30),
            (30, 30),
        ]
        round_type_options = ['TiesToEven', 'TiesAwayFromZero']
        is_sign_options = [True, False]
        for dtype, input_shape, round_type, is_sign in itertools.product(
            dtype_options,
            input_shape_options,
            round_type_options,
            is_sign_options,
        ):
            with self.subTest(
                dtype=dtype,
                input_shape=input_shape,
                round_type=round_type,
                is_sign=is_sign,
            ):
                self._fake_quantize_dequantize_lsqplus(
                    dtype, input_shape, distributions, round_type, is_sign
                )


def ref_lsqplus(
    x, alpha, beta, g_scale, bit_length, is_sign, round_type, dtype
):
    if is_sign:
        Qp = 2 ** (bit_length - 1) - 1
        Qn = -(2 ** (bit_length - 1))
    else:
        Qn = 0
        Qp = 2 ** (bit_length) - 1

    compute_dtype = get_compute_type(dtype)
    x = x.astype(compute_dtype)
    alpha = alpha.astype(compute_dtype)
    beta = beta.astype(compute_dtype)

    if round_type == 1:
        # clip then round
        # out = np.round(np.clip((x - beta)*inverse(alpha), Qn, Qp)) * alpha + beta
        out = (
            round_c(np.clip((x - beta) * inverse(alpha), Qn, Qp)) * alpha + beta
        )
        # out = np.clip(round_c((x - beta)*inverse(alpha)), Qn, Qp) * alpha + beta
    else:
        # round then clip
        out = (
            np.clip(round_t((x - beta) * inverse(alpha)), Qn, Qp) * alpha + beta
        )

    out = out.astype(dtype)
    return out


class TestLsqplus(unittest.TestCase):
    def setUp(self):
        self.bit_width = 8

    def run_dyamic(
        self,
        input_data,
        alpha,
        beta,
        g_scale,
        bit_width,
        is_sign,
        round_type,
        place='cpu',
    ):
        input_data = paddle.to_tensor(input_data)
        input_data = input_data.to(place)
        alpha = paddle.to_tensor(alpha)
        alpha = alpha.to(place)
        beta = paddle.to_tensor(beta)
        g_scale = paddle.to_tensor(g_scale)
        g_scale = g_scale.to(place)
        out = fake_quantize_dequantize_lsqplus(
            input_data, alpha, beta, g_scale, bit_width, is_sign, round_type
        )
        return out.numpy()

    def run_static(
        self,
        input_data,
        alpha_v,
        beta_v,
        g_scale_v,
        bit_width,
        is_sign,
        round_type,
        dtype,
        place='cpu',
    ):
        paddle.enable_static()
        x = paddle.static.data(name='x', shape=input_data.shape, dtype=dtype)
        x.stop_gradient = False
        alpha = paddle.static.data(name='alpha', shape=[1], dtype=dtype)
        alpha.stop_gradient = False
        beta = paddle.static.data(name='beta', shape=[1], dtype=dtype)
        beta.stop_gradient = False
        g_scale = paddle.static.data(name='g_scale', shape=[1], dtype=dtype)
        g_scale.stop_gradient = True

        if place == 'cpu':
            place = paddle.CPUPlace()
        elif place == 'gpu':
            place = paddle.CUDAPlace(0)
        else:
            raise ValueError("Unsupported place")

        exe = paddle.static.Executor(place)
        out = fake_quantize_dequantize_lsqplus(
            x, alpha, beta, g_scale, bit_width, is_sign, round_type
        )
        exe.run(paddle.static.default_startup_program())
        out_result = exe.run(
            paddle.static.default_main_program(),
            feed={
                'x': input_data,
                'alpha': alpha_v,
                'beta': beta_v,
                'g_scale': g_scale_v,
            },
            fetch_list=[out],
        )
        paddle.disable_static()
        return out_result

    def run_fake_quant(
        self,
        dtype,
        input_shape,
        distributions,
        round_type='TiesAwayFromZero',
        is_sign=False,
        dygraph=True,
        place='cpu',
    ):
        input_data = distributions[0](input_shape).astype(dtype)
        alpha = distributions[1]([1]).astype(dtype)
        beta = distributions[2]([1]).astype(dtype)
        g_scale = distributions[3]([1]).astype(dtype)

        round_type = 0 if round_type == 'TiesToEven' else 1
        ref_out = ref_lsqplus(
            input_data,
            alpha,
            beta,
            g_scale,
            self.bit_width,
            is_sign,
            round_type,
            dtype,
        )
        if dtype == np.float16:
            dtype = 'float16'
        elif dtype == np.float32:
            dtype = 'float32'
        else:
            raise ValueError("Unsupported data type")
        if dygraph:
            out = self.run_dyamic(
                input_data,
                alpha,
                beta,
                g_scale,
                self.bit_width,
                is_sign,
                round_type,
                place=place,
            )
        else:
            out = self.run_static(
                input_data,
                alpha,
                beta,
                g_scale,
                self.bit_width,
                is_sign,
                round_type,
                dtype=dtype,
                place=place,
            )
        # print(np.abs(ref_out-out).max())
        self.assertEqual(np.allclose(out, ref_out), True, "output not equal")

    def test_fake_quantize_dequantize_dygraph(self):
        distributions = [
            np.random.random,
            np.random.random,
            np.random.random,
            np.random.random,
        ]

        self.run_fake_quant(
            np.float32,
            (128, 128),
            distributions,
            round_type='TiesToEven',
            is_sign=True,
            dygraph=True,
        )

    def test_fake_quantize_dequantize_static(self):
        distributions = [
            np.random.random,
            np.random.random,
            np.random.random,
            np.random.random,
        ]

        self.run_fake_quant(
            np.float32,
            (128, 128),
            distributions,
            round_type='TiesToEven',
            is_sign=True,
            dygraph=False,
        )

    def test_fake_quantize_dequantize_round1(self):
        distributions = [
            np.random.random,
            np.random.random,
            np.random.random,
            np.random.random,
        ]

        self.run_fake_quant(
            np.float32,
            (128, 128),
            distributions,
            round_type='TiesAwayFromZero',
            is_sign=True,
            dygraph=True,
        )

    def test_fake_quantize_dequantize_not_sign(self):
        distributions = [
            np.random.random,
            np.random.random,
            np.random.random,
            np.random.random,
        ]

        self.run_fake_quant(
            np.float32,
            (128, 128),
            distributions,
            round_type='TiesToEven',
            is_sign=False,
            dygraph=True,
        )


if __name__ == '__main__':
    unittest.main()
