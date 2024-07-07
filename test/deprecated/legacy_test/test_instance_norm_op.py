#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import parameterized as param
from op_test import OpTest
from utils import static_guard

import paddle
from paddle import base, nn
from paddle.base import core


def _reference_instance_norm_naive(x, scale, bias, epsilon, mean, var):
    x_shape = x.shape
    if len(x_shape) == 2:
        x = np.reshape(x, (x.shape[0], x.shape[1], 1, 1))
    n, c, h, w = x.shape

    mean_tile = np.reshape(mean, (n, c, 1, 1))
    mean_tile = np.tile(mean_tile, (1, 1, h, w))
    var_tile = np.reshape(var, (n, c, 1, 1))
    var_tile = np.tile(var_tile, (1, 1, h, w))

    x_norm = (x - mean_tile) / np.sqrt(var_tile + epsilon)
    scale_tile = np.reshape(scale, (1, c, 1, 1))
    scale_tile = np.tile(scale_tile, (n, 1, h, w))
    bias_tile = np.reshape(bias, (1, c, 1, 1))
    bias_tile = np.tile(bias_tile, (n, 1, h, w))
    y = scale_tile * x_norm + bias_tile
    if len(x_shape) == 2:
        y = np.reshape(y, x_shape)
    return y, mean, var


def _reference_instance_norm_grad(x, d_y, scale, mean, var, epsilon):
    # d_scale = sum(d_y * (x-mean) / sqrt(var+epsilon))
    # d_offset = sum(d_y)
    # d_x = scale / sqrt(var+epsilon) * (d_y - np.mean(d_y, axis=(2,3)) - (x-mean)/sqrt(var+epsilon)* np.mean(y_grad * (x-mean)/sqrt(var+epsilon), axis=(2,3)))
    n, c, h, w = x.shape

    d_bias = np.sum(d_y, axis=(0, 2, 3))

    mean_tile = np.reshape(mean, (n, c, 1, 1))
    mean_tile = np.tile(mean_tile, (1, 1, h, w))
    var_tile = np.reshape(var, (n, c, 1, 1))
    var_tile = np.tile(var_tile, (1, 1, h, w))

    d_scale = np.sum(d_y * (x - mean_tile) * var_tile, axis=(0, 2, 3))
    var_inv = var_tile
    scale_tile = np.reshape(scale, (1, c, 1, 1))
    scale_tile = np.tile(scale_tile, (n, 1, h, w))

    d_x = (
        scale_tile
        * var_inv
        * (
            d_y
            - np.mean(d_y, axis=(2, 3), keepdims=True)
            - (x - mean_tile)
            * var_inv
            * np.mean(
                d_y * (x - mean_tile) * var_inv, axis=(2, 3), keepdims=True
            )
        )
    )
    return d_x, d_scale, d_bias


def _cal_mean_variance(x, epsilon, mean_shape):
    mean = np.reshape(np.mean(x, axis=(2, 3)), mean_shape)
    var = np.reshape(np.var(x, axis=(2, 3)), mean_shape)
    return mean, var


def instance_norm_wrapper(x, weight=None, bias=None, esp=1e-05):
    return paddle.nn.functional.instance_norm(
        x, None, None, weight, bias, True, 0.9, esp
    )


class TestInstanceNormOp(OpTest):
    def setUp(self):
        self.op_type = "instance_norm"
        self.prim_op_type = "comp"
        self.python_api = instance_norm_wrapper
        self.public_python_api = instance_norm_wrapper
        self.python_out_sig = ['Y']
        self.fw_comp_rtol = 1e-6
        self.fw_comp_atol = 1e-6
        self.rev_comp_rtol = 1e-4
        self.rev_comp_atol = 1e-4
        self.cinn_rtol = 1e-4
        self.cinn_atol = 1e-4
        self.init_test_case()
        ref_y_np, ref_mean_np, ref_var_np_tmp = _reference_instance_norm_naive(
            self.x_np,
            self.scale_np,
            self.bias_np,
            self.epsilon,
            self.mean_np,
            self.var_np,
        )

        ref_var_np = 1 / np.sqrt(ref_var_np_tmp + self.epsilon)
        self.inputs = {
            'X': self.x_np,
            'Scale': self.scale_np,
            'Bias': self.bias_np,
        }
        self.attrs = {'epsilon': self.epsilon}
        self.outputs = {
            'Y': ref_y_np,
            'SavedMean': ref_mean_np,
            'SavedVariance': ref_var_np,
        }

    def test_check_output(self):
        self.check_output(check_prim=True, check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X', 'Scale', 'Bias'],
            'Y',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )

    def init_test_case(self):
        x_shape = [2, 100, 4, 5]
        n, c, h, w = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        self.epsilon = 1e-05
        dtype = np.float32
        scale_shape = [c]
        mean_shape = [n * c]
        np.random.seed()
        self.x_np = np.random.random_sample(x_shape).astype(dtype)
        self.scale_np = np.random.random_sample(scale_shape).astype(dtype)
        self.bias_np = np.random.random_sample(scale_shape).astype(dtype)
        self.mean_np, self.var_np = _cal_mean_variance(
            self.x_np, self.epsilon, mean_shape
        )
        self.dtype = dtype


class TestInstanceNormFP64(TestInstanceNormOp):
    def init_test_case(self):
        x_shape = [2, 100, 4, 5]
        n, c, h, w = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        self.epsilon = 1e-5
        dtype = np.float64
        scale_shape = [c]
        mean_shape = [n * c]
        np.random.seed()
        self.x_np = np.random.random_sample(x_shape).astype(dtype)
        self.scale_np = np.ones(scale_shape).astype(dtype)
        self.bias_np = np.zeros(scale_shape).astype(dtype)
        self.mean_np, self.var_np = _cal_mean_variance(
            self.x_np, self.epsilon, mean_shape
        )
        self.cinn_atol = 1e-13
        self.cinn_rtol = 1e-13
        self.fw_comp_rtol = 1e-14
        self.fw_comp_atol = 1e-14
        self.rev_comp_rtol = 1e-13
        self.rev_comp_atol = 1e-13
        self.dtype = dtype


class PrimGroupNorm(paddle.nn.Layer):
    def __init__(self, num_channels, scale, bias):
        super().__init__()
        self.func = nn.InstanceNorm2D(num_channels)
        paddle.assign(scale, self.func.scale)
        paddle.assign(bias, self.func.bias)

    def forward(self, x):
        out = self.func(x)
        return out


def apply_to_static(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(net, build_strategy=False, full_graph=True)


places = [paddle.CPUPlace()]
if paddle.is_compiled_with_cuda():
    places.append(paddle.CUDAPlace(0))


@param.parameterized_class(
    (
        'name',
        'shape',
        'epsilon',
        'data_format',
        'places',
        'dtype',
        'threshold_list',
        'special_threshold',
    ),
    (
        (
            'test0',
            (2, 100, 3, 5),
            1e-5,
            'NCHW',
            places,
            'float32',
            [
                [1e-5, 1e-5, 1e-5],  # cpu thresholds for static
                [1e-5, 1e-5, 1e-5],  # gpu thresholds for static
            ],
            None,
        ),
        (
            'test1',
            (2, 100, 3, 5),
            1e-5,
            'NCHW',
            places,
            'float32',
            [
                [1e-5, 1e-5, 1e-5],  # cpu thresholds for static
                [1e-5, 1e-5, 1e-5],  # gpu thresholds for static
            ],
            None,
        ),
        (
            'testbigdata_fp32',
            (8, 32, 32, 64),
            1e-5,
            'NCHW',
            places,
            'float32',
            [
                [1e-5, 1e-5, 1e-5],  # cpu thresholds for static
                [1e-5, 1e-5, 1e-5],  # gpu thresholds for static
            ],  # gpu thresholds
            [2e-2, 2e-2, 2e-2],  # special grad threshold for scale
        ),
        (
            'test0_fp64',
            (2, 100, 3, 5),
            1e-5,
            'NCHW',
            places,
            'float64',
            [
                [1e-14, 1e-14, 1e-14],  # cpu thresholds for static
                [1e-14, 1e-14, 1e-14],  # gpu thresholds for static
            ],
            [1e-13, 1e-13, 1e-13],
        ),
        (
            'test1_fp64',
            (2, 100, 3, 5),
            1e-5,
            'NCHW',
            places,
            'float64',
            [
                [1e-14, 1e-14, 1e-14],  # cpu thresholds for static
                [1e-14, 1e-14, 1e-14],  # gpu thresholds for static
            ],
            [1e-13, 1e-13, 1e-13],
        ),
        (
            'testbigdata_fp64',
            (8, 32, 32, 64),
            1e-5,
            'NCHW',
            places,
            'float64',
            [
                [1e-14, 1e-14, 1e-14],  # cpu thresholds
                [1e-14, 1e-14, 1e-14],
            ],  # gpu thresholds
            [5e-11, 5e-11, 5e-11],  # for X_grad
        ),
    ),
)
class TestCompositeInstanceNormNorm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core._set_prim_all_enabled(True)

    @classmethod
    def tearDownClass(cls):
        core._set_prim_all_enabled(False)

    def setUp(self):
        np.random.seed(1234)
        self.fwd_desire = []
        self.rev_desire = []
        self.x = np.random.random(self.shape).astype(self.dtype)
        self.scale = np.random.random([self.shape[1]]).astype(self.dtype)
        self.bias = np.random.random([self.shape[1]]).astype(self.dtype)
        self.num_channels = self.shape[1]

        self.static_fwd_desire = []
        self.static_rev_desire = []
        for place in self.places:
            fwd_desire, rev_desire = self.get_eager_desire(place)
            self.fwd_desire.append(fwd_desire.numpy())
            self.rev_desire.append(rev_desire.numpy())
            self.static_fwd_desire.append([])
            self.static_rev_desire.append([])
            fwd, rev = self.get_static_desire(place)
            self.static_fwd_desire[-1].append(fwd[0])
            self.static_fwd_desire[-1].append(fwd[1])
            self.static_fwd_desire[-1].append(fwd[2])
            self.static_rev_desire[-1].append(rev[0])
            self.static_rev_desire[-1].append(rev[1])
            self.static_rev_desire[-1].append(rev[2])

    def get_eager_desire(self, place):
        if isinstance(place, base.CPUPlace):
            paddle.set_device("cpu")
        if isinstance(place, base.CUDAPlace):
            paddle.set_device("gpu")
        core.set_prim_eager_enabled(False)
        paddle.disable_static()
        input_ = paddle.to_tensor(
            data=self.x, dtype=self.dtype, place=place, stop_gradient=False
        )
        scale_ = paddle.to_tensor(
            data=self.scale, dtype=self.dtype, place=place, stop_gradient=False
        )
        bias_ = paddle.to_tensor(
            data=self.bias, dtype=self.dtype, place=place, stop_gradient=False
        )
        output = paddle.nn.functional.instance_norm(
            input_, None, None, scale_, bias_, True, 0.9, self.epsilon
        )
        grad = paddle.grad(output, input_)

        return output, grad[0]

    def get_static_desire(self, place):
        core._set_prim_all_enabled(False)
        paddle.enable_static()
        if isinstance(place, base.CPUPlace):
            paddle.set_device("cpu")
        if isinstance(place, base.CUDAPlace):
            paddle.set_device("gpu")

        mp, sp = paddle.static.Program(), paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            input_ = paddle.static.data(
                'x', shape=self.x.shape, dtype=self.x.dtype
            )
            input_.stop_gradient = False

            scale_ = paddle.static.data(
                'scale_', shape=self.scale.shape, dtype=self.scale.dtype
            )
            scale_.stop_gradient = False

            bias_ = paddle.static.data(
                'bias_', shape=self.bias.shape, dtype=self.bias.dtype
            )
            bias_.stop_gradient = False

            output = paddle.nn.functional.instance_norm(
                input_, None, None, scale_, bias_, True, 0.9, self.epsilon
            )

            blocks = mp.blocks
            names = dict(
                zip(
                    blocks[0].ops[0].output_names,
                    blocks[0].ops[0].output_arg_names,
                )
            )
            vars_list = [
                names[key]
                for key in [
                    "Y",
                    "SavedMean",
                    "SavedVariance",
                ]
            ]

            fwd_ops = [op.type for op in blocks[0].ops]
            # Ensure that instance_norm in original block
            assert 'instance_norm' in fwd_ops

            if core._is_fwd_prim_enabled():
                paddle.incubate.autograd.primapi.to_prim(mp.blocks)
                fwd_ops_new = [op.type for op in blocks[0].ops]
                # Ensure that instance_norm is splitted into small ops
                assert 'instance_norm' not in fwd_ops_new

            grads = paddle.static.gradients([output], [input_, scale_, bias_])

        exe = paddle.static.Executor(place)
        exe.run(sp)
        out_list = exe.run(
            mp,
            feed={
                input_.name: self.x,
                scale_.name: self.scale,
                bias_.name: self.bias,
            },
            fetch_list=vars_list + [grads],
        )
        paddle.disable_static()
        core._set_prim_all_enabled(True)

        return out_list[:3], out_list[3:]

    def test_static_comp(self):
        paddle.enable_static()
        mps = []
        fwd_actual = []
        rev_actual = []
        if len(self.places) < 1:
            return

        with static_guard():
            for place in self.places:
                fwd_actual.append([])
                rev_actual.append([])
                mp, sp = paddle.static.Program(), paddle.static.Program()
                with paddle.static.program_guard(mp, sp):
                    input_ = paddle.static.data(
                        'x', shape=self.x.shape, dtype=self.x.dtype
                    )
                    input_.stop_gradient = False

                    scale_ = paddle.static.data(
                        'scale_', shape=self.scale.shape, dtype=self.scale.dtype
                    )
                    scale_.stop_gradient = False

                    bias_ = paddle.static.data(
                        'bias_', shape=self.bias.shape, dtype=self.bias.dtype
                    )
                    bias_.stop_gradient = False

                    output = paddle.nn.functional.instance_norm(
                        input_,
                        None,
                        None,
                        scale_,
                        bias_,
                        True,
                        0.9,
                        self.epsilon,
                    )

                    blocks = mp.blocks
                    names = dict(
                        zip(
                            blocks[0].ops[0].output_names,
                            blocks[0].ops[0].output_arg_names,
                        )
                    )
                    vars_list = [
                        names[key]
                        for key in [
                            "Y",
                            "SavedMean",
                            "SavedVariance",
                        ]
                    ]

                    fwd_ops = [op.type for op in blocks[0].ops]
                    # Ensure that instance_norm in original block
                    assert 'instance_norm' in fwd_ops

                    if core._is_fwd_prim_enabled():
                        paddle.incubate.autograd.primapi.to_prim(mp.blocks)
                        fwd_ops_new = [op.type for op in blocks[0].ops]
                        # Ensure that instance_norm is splitted into small ops
                        assert 'instance_norm' not in fwd_ops_new

                    grads = paddle.static.gradients(
                        output, [input_, scale_, bias_]
                    )
                exe = paddle.static.Executor(place)
                exe.run(sp)
                out_list = exe.run(
                    mp,
                    feed={
                        input_.name: self.x,
                        scale_.name: self.scale,
                        bias_.name: self.bias,
                    },
                    fetch_list=vars_list + [grads],
                )
                fwd_actual[-1].append(out_list[0])
                fwd_actual[-1].append(out_list[1])
                fwd_actual[-1].append(out_list[2])
                rev_actual[-1].append(out_list[3])
                rev_actual[-1].append(out_list[4])
                rev_actual[-1].append(out_list[5])
                mps.append(mp)

        vars_name = [
            "Y",
            "SavedMean",
            "SavedVariance",
            "X_grad",
            "Scale_grad",
            "Bias_grad",
        ]

        for i in range(len(self.places)):
            self.assertTrue(
                'instance_norm' not in [op.type for op in mps[i].block(0).ops]
            )
            atol = self.threshold_list[i][0]
            rtol = self.threshold_list[i][0]
            for j in range(len(self.static_fwd_desire[i])):
                # in float16 type, Y is float16, mean and var are float16
                # so check mean and var with float32 gpu threshold
                if self.dtype == 'float16' and j > 0:
                    atol = 1e-5
                    rtol = 1e-5

                np.testing.assert_allclose(
                    self.static_fwd_desire[i][j],
                    fwd_actual[i][j],
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"Check diff failed of place:{self.places[i]}, output: {vars_name[j]}",
                )
                max_abs_diff = np.max(
                    np.abs(self.static_fwd_desire[i][j] - fwd_actual[i][j])
                )
                print(
                    self.shape,
                    self.dtype,
                    self.places[i],
                    vars_name[j],
                    max_abs_diff,
                )
            # compare with eager_desire
            np.testing.assert_allclose(
                self.fwd_desire[i],
                fwd_actual[i][0],
                rtol=rtol,
                atol=atol,
                err_msg=f"Check diff failed with fwd_eager:{self.places[i]}",
            )

            for j in range(len(self.static_rev_desire[i])):
                if self.special_threshold is not None and j <= 1:
                    atol = self.special_threshold[i]
                    rtol = self.special_threshold[i]
                else:
                    atol = self.threshold_list[i][0]
                    rtol = self.threshold_list[i][0]

                max_abs_diff = np.max(
                    np.abs(self.static_rev_desire[i][j] - rev_actual[i][j])
                )

                print(
                    self.shape,
                    self.dtype,
                    self.places[i],
                    vars_name[j + 3],
                    max_abs_diff,
                )

                np.testing.assert_allclose(
                    self.static_rev_desire[i][j],
                    rev_actual[i][j],
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"Check diff failed of place:{self.places[i]}, output: {vars_name[j + 3]}",
                )

            # now use larger threshold when testing cpu grads to bypass cpu grad test
            if self.special_threshold is not None and i == 0:
                atol = self.special_threshold[i]
                rtol = self.special_threshold[i]
            # compare with eager_desire
            np.testing.assert_allclose(
                self.rev_desire[i],
                rev_actual[i][0],
                rtol=rtol,
                atol=atol,
                err_msg=f"Check diff failed with rev_eager:{self.places[i]}",
            )

        paddle.disable_static()

    def test_jit_comp(self):
        fwd_actual = []
        rev_actual = []
        for place in self.places:
            input_ = paddle.to_tensor(
                data=self.x, dtype=self.dtype, place=place, stop_gradient=False
            )
            scale_ = paddle.to_tensor(
                data=self.scale,
                dtype=self.dtype,
                place=place,
                stop_gradient=False,
            )
            bias_ = paddle.to_tensor(
                data=self.bias,
                dtype=self.dtype,
                place=place,
                stop_gradient=False,
            )
            net = PrimGroupNorm(self.num_channels, scale_, bias_)
            net = apply_to_static(net, False)
            output = net(input_)

            grad = paddle.grad(output, input_)
            fwd_actual.append(output.numpy())
            rev_actual.append(grad[0].numpy())

        for i in range(len(self.places)):
            atol = self.threshold_list[i][1]
            rtol = self.threshold_list[i][1]
            np.testing.assert_allclose(
                self.fwd_desire[i],
                fwd_actual[i],
                rtol=rtol,
                atol=atol,
                err_msg=f'{self.places[i]} jit fwd',
            )

            # now use larger threshold when testing cpu grads to bypass cpu grad test
            if self.special_threshold is not None:
                atol = self.special_threshold[i]
                rtol = self.special_threshold[i]

            np.testing.assert_allclose(
                self.rev_desire[i],
                rev_actual[i],
                rtol=rtol,
                atol=atol,
                err_msg=f'{self.places[i]} jit rev',
            )

    def test_jit_comp_with_cinn(self):
        fwd_actual = []
        rev_actual = []
        for place in self.places:
            input_ = paddle.to_tensor(
                data=self.x, dtype=self.dtype, place=place, stop_gradient=False
            )
            scale_ = paddle.to_tensor(
                data=self.scale,
                dtype=self.dtype,
                place=place,
                stop_gradient=False,
            )
            bias_ = paddle.to_tensor(
                data=self.bias,
                dtype=self.dtype,
                place=place,
                stop_gradient=False,
            )
            net = PrimGroupNorm(self.num_channels, scale_, bias_)
            net = apply_to_static(net, True)
            output = net(input_)
            grad = paddle.grad(output, input_)
            fwd_actual.append(output.numpy())
            rev_actual.append(grad[0].numpy())

        for i in range(len(self.places)):
            atol = self.threshold_list[i][2]
            rtol = self.threshold_list[i][2]
            np.testing.assert_allclose(
                self.fwd_desire[i],
                fwd_actual[i],
                rtol=rtol,  # mean of uniform distribution, scale for avoid random failed
                atol=atol,
                err_msg=f'{self.places[i]} jit_cinn fwd',
            )
            # now use larger threshold when testing cpu grads to bypass cpu grad test
            if self.special_threshold is not None:
                atol = self.special_threshold[i]
                rtol = self.special_threshold[i]
            np.testing.assert_allclose(
                self.rev_desire[i],
                rev_actual[i],
                rtol=rtol,  # mean of uniform distribution, scale for avoid random failed
                atol=atol,
                err_msg=f'{self.places[i]} jit_cinn rev',
            )


class TestInstanceNormCase1(TestInstanceNormOp):
    def init_test_case(self):
        x_shape = [2, 100, 4, 5]
        n, c, h, w = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        self.epsilon = 1e-05
        dtype = np.float32
        scale_shape = [c]
        mean_shape = [n * c]
        np.random.seed()
        self.x_np = np.random.random_sample(x_shape).astype(dtype)
        self.scale_np = np.ones(scale_shape).astype(dtype)
        self.bias_np = np.zeros(scale_shape).astype(dtype)
        self.mean_np, self.var_np = _cal_mean_variance(
            self.x_np, self.epsilon, mean_shape
        )


class TestElasticNormOp(unittest.TestCase):
    def init_test_case(self):
        self.epsilon = 1e-5
        self.places = [core.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu(
            "instance_norm"
        ):
            self.places.append(core.CUDAPlace(0))

    def test_norm(self):
        self.init_test_case()
        inputs = np.random.random((2, 3, 5, 5)).astype(np.float32)
        shape = inputs.shape
        n, c, h, w = shape[0], shape[1], shape[2], shape[3]
        scale_shape = [c]
        mean_shape = [n * c]
        scale = np.ones(scale_shape).astype(np.float32)
        bias = np.zeros(scale_shape).astype(np.float32)
        mean, variance = _cal_mean_variance(inputs, self.epsilon, mean_shape)
        out_np, _, _ = _reference_instance_norm_naive(
            inputs, scale, bias, self.epsilon, mean, variance
        )

        for place in self.places:
            with base.dygraph.guard(place):
                instance_norm = paddle.nn.InstanceNorm2D(
                    5, weight_attr=False, bias_attr=False
                )
                outputs = instance_norm(paddle.to_tensor(inputs))
                np.testing.assert_allclose(
                    outputs.numpy(), out_np, rtol=1e-05, atol=1e-06
                )


class TestElasticNormOpCase2(unittest.TestCase):
    def init_test_case(self):
        self.epsilon = 1e-5
        self.places = [core.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu(
            "instance_norm"
        ):
            self.places.append(core.CUDAPlace(0))

    def test_norm(self):
        self.init_test_case()
        inputs = np.random.random((2, 3, 5, 5)).astype(np.float32)
        shape = inputs.shape
        n, c, h, w = shape[0], shape[1], shape[2], shape[3]
        scale_shape = [c]
        mean_shape = [n * c]
        scale = np.ones(scale_shape).astype(np.float32)
        bias = np.zeros(scale_shape).astype(np.float32)
        mean, variance = _cal_mean_variance(inputs, self.epsilon, mean_shape)
        out_np, _, _ = _reference_instance_norm_naive(
            inputs, scale, bias, self.epsilon, mean, variance
        )

        for place in self.places:
            with base.dygraph.guard(place):
                instance_norm = paddle.nn.InstanceNorm2D(
                    3, weight_attr=True, bias_attr=True
                )
                outputs = instance_norm(paddle.to_tensor(inputs))
                np.testing.assert_allclose(
                    outputs.numpy(), out_np, rtol=1e-05, atol=1e-06
                )


if __name__ == '__main__':
    unittest.main()
