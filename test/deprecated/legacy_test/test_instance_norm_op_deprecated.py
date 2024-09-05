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

import os
import unittest

import numpy as np
import parameterized as param
from utils import static_guard

import paddle
from paddle import base, nn
from paddle.base import Program, core, program_guard


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


class TestInstanceNormOpTraining(unittest.TestCase):
    def setUp(self):
        self.epsilon = 1e-5
        self.init_test_case()

    def init_test_case(self):
        self.shape = [2, 3, 4, 5]
        self.no_grad_set = set()
        self.fetch_list = [
            'y',
            'saved_mean',
            'saved_variance',
            'x@GRAD',
            'scale@GRAD',
            'bias@GRAD',
        ]

    def __assert_close(self, tensor, np_array, msg, atol=1e-4):
        np.testing.assert_allclose(
            np.array(tensor), np_array, rtol=1e-05, atol=atol, err_msg=msg
        )

    def set_global_mean_var(self, mean_shape, x):
        mean, variance = _cal_mean_variance(x, self.epsilon, mean_shape)
        return mean, variance

    def test_forward_backward(self):
        def test_with_place(place, shape):
            paddle.enable_static()
            epsilon = self.epsilon
            n, c, h, w = shape[0], shape[1], shape[2], shape[3]
            scale_shape = [c]
            mean_shape = [n * c]

            np.random.seed()
            x = np.random.random_sample(shape).astype(np.float32)
            scale = np.random.random_sample(scale_shape).astype(np.float32)
            bias = np.random.random_sample(scale_shape).astype(np.float32)
            mean, variance = self.set_global_mean_var(mean_shape, x)
            d_y = np.random.random_sample(shape).astype(np.float32)

            y, saved_mean, variance_tmp = _reference_instance_norm_naive(
                x, scale, bias, epsilon, mean, variance
            )

            saved_variance = 1 / np.sqrt(variance_tmp + epsilon)

            d_x, d_scale, d_bias = _reference_instance_norm_grad(
                x, d_y, scale, saved_mean, saved_variance, epsilon
            )

            var_dict = locals()
            var_dict['y@GRAD'] = d_y
            var_dict['x@GRAD'] = d_x
            var_dict['scale@GRAD'] = d_scale
            var_dict['bias@GRAD'] = d_bias

            var_names = [
                'x',
                'scale',
                'bias',
                'y',
                'saved_mean',
                'saved_variance',
            ]
            ground_truth = {name: var_dict[name] for name in var_names}

            program = base.Program()
            with base.program_guard(program):
                block = program.global_block()
                for name in ground_truth:
                    block.create_var(
                        name=name,
                        dtype='float32',
                        shape=ground_truth[name].shape,
                    )
                in_op = block.append_op(
                    type="instance_norm",
                    inputs={
                        "X": block.var("x"),
                        "Scale": block.var("scale"),
                        "Bias": block.var("bias"),
                    },
                    outputs={
                        "Y": block.var("y"),
                        "SavedMean": block.var("saved_mean"),
                        "SavedVariance": block.var("saved_variance"),
                    },
                    attrs={
                        "epsilon": epsilon,
                    },
                )

                block.create_var(name="y@GRAD", dtype='float32', shape=y.shape)

                grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(
                    in_op.desc, self.no_grad_set, []
                )
                grad_op_desc = grad_op_desc_list[0]
                new_op_desc = block.desc.append_op()
                new_op_desc.copy_from(grad_op_desc)
                for var_name in grad_op_desc.output_arg_names():
                    block.desc.var(var_name.encode("ascii"))
                grad_op_desc.infer_var_type(block.desc)
                grad_op_desc.infer_shape(block.desc)
                for arg in grad_op_desc.output_arg_names():
                    grad_var = block.desc.find_var(arg.encode("ascii"))
                    grad_var.set_dtype(core.VarDesc.VarType.FP32)

                program._sync_with_cpp()

                exe = base.Executor(place)
                out = exe.run(
                    program,
                    feed={
                        name: var_dict[name]
                        for name in ['x', 'scale', 'bias', 'y@GRAD']
                    },
                    fetch_list=self.fetch_list,
                )

            for id, name in enumerate(self.fetch_list):
                self.__assert_close(var_dict[name], out[id], name)
            print("op test forward passes: ", str(place))
            paddle.disable_static()

        places = []
        if os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower() in [
            '1',
            'true',
            'on',
        ] or not (
            core.is_compiled_with_cuda()
            and core.op_support_gpu("instance_norm")
        ):
            places.append(core.CPUPlace())
        if core.is_compiled_with_cuda() and core.op_support_gpu(
            "instance_norm"
        ):
            places.append(core.CUDAPlace(0))
        for place in places:
            test_with_place(place, self.shape)


class TestInstanceNormOpTrainingCase1(TestInstanceNormOpTraining):
    def init_test_case(self):
        self.shape = [2, 3, 4, 5]
        self.no_grad_set = {'scale@GRAD', 'bias@GRAD'}
        self.fetch_list = ['y', 'saved_mean', 'saved_variance', 'x@GRAD']


class TestInstanceNormOpTrainingCase2(TestInstanceNormOpTraining):
    def init_test_case(self):
        self.shape = [20, 50, 4, 5]
        self.no_grad_set = {'scale@GRAD', 'bias@GRAD'}
        self.fetch_list = ['y', 'saved_mean', 'saved_variance', 'x@GRAD']


class TestInstanceNormOpError(unittest.TestCase):
    def test_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            # the input of instance_norm must be Variable.
            x1 = base.create_lod_tensor(
                np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], base.CPUPlace()
            )
            self.assertRaises(TypeError, paddle.static.nn.instance_norm, x1)

            # the input dtype of instance_norm must be float32 or float64
            x2 = paddle.static.data(
                name='x2', shape=[-1, 3, 4, 5, 6], dtype="int32"
            )
            self.assertRaises(TypeError, paddle.static.nn.instance_norm, x2)
        paddle.disable_static()


class TestInstanceNormOpErrorCase1(unittest.TestCase):
    def test_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            # the first dimension of input for instance_norm must between [2d, 5d]
            x = paddle.static.data(name='x', shape=[3], dtype="float32")
            self.assertRaises(ValueError, paddle.static.nn.instance_norm, x)
        paddle.disable_static()


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
            fetch_list=[*vars_list, grads],
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
                    fetch_list=[*vars_list, grads],
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


if __name__ == '__main__':
    unittest.main()
