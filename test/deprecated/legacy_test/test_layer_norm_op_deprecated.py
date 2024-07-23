#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from functools import reduce
from operator import mul

import numpy as np
from op_test import _set_use_system_allocator

import paddle
from paddle import base
from paddle.base import core

paddle.enable_static()

np.random.seed(123)
paddle.seed(123)

_set_use_system_allocator(True)


def _reference_layer_norm_naive(x, scale, beta, epsilon, begin_norm_axis=1):
    x_shape = x.shape
    N = reduce(mul, x_shape[0:begin_norm_axis], 1)
    D = reduce(mul, x_shape[begin_norm_axis : len(x_shape)], 1)
    x.shape = [N, D]

    mean = np.mean(x, axis=1)
    var = np.var(x, axis=1) + epsilon
    output = np.divide(
        (x - mean.reshape([N, 1])), (np.sqrt(var)).reshape([N, 1])
    )
    if scale is not None:
        output = scale.reshape([1, D]) * output
    if beta is not None:
        output = output + beta.reshape([1, D])

    x.shape, output.shape = x_shape, x_shape
    return output, mean, var


def _reference_layer_norm_grad(
    x, grad_y, scale, bias, mean, var, begin_norm_axis=1
):
    x_shape = x.shape
    N = reduce(mul, x_shape[0:begin_norm_axis], 1)
    D = reduce(mul, x_shape[begin_norm_axis : len(x_shape)], 1)

    if scale is not None:
        scale_shape = scale.shape
        scale.shape = [1, D]
    x.shape, grad_y.shape = [N, D], [N, D]
    var.shape, mean.shape = [N, 1], [N, 1]

    # d_bias
    if bias is not None:
        d_bias = np.sum(grad_y, axis=0).reshape([1, D])
    else:
        d_bias = None
    # d_scale
    if scale is not None:
        d_scale = np.sum(
            ((x - mean) * np.sqrt(1 / var)) * grad_y, axis=0
        ).reshape([1, D])
    else:
        d_scale = None
    # dx
    if scale is not None:
        dx_end = scale * np.sqrt(1.0 / var) * grad_y
        d_mean_0 = np.sum(-np.sqrt(1.0 / var) * grad_y * scale, axis=1).reshape(
            [N, 1]
        )  # the second part equals to zero.
        d_mean = 1.0 / D * d_mean_0
        d_std = np.sum(
            -(1.0 / var) * (x - mean) * grad_y * scale, axis=1
        ).reshape([N, 1]) * (
            1.0 / D * np.sqrt(1.0 / var).reshape([N, 1]) * (x - mean)
        )
    else:
        dx_end = 1.0 * np.sqrt(1.0 / var) * grad_y
        d_mean_0 = np.sum(-np.sqrt(1.0 / var) * grad_y * 1.0, axis=1).reshape(
            [N, 1]
        )  # the second part equals to zero.
        d_mean = 1.0 / D * d_mean_0
        d_std = np.sum(
            -(1.0 / var) * (x - mean) * grad_y * 1.0, axis=1
        ).reshape([N, 1]) * (
            1.0 / D * np.sqrt(1.0 / var).reshape([N, 1]) * (x - mean)
        )

    grad_x = dx_end + d_mean + d_std

    grad_x.shape, x.shape, grad_y.shape = x_shape, x_shape, x_shape
    var.shape, mean.shape = [N], [N]

    if scale is not None:
        scale.shape = scale_shape
    return grad_x, d_scale, d_bias


def layer_norm_wrapper(
    x, scale=None, bias=None, epsilon=1e-05, begin_norm_axis=1
):
    input_shape = list(x.shape)
    normalized_shape = input_shape[begin_norm_axis:]
    return paddle.nn.functional.layer_norm(
        x, normalized_shape, weight=scale, bias=bias, epsilon=epsilon
    )


class TestLayerNormOp(unittest.TestCase):
    def setUp(self):
        self.use_cudnn = True
        paddle.enable_static()

    def __assert_close(self, tensor, np_array, msg, atol=1e-4):
        np.testing.assert_allclose(
            np.array(tensor).flatten(),
            np_array.flatten(),
            rtol=1e-3,
            atol=atol,
            err_msg=msg,
        )

    def check_forward_backward(
        self,
        shape,
        begin_norm_axis,
        has_scale=True,
        has_bias=True,
        y_grad_scale=1.0,
        use_mkldnn=False,
    ):
        def test_with_place(
            place, shape, begin_norm_axis, use_mkldnn=use_mkldnn
        ):
            # attr
            epsilon = 0.00001
            x_shape = shape
            D = reduce(mul, x_shape[begin_norm_axis : len(x_shape)], 1)
            scale_shape = [D]

            np.random.seed(123)
            x = np.random.random_sample(x_shape).astype(np.float32)
            scale = (
                np.random.random_sample(scale_shape).astype(np.float32)
                if has_scale
                else None
            )
            bias = (
                np.random.random_sample(scale_shape).astype(np.float32)
                if has_bias
                else None
            )
            y_grad = (np.random.random_sample(x_shape) * y_grad_scale).astype(
                np.float32
            )

            # reference forward & backward
            y, mean, variance = _reference_layer_norm_naive(
                x, scale, bias, epsilon, begin_norm_axis
            )
            x_grad, scale_grad, bias_grad = _reference_layer_norm_grad(
                x, y_grad, scale, bias, mean, variance, begin_norm_axis
            )

            var_dict = locals()
            var_dict['y@GRAD'] = y_grad
            var_names = ['x', 'mean', 'variance', 'y', 'y@GRAD']
            if has_scale:
                var_names += ['scale']
            if has_bias:
                var_names += ['bias']
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
                inputs = {"X": block.var('x')}
                fetch_list = [
                    'y',
                    'mean',
                    'variance',
                    'x@GRAD',
                ]
                if has_scale:
                    inputs["Scale"] = block.var('scale')
                    fetch_list += ['scale@GRAD']
                if has_bias:
                    inputs["Bias"] = block.var('bias')
                    fetch_list += ['bias@GRAD']
                layer_norm_op = block.append_op(
                    type="layer_norm",
                    inputs=inputs,
                    outputs={
                        "Y": block.var('y'),
                        "Mean": block.var('mean'),  # share the same memory
                        "Variance": block.var(
                            'variance'
                        ),  # share the same memory
                    },
                    attrs={
                        "epsilon": epsilon,
                        "begin_norm_axis": begin_norm_axis,
                        "use_mkldnn": use_mkldnn,
                    },
                )
                # generate backward op_desc
                grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(
                    layer_norm_op.desc, set(), []
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
                name_list = ['x', 'y@GRAD']
                if has_scale:
                    name_list += ['scale']
                if has_bias:
                    name_list += ['bias']

                out = exe.run(
                    program,
                    feed={name: var_dict[name] for name in name_list},
                    fetch_list=fetch_list,
                )
                # print(y)
                # print(out[0])
                self.__assert_close(y, out[0], "y")
                self.__assert_close(mean, out[1], "mean")
                self.__assert_close(variance, out[2], "variance", 1e-3)
                self.__assert_close(x_grad, out[3], "x_grad")
                if has_scale:
                    self.__assert_close(
                        scale_grad,
                        out[fetch_list.index('scale@GRAD')],
                        "scale_grad",
                        1e-3,
                    )
                if has_bias:
                    self.__assert_close(
                        bias_grad,
                        out[fetch_list.index('bias@GRAD')],
                        "bias_grad",
                    )

        places = [core.CPUPlace()]
        if (
            core.is_compiled_with_cuda()
            and core.op_support_gpu("layer_norm")
            and self.use_cudnn
        ):
            places.append(core.CUDAPlace(0))

        for place in places:
            test_with_place(place, shape, begin_norm_axis)

    def test_check_forward_backward_with_scale_and_bias(self):
        self.check_forward_backward(shape=[2, 3, 4, 5], begin_norm_axis=1)
        self.check_forward_backward(shape=[1, 3, 4, 5], begin_norm_axis=1)
        self.check_forward_backward(
            shape=[2, 3, 4, 5],
            begin_norm_axis=1,
            has_scale=False,
            has_bias=True,
        )
        self.check_forward_backward(
            shape=[2, 3, 4, 5],
            begin_norm_axis=1,
            has_scale=True,
            has_bias=False,
        )
        self.check_forward_backward(
            shape=[2, 3, 4, 5],
            begin_norm_axis=1,
            has_scale=False,
            has_bias=False,
        )
        self.check_forward_backward(shape=[2, 3, 4, 5], begin_norm_axis=3)
        self.check_forward_backward(
            shape=[92, 513, 129], begin_norm_axis=2, y_grad_scale=0.1
        )
        self.check_forward_backward(shape=[3, 34, 1134], begin_norm_axis=2)
        self.check_forward_backward(shape=[3, 2, 1133], begin_norm_axis=2)
        self.check_forward_backward(
            shape=[92, 513, 1134], begin_norm_axis=2, y_grad_scale=0.1
        )
        self.check_forward_backward(
            shape=[92, 513, 1134],
            begin_norm_axis=2,
            has_scale=False,
            has_bias=True,
            y_grad_scale=0.1,
        )
        self.check_forward_backward(
            shape=[92, 513, 1134],
            begin_norm_axis=2,
            has_scale=True,
            has_bias=False,
            y_grad_scale=0.1,
        )
        self.check_forward_backward(
            shape=[92, 513, 1134],
            begin_norm_axis=2,
            has_scale=False,
            has_bias=False,
            y_grad_scale=0.1,
        )
        self.check_forward_backward(
            shape=[512, 1024], begin_norm_axis=1, has_scale=True, has_bias=True
        )
        self.check_forward_backward(
            shape=[1, 128, 256, 256],
            begin_norm_axis=3,
            has_scale=True,
            has_bias=True,
        )
        self.check_forward_backward(
            shape=[1, 256, 384],
            begin_norm_axis=2,
            has_scale=True,
            has_bias=True,
        )


class TestLayerNormAPI(unittest.TestCase):
    def test_case(self):
        x = paddle.static.data(name='x', shape=[64, 32, 256], dtype='float32')
        x = paddle.static.nn.layer_norm(
            x,
            scale=True,
            shift=True,
            begin_norm_axis=1,
            epsilon=1e-05,
            param_attr=None,
            bias_attr=None,
        )
        x = paddle.static.nn.layer_norm(
            x,
            scale=False,
            shift=False,
            begin_norm_axis=1,
            epsilon=1e-05,
            param_attr=None,
            bias_attr=None,
        )
        x = paddle.static.nn.layer_norm(
            x,
            scale=True,
            shift=True,
            begin_norm_axis=1,
            epsilon=1e-05,
            param_attr="scale",
            bias_attr="shift",
        )
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



if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
