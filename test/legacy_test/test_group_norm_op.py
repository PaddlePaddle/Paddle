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

import numpy as np
import parameterized as param
from eager_op_test import (
    OpTest,
    convert_float_to_uint16,
    convert_uint16_to_float,
    paddle_static_guard,
    skip_check_grad_ci,
)
from testsuite import create_op

import paddle
from paddle import base
from paddle.base import core


def group_norm_naive(x, scale, bias, epsilon, groups, data_layout):
    if data_layout == "NHWC":
        x = np.transpose(x, (0, 3, 1, 2))  # NHWC => NCHW
    N, C, H, W = x.shape
    G = groups
    x = x.reshape((N * G, -1))
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    output = (x - mean) / np.sqrt(var + epsilon)
    output = output.reshape((N, C, H, W)) * scale.reshape(
        (-1, 1, 1)
    ) + bias.reshape((-1, 1, 1))
    if data_layout == "NHWC":
        output = np.transpose(output, (0, 2, 3, 1))  # NCHW => NHWC
    return output, mean.reshape((N, G)), var.reshape((N, G))


class TestGroupNormOpError(unittest.TestCase):
    def test_errors(self):
        with paddle_static_guard():
            with base.program_guard(base.Program(), base.Program()):

                def test_x_type():
                    input = np.random.random(2, 100, 3, 5).astype('float32')
                    groups = 2
                    paddle.static.nn.group_norm(input, groups)

                self.assertRaises(TypeError, test_x_type)

                def test_x_dtype():
                    x2 = paddle.static.data(
                        name='x2', shape=[-1, 2, 100, 3, 5], dtype='int32'
                    )
                    groups = 2
                    paddle.static.nn.group_norm(x2, groups)

                self.assertRaises(TypeError, test_x_dtype)


def group_norm_wrapper(
    input, weight, bias, epsilon=1e-5, num_groups=0, data_format="NCHW"
):
    if data_format == "AnyLayout":
        data_format = "NCDHW"
    return paddle._C_ops.group_norm(
        input, weight, bias, epsilon, num_groups, data_format
    )


class TestGroupNormOp(OpTest):
    def setUp(self):
        self.op_type = "group_norm"
        self.python_api = group_norm_wrapper
        self.python_out_sig = ["Y"]
        self.data_format = "NCHW"
        self.dtype = np.float64
        self.shape = (2, 100, 3, 5)
        self.attrs = {'epsilon': 1e-5, 'groups': 2, 'data_layout': "NCHW"}
        self.compare_between_place = False
        self.init_test_case()

        input = np.random.random(self.shape).astype(self.dtype)
        if self.data_format == "NHWC":
            input = np.transpose(input, (0, 2, 3, 1))
        scale = np.random.random([self.shape[1]]).astype(self.dtype)
        bias = np.random.random([self.shape[1]]).astype(self.dtype)
        output, mean, var = group_norm_naive(
            input,
            scale,
            bias,
            self.attrs['epsilon'],
            self.attrs['groups'],
            self.data_format,
        )

        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(input),
            'Scale': OpTest.np_dtype_to_base_dtype(scale),
            'Bias': OpTest.np_dtype_to_base_dtype(bias),
        }
        self.outputs = {'Y': output, 'Mean': mean, 'Variance': var}
        self.attrs['data_layout'] = self.data_format

    def test_check_output(self):
        atol = 0
        inplace_atol = 0
        place = core.CPUPlace()

        self.check_output_with_place(place, atol=atol)

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            # group_norm uses AtomicAdd on CUDAPlace, which do not ensure
            # computation order when multiple threads write the same address. So the
            # result of group_norm is non-deterministic when datatype is float.
            # When inplace_atol is not None, the inplace check uses numpy.allclose
            # to check inplace result instead of numpy.array_equal.
            # Set to inplace_atol to 0, which means the absolute error is 0, and the
            # relative error is 1e-05 in numpy.allclose by default.
            # Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html
            self.check_output_with_place(
                place, atol=atol, inplace_atol=inplace_atol
            )

    def do_compare_between_place(self):
        if not core.is_compiled_with_cuda():
            return
        place = core.CPUPlace()
        place2 = core.CUDAPlace(0)
        self.scope = core.Scope()
        op_inputs = self.inputs if hasattr(self, "inputs") else {}
        op_outputs = self.outputs if hasattr(self, "outputs") else {}
        op_attrs = self.attrs if hasattr(self, "attrs") else {}
        self.op = create_op(
            self.scope, self.op_type, op_inputs, op_outputs, op_attrs
        )
        inputs_to_check = {'X', 'Scale', 'Bias'}
        output_names = 'Y'
        cpu_grads = self._get_gradient(
            inputs_to_check, place, output_names, None
        )
        gpu_grads = self._get_gradient(
            inputs_to_check, place2, output_names, None
        )
        self._assert_is_close(
            cpu_grads,
            gpu_grads,
            inputs_to_check,
            0.005,
            "Gradient Check On %s" % str(place),
        )

    def test_check_grad(self):
        if self.compare_between_place:
            self.do_compare_between_place()
            return

        place = core.CPUPlace()
        self.check_grad_with_place(place, {'X', 'Scale', 'Bias'}, 'Y')
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                {'X', 'Scale', 'Bias'},
                'Y',
            )

    def init_test_case(self):
        pass


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_float16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestGroupNormFP16OP(TestGroupNormOp):
    def test_check_output(self):
        atol = 1e-3
        inplace_atol = 1e-3

        place = core.CUDAPlace(0)
        # group_norm uses AtomicAdd on CUDAPlace, which do not ensure
        # computation order when multiple threads write the same address. So the
        # result of group_norm is non-deterministic when datatype is float.
        # When inplace_atol is not None, the inplace check uses numpy.allclose
        # to check inplace result instead of numpy.array_equal.
        # Set to inplace_atol to 0, which means the absolute error is 0, and the
        # relative error is 1e-05 in numpy.allclose by default.
        # Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html
        self.check_output_with_place(place)

    def test_check_grad(self):
        if self.compare_between_place:
            return

        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, {'X', 'Scale', 'Bias'}, 'Y')

    def init_test_case(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestGroupNormBF16Op(OpTest):
    def setUp(self):
        self.op_type = "group_norm"
        self.python_api = group_norm_wrapper
        self.python_out_sig = ["Y"]
        self.data_format = "NCHW"
        self.dtype = np.uint16
        self.shape = (2, 100, 3, 5)
        self.attrs = {'epsilon': 1e-5, 'groups': 2, 'data_layout': "NCHW"}
        self.compare_between_place = False
        self.init_test_case()

        input = np.random.random(self.shape).astype(np.float32)
        if self.data_format == "NHWC":
            input = np.transpose(input, (0, 2, 3, 1))
        scale = np.random.random([self.shape[1]]).astype(np.float32)
        bias = np.random.random([self.shape[1]]).astype(np.float32)
        output, mean, var = group_norm_naive(
            input,
            scale,
            bias,
            self.attrs['epsilon'],
            self.attrs['groups'],
            self.data_format,
        )

        self.inputs = {
            'X': convert_float_to_uint16(input),
            'Scale': convert_float_to_uint16(scale),
            'Bias': convert_float_to_uint16(bias),
        }
        self.outputs = {'Y': output, 'Mean': mean, 'Variance': var}
        self.attrs['data_layout'] = self.data_format

    def test_check_output(self):
        atol = 1e-2
        inplace_atol = 1e-2

        place = core.CUDAPlace(0)
        # group_norm uses AtomicAdd on CUDAPlace, which do not ensure
        # computation order when multiple threads write the same address. So the
        # result of group_norm is non-deterministic when datatype is float.
        # When inplace_atol is not None, the inplace check uses numpy.allclose
        # to check inplace result instead of numpy.array_equal.
        # Set to inplace_atol to 0, which means the absolute error is 0, and the
        # relative error is 1e-05 in numpy.allclose by default.
        # Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html
        self.check_output_with_place(place)

    def test_check_grad(self):
        if self.compare_between_place:
            return

        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, {'X', 'Scale', 'Bias'}, 'Y')

    def init_test_case(self):
        pass


class TestGroupNormOp1(TestGroupNormOp):
    def init_test_case(self):
        self.attrs['groups'] = 1


class TestGroupNormFP16Op1(TestGroupNormFP16OP):
    def init_test_case(self):
        self.attrs['groups'] = 1
        self.dtype = np.float16


class TestGroupNormBF16Op1(TestGroupNormBF16Op):
    def init_test_case(self):
        self.attrs['groups'] = 1


class TestGroupNormOp2(TestGroupNormOp):
    def init_test_case(self):
        self.attrs['groups'] = 4


class TestGroupNormFP16Op2(TestGroupNormFP16OP):
    def init_test_case(self):
        self.attrs['groups'] = 4
        self.dtype = np.float16


class TestGroupNormBF16Op2(TestGroupNormBF16Op):
    def init_test_case(self):
        self.attrs['groups'] = 4


class TestGroupNormOpBigEps1(TestGroupNormOp):
    def init_test_case(self):
        self.attrs['groups'] = 1
        self.attrs['epsilon'] = 0.5


class TestGroupNormOpBigEps2(TestGroupNormOp):
    def init_test_case(self):
        self.attrs['groups'] = 4
        self.attrs['epsilon'] = 0.5


class TestGroupNormOpBigEps3(TestGroupNormOp):
    def init_test_case(self):
        self.attrs['epsilon'] = 0.5


@skip_check_grad_ci(
    reason='''This test case is used to ensure whether the gradient checking results between CPU and GPU
            are consistent when using the same inputs, thus, it doesn't need to call check_grad.'''
)
class TestGroupNormOpLargeData(TestGroupNormOp):
    def init_test_case(self):
        self.shape = (2, 32, 64, 64)
        self.attrs['groups'] = 8
        self.compare_between_place = True


class TestGroupNormOp1_With_NHWC(TestGroupNormOp):
    def init_test_case(self):
        self.attrs['groups'] = 1
        self.data_format = "NHWC"


class TestGroupNormOp2_With_NHWC(TestGroupNormOp):
    def init_test_case(self):
        self.attrs['groups'] = 4
        self.data_format = "NHWC"


class TestGroupNormFP16Op_With_NHWC(TestGroupNormFP16OP):
    def init_test_case(self):
        self.no_need_check_inplace = True
        self.attrs['groups'] = 1
        self.data_format = "NHWC"
        self.attrs['epsilon'] = 0.5
        self.shape = (1, 100, 4, 4)
        self.dtype = np.float16

    def test_check_output(self):
        rtol = 2e-3
        atol = 2e-3
        inplace_atol = 2e-3
        place = core.CUDAPlace(0)
        self.check_output_with_place(
            place, rtol=rtol, atol=atol, inplace_atol=inplace_atol
        )


class TestGroupNormBF16Op_With_NHWC(TestGroupNormBF16Op):
    def setUp(self):
        self.op_type = "group_norm"
        self.python_api = group_norm_wrapper
        self.python_out_sig = ["Y"]
        self.data_format = "NHWC"
        self.dtype = np.uint16
        self.shape = (1, 3, 5, 100)
        self.attrs = {
            'epsilon': 5e-2,
            'groups': 2,
            'data_layout': self.data_format,
        }
        self.compare_between_place = False
        self.init_test_case()
        input = (
            np.sin(
                np.arange(
                    self.shape[0]
                    * self.shape[1]
                    * self.shape[2]
                    * self.shape[3]
                )
            )
            .reshape(self.shape)
            .astype(np.float32)
        )
        scale = np.sin(np.arange(self.shape[3])).astype(np.float32)
        bias = np.sin(np.arange(self.shape[3])).astype(np.float32)
        output, mean, var = group_norm_naive(
            input,
            scale,
            bias,
            self.attrs['epsilon'],
            self.attrs['groups'],
            self.data_format,
        )

        self.inputs = {
            'X': convert_float_to_uint16(input),
            'Scale': convert_float_to_uint16(scale),
            'Bias': convert_float_to_uint16(bias),
        }
        self.outputs = {'Y': output, 'Mean': mean, 'Variance': var}

    def test_check_output(self):
        rtol = 2e-2
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, rtol=rtol)


class TestGroupNormOpBigEps1_With_NHWC(TestGroupNormOp):
    def init_test_case(self):
        self.attrs['groups'] = 1
        self.attrs['epsilon'] = 0.5
        self.data_format = "NHWC"


class TestGroupNormOpBigEps2_With_NHWC(TestGroupNormOp):
    def init_test_case(self):
        self.attrs['groups'] = 4
        self.attrs['epsilon'] = 0.5
        self.data_format = "NHWC"


class TestGroupNormOpBigEps3_With_NHWC(TestGroupNormOp):
    def init_test_case(self):
        self.attrs['epsilon'] = 0.5
        self.data_format = "NHWC"


@skip_check_grad_ci(
    reason='''This test case is used to ensure whether the gradient checking results between CPU and GPU
            are consistent when using the same inputs, thus, it doesn't need to call check_grad.'''
)
class TestGroupNormOpLargeData_With_NHWC(TestGroupNormOp):
    def init_test_case(self):
        self.shape = (2, 64, 32, 32)  # NCHW
        self.attrs['groups'] = 8
        self.data_format = "NHWC"
        self.compare_between_place = True


class TestGroupNormAPI_With_NHWC(unittest.TestCase):
    def test_case1(self):
        with paddle_static_guard():
            data1 = paddle.static.data(
                name='data1', shape=[None, 3, 3, 4], dtype='float64'
            )
            out1 = paddle.static.nn.group_norm(
                input=data1, groups=2, data_layout="NHWC"
            )
            data2 = paddle.static.data(
                name='data2', shape=[None, 4, 3, 3], dtype='float64'
            )
            out2 = paddle.static.nn.group_norm(
                input=data2, groups=2, data_layout="NCHW"
            )

            data1_np = np.random.random((2, 3, 3, 4)).astype("float64")
            data2_np = np.random.random((2, 4, 3, 3)).astype("float64")
            scale = np.array([1]).astype("float64")
            bias = np.array([0]).astype("float64")

            place = core.CPUPlace()
            exe = base.Executor(place)
            results = exe.run(
                base.default_main_program(),
                feed={"data1": data1_np, "data2": data2_np},
                fetch_list=[out1, out2],
                return_numpy=True,
            )
            expect_res1 = group_norm_naive(
                data1_np,
                scale,
                bias,
                epsilon=1e-5,
                groups=2,
                data_layout="NHWC",
            )
            expect_res2 = group_norm_naive(
                data2_np,
                scale,
                bias,
                epsilon=1e-5,
                groups=2,
                data_layout="NCHW",
            )
            np.testing.assert_allclose(results[0], expect_res1[0], rtol=1e-05)
            np.testing.assert_allclose(results[1], expect_res2[0], rtol=1e-05)


class TestGroupNormException(unittest.TestCase):
    # data_layout is not NHWC or NCHW
    def test_exception(self):
        with paddle_static_guard():
            data = paddle.static.data(
                name='data', shape=[None, 3, 3, 4], dtype="float64"
            )

            def attr_data_format():
                out = paddle.static.nn.group_norm(
                    input=data, groups=2, data_layout="NDHW"
                )

            self.assertRaises(ValueError, attr_data_format)


class TestGroupNormEager(unittest.TestCase):
    def test_dygraph_api(self):
        # not supported float64
        # only support float32
        self.dtype = np.float32

        self.shape = (8, 32, 32)
        input = np.random.random(self.shape).astype(self.dtype)

        with base.dygraph.guard():
            tensor_1 = base.dygraph.to_variable(input)
            tensor_1.stop_gradient = False
            groupNorm = paddle.nn.GroupNorm(num_channels=32, num_groups=4)
            ret1 = groupNorm(tensor_1)
            ret1.backward()
            tensor_eager_1 = base.dygraph.to_variable(input)
            tensor_eager_1.stop_gradient = False
            groupNorm_eager = paddle.nn.GroupNorm(num_channels=32, num_groups=4)
            ret2 = groupNorm_eager(tensor_eager_1)
            ret2.backward()
            self.assertEqual(
                (tensor_1.grad.numpy() == tensor_eager_1.grad.numpy()).all(),
                True,
            )

        self.dtype = np.float32
        self.shape = (8, 32, 32)
        input = np.random.random(self.shape).astype(self.dtype)

        with base.dygraph.guard():
            tensor_1 = base.dygraph.to_variable(input)
            tensor_1.stop_gradient = False
            groupNorm = paddle.nn.GroupNorm(num_channels=32, num_groups=4)
            ret1 = groupNorm(tensor_1)
            ret1.backward()
            tensor_eager_1 = base.dygraph.to_variable(input)
            tensor_eager_1.stop_gradient = False
            groupNorm_eager = paddle.nn.GroupNorm(num_channels=32, num_groups=4)
            ret2 = groupNorm_eager(tensor_eager_1)
            ret2.backward()
            self.assertEqual(
                (tensor_1.grad.numpy() == tensor_eager_1.grad.numpy()).all(),
                True,
            )


class TestGroupNormEager_fp16(unittest.TestCase):
    def test_dygraph_api(self):
        # not supported float16
        # only support float32
        self.dtype = np.float32

        self.shape = (8, 32, 32)
        input = np.random.random(self.shape).astype(self.dtype)

        with base.dygraph.guard():
            tensor_1 = base.dygraph.to_variable(input)
            tensor_1.stop_gradient = False
            groupNorm = paddle.nn.GroupNorm(num_channels=32, num_groups=4)
            ret1 = groupNorm(tensor_1)
            ret1.backward()
            tensor_eager_1 = base.dygraph.to_variable(input)
            tensor_eager_1.stop_gradient = False
            groupNorm_eager = paddle.nn.GroupNorm(num_channels=32, num_groups=4)
            ret2 = groupNorm_eager(tensor_eager_1)
            ret2.backward()
            self.assertEqual(
                (tensor_1.grad.numpy() == tensor_eager_1.grad.numpy()).all(),
                True,
            )


places = [paddle.CPUPlace()]
if paddle.is_compiled_with_cuda():
    places.append(paddle.CUDAPlace(0))


class PrimNet(paddle.nn.Layer):
    def __init__(
        self,
        num_groups,
        num_channels,
        scale,
        bias,
        epsilon=1e-05,
        data_format='NCHW',
        name=None,
    ):
        super().__init__()
        self.func = paddle.nn.GroupNorm(
            num_groups, num_channels, epsilon, False, False, data_format, name
        )
        paddle.assign(scale, self.func.weight)
        paddle.assign(bias, self.func.bias)

    def forward(self, x):
        out = self.func(x)
        return out


def apply_to_static(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(net, build_strategy=build_strategy)


# The original GroupNorm cannot support NHWC format
@param.parameterized_class(
    (
        'name',
        'shape',
        'epsilon',
        'groups',
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
            2,
            'NCHW',
            places,
            'float32',
            [
                [5e-5, 5e-5, 5e-5],  # cpu thresholds for static, jit, jit_cinn
                [1e-5, 1e-5, 1e-5],
            ],  # gpu thresholds for static, jit, jit_cinn
            None,
        ),
        (
            'test1',
            (2, 100, 3, 5),
            1e-5,
            1,
            'NCHW',
            places,
            'float32',
            [
                [5e-5, 5e-5, 5e-5],  # cpu thresholds for static, jit, jit_cinn
                [1e-5, 1e-5, 1e-5],
            ],  # gpu thresholds for static, jit, jit_cinn
            None,
        ),
        (
            'test2',
            (2, 100, 3, 5),
            1e-5,
            4,
            'NCHW',
            places,
            'float32',
            [
                [5e-5, 5e-5, 5e-5],  # cpu thresholds for static, jit, jit_cinn
                [1e-5, 1e-5, 1e-5],
            ],  # gpu thresholds for static, jit, jit_cinn
            None,
        ),
        (
            'bigeps1',
            (2, 100, 3, 5),
            0.5,
            1,
            'NCHW',
            places,
            'float32',
            [
                [5e-5, 5e-5, 5e-5],  # cpu thresholds for static, jit, jit_cinn
                [1e-5, 1e-5, 1e-5],
            ],  # gpu thresholds for static, jit, jit_cinn
            None,
        ),
        (
            'bigeps2',
            (2, 100, 3, 5),
            0.5,
            4,
            'NCHW',
            places,
            'float32',
            [
                [5e-5, 5e-5, 5e-5],  # cpu thresholds for static, jit, jit_cinn
                [1e-5, 1e-5, 1e-5],
            ],  # gpu thresholds for static, jit, jit_cinn
            None,
        ),
        (
            'bigeps3',
            (2, 100, 3, 5),
            0.5,
            2,
            'NCHW',
            places,
            'float32',
            [
                [5e-5, 5e-5, 5e-5],  # cpu thresholds for static, jit, jit_cinn
                [1e-5, 1e-5, 1e-5],
            ],  # gpu thresholds for static, jit, jit_cinn
            None,
        ),
        (
            'largedata',
            (2, 32, 64, 64),
            1e-5,
            4,
            'NCHW',
            places,
            'float32',
            [
                [5e-5, 5e-5, 5e-5],  # cpu thresholds for static, jit, jit_cinn
                [1e-5, 1e-5, 1e-5],
            ],  # gpu thresholds for static, jit, jit_cinn
            [
                5e-2,
                5e-3,
            ],  # threshold for cpu x_grad (5e-2), cpu scale_grad (5e-2) and gpu scale_grad (5e-3)
        ),
        (
            'test0_fp64',
            (2, 100, 3, 5),
            1e-5,
            2,
            'NCHW',
            places,
            'float64',
            [
                [
                    5e-14,
                    5e-14,
                    5e-14,
                ],  # cpu thresholds for static, jit, jit_cinn
                [1e-14, 1e-14, 1e-14],
            ],  # gpu thresholds for static, jit, jit_cinn
            [
                5e-14,
                2e-14,
            ],  # threshold for cpu x_grad, cpu scale_grad and gpu scale_grad
        ),
        (
            'test1_fp64',
            (2, 100, 3, 5),
            1e-5,
            1,
            'NCHW',
            places,
            'float64',
            [
                [
                    5e-14,
                    5e-14,
                    5e-14,
                ],  # cpu thresholds for static, jit, jit_cinn
                [1e-14, 1e-14, 1e-14],
            ],  # gpu thresholds for static, jit, jit_cinn
            [
                5e-14,
                2e-14,
            ],  # threshold for cpu x_grad, cpu scale_grad and gpu scale_grad
        ),
        (
            'test2_fp64',
            (2, 100, 3, 5),
            1e-5,
            4,
            'NCHW',
            places,
            'float64',
            [
                [
                    5e-14,
                    5e-14,
                    5e-14,
                ],  # cpu thresholds for static, jit, jit_cinn
                [1e-14, 1e-14, 1e-14],
            ],  # gpu thresholds for static, jit, jit_cinn
            [5e-14, 2e-14],  # threshold for scale_grad on cpu and gpu
        ),
        (
            'bigeps1_fp64',
            (2, 100, 3, 5),
            0.5,
            1,
            'NCHW',
            places,
            'float64',
            [
                [
                    5e-14,
                    5e-14,
                    5e-14,
                ],  # cpu thresholds for static, jit, jit_cinn
                [1e-14, 1e-14, 1e-14],
            ],  # gpu thresholds for static, jit, jit_cinn
            [5e-14, 2e-14],  # threshold for scale_grad on cpu and gpu
        ),
        (
            'bigeps2_fp64',
            (2, 100, 3, 5),
            0.5,
            4,
            'NCHW',
            places,
            'float64',
            [
                [
                    5e-14,
                    5e-14,
                    5e-14,
                ],  # cpu thresholds for static, jit, jit_cinn
                [1e-14, 1e-14, 1e-14],
            ],  # gpu thresholds for static, jit, jit_cinn
            [5e-14, 2e-14],  # threshold for scale_grad on cpu and gpu
        ),
        (
            'bigeps3_fp64',
            (2, 100, 3, 5),
            0.5,
            2,
            'NCHW',
            places,
            'float64',
            [
                [
                    5e-14,
                    5e-14,
                    5e-14,
                ],  # cpu thresholds for static, jit, jit_cinn
                [1e-14, 1e-14, 1e-14],
            ],  # gpu thresholds for static, jit, jit_cinn
            [5e-14, 2e-14],  # threshold for scale_grad on cpu and gpu
        ),
        (
            'largedata_fp64',
            (2, 32, 64, 64),
            1e-5,
            4,
            'NCHW',
            places,
            'float64',
            [
                [
                    5e-14,
                    5e-14,
                    5e-14,
                ],  # cpu thresholds for static, jit, jit_cinn
                [1e-14, 1e-14, 1e-14],
            ],  # gpu thresholds for static, jit, jit_cinn
            [5e-11, 5e-12],  # threshold for scale_grad on cpu and gpu
        ),
        (
            'test0_fp16',
            (2, 100, 3, 5),
            1e-5,
            2,
            'NCHW',
            places,
            'float16',
            [[1e-3, 1e-3, 1e-3]],  # gpu thresholds for static, jit, jit_cinn
            None,
        ),
        (
            'test0_bfp16',
            (2, 100, 3, 5),
            1e-5,
            2,
            'NCHW',
            places,
            'bfloat16',
            [
                [
                    1e-2,
                    1e-2,
                    1e-2,
                ],  # cpu thresholds for static, jit, jit_cinn
                [1e-2, 1e-2, 1e-2],
            ],  # gpu thresholds for static, jit, jit_cinn
            None,
        ),
        (
            'test1_bfp16',
            (2, 100, 3, 5),
            1e-5,
            1,
            'NCHW',
            places,
            'bfloat16',
            [
                [
                    1e-2,
                    1e-2,
                    1e-2,
                ],  # cpu thresholds for static, jit, jit_cinn
                [1e-2, 1e-2, 1e-2],
            ],  # gpu thresholds for static, jit, jit_cinn
            None,
        ),
        (
            'test2_bfp16',
            (2, 100, 3, 5),
            1e-5,
            4,
            'NCHW',
            places,
            'bfloat16',
            [
                [
                    1e-2,
                    1e-2,
                    1e-2,
                ],  # cpu thresholds for static, jit, jit_cinn
                [1e-2, 1e-2, 1e-2],
            ],  # gpu thresholds for static, jit, jit_cinn
            None,
        ),
        (
            'bigeps3_bfp16',
            (2, 100, 3, 5),
            0.5,
            2,
            'NCHW',
            places,
            'bfloat16',
            [
                [
                    1e-2,
                    1e-2,
                    1e-2,
                ],  # cpu thresholds for static, jit, jit_cinn
                [1e-2, 1e-2, 1e-2],
            ],  # gpu thresholds for static, jit, jit_cinn
            None,
        ),
        (
            'largedata_bfp16',
            (2, 32, 64, 64),
            1e-5,
            4,
            'NCHW',
            places,
            'bfloat16',
            [
                [
                    1e-2,
                    1e-2,
                    1e-2,
                ],  # cpu thresholds for static, jit, jit_cinn
                [1e-2, 1e-2, 1e-2],
            ],  # gpu thresholds for static, jit, jit_cinn
            None,
        ),
    ),
)
class TestCompositeGroupNorm(unittest.TestCase):
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
        if self.dtype != "bfloat16":
            self.x = np.random.random(self.shape).astype(self.dtype)
            self.scale = np.random.random([self.shape[1]]).astype(self.dtype)
            self.bias = np.random.random([self.shape[1]]).astype(self.dtype)
        else:
            self.x = convert_float_to_uint16(
                np.random.random(self.shape).astype("float32")
            )
            self.scale = convert_float_to_uint16(
                np.random.random([self.shape[1]]).astype("float32")
            )
            self.bias = convert_float_to_uint16(
                np.random.random([self.shape[1]]).astype("float32")
            )
        self.num_channels = self.shape[1]

        if self.dtype in ['float16', 'bfloat16']:
            self.places = []
            if paddle.is_compiled_with_cuda():
                self.places.append(paddle.CUDAPlace(0))

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
        group_norm = paddle.nn.GroupNorm(
            self.groups,
            self.num_channels,
            self.epsilon,
            False,
            False,
            self.data_format,
        )
        paddle.assign(scale_, group_norm.weight)
        paddle.assign(bias_, group_norm.bias)
        output = group_norm(input_)
        grad = paddle.grad(output, input_)
        if self.dtype == "bfloat16":
            output = paddle.cast(output, "float32")
            grad = paddle.utils.map_structure(
                lambda x: paddle.cast(x, "float32"), grad
            )
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
                'scale_', shape=self.scale.shape, dtype=self.bias.dtype
            )
            scale_.stop_gradient = False

            bias_ = paddle.static.data(
                'bias_', shape=self.bias.shape, dtype=self.x.dtype
            )
            bias_.stop_gradient = False

            group_norm = paddle.nn.GroupNorm(
                self.groups,
                self.num_channels,
                self.epsilon,
                False,
                False,
                self.data_format,
            )
            group_norm.weight.stop_gradient = False
            group_norm.bias.stop_gradient = False

            paddle.assign(scale_, group_norm.weight)
            paddle.assign(bias_, group_norm.bias)
            output = group_norm(input_)

            blocks = mp.blocks
            names = dict(
                zip(
                    blocks[0].ops[2].output_names,
                    blocks[0].ops[2].output_arg_names,
                )
            )
            vars_list = [
                names[key]
                for key in [
                    "Y",
                    "Mean",
                    "Variance",
                ]
            ]

            fwd_ops = [op.type for op in blocks[0].ops]
            # Ensure that group_norm in original block
            assert 'group_norm' in fwd_ops

            if core._is_fwd_prim_enabled():
                paddle.incubate.autograd.primapi.to_prim(mp.blocks)
                fwd_ops_new = [op.type for op in blocks[0].ops]
                # Ensure that group_norm is splitted into small ops
                assert 'group_norm' not in fwd_ops_new

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
        if self.dtype == "bfloat16":
            out_list[0] = convert_uint16_to_float(out_list[0])
            i = 3
            for i in range(3, len(out_list)):
                out_list[i] = convert_uint16_to_float(out_list[i])
        return out_list[:3], out_list[3:]

    def test_static_comp(self):
        paddle.enable_static()
        mps = []
        fwd_actual = []
        rev_actual = []
        if len(self.places) < 1:
            return

        with paddle.base.framework._static_guard():
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
                        'scale_', shape=self.scale.shape, dtype=self.bias.dtype
                    )
                    scale_.stop_gradient = False

                    bias_ = paddle.static.data(
                        'bias_', shape=self.bias.shape, dtype=self.x.dtype
                    )
                    bias_.stop_gradient = False

                    group_norm = paddle.nn.GroupNorm(
                        self.groups,
                        self.num_channels,
                        self.epsilon,
                        False,
                        False,
                        self.data_format,
                    )
                    group_norm.weight.stop_gradient = False
                    group_norm.bias.stop_gradient = False

                    paddle.assign(scale_, group_norm.weight)
                    paddle.assign(bias_, group_norm.bias)
                    output = group_norm(input_)

                    blocks = mp.blocks
                    names = dict(
                        zip(
                            blocks[0].ops[2].output_names,
                            blocks[0].ops[2].output_arg_names,
                        )
                    )
                    vars_list = [
                        names[key]
                        for key in [
                            "Y",
                            "Mean",
                            "Variance",
                        ]
                    ]

                    fwd_ops = [op.type for op in blocks[0].ops]
                    # Ensure that group_norm in original block
                    assert 'group_norm' in fwd_ops

                    if core._is_fwd_prim_enabled():
                        paddle.incubate.autograd.primapi.to_prim(mp.blocks)
                        fwd_ops_new = [op.type for op in blocks[0].ops]
                        # Ensure that group_norm is splitted into small ops
                        assert 'group_norm' not in fwd_ops_new

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
                if self.dtype == "bfloat16":
                    out_list[0] = convert_uint16_to_float(out_list[0])
                    i = 3
                    for i in range(3, len(out_list)):
                        out_list[i] = convert_uint16_to_float(out_list[i])
                fwd_actual[-1].append(out_list[0])
                fwd_actual[-1].append(out_list[1])
                fwd_actual[-1].append(out_list[2])
                rev_actual[-1].append(out_list[3])
                rev_actual[-1].append(out_list[4])
                rev_actual[-1].append(out_list[5])
                mps.append(mp)

        vars_name = [
            "Y",
            "Mean",
            "Variance",
            "X_grad",
            "Scale_grad",
            "Bias_grad",
        ]

        for i in range(len(self.places)):
            self.assertTrue(
                'group_norm' not in [op.type for op in mps[i].block(0).ops]
            )
            atol = self.threshold_list[i][0]
            rtol = self.threshold_list[i][0]
            for j in range(len(self.static_fwd_desire[i])):
                # in float16 type, Y is float16, mean and var are float32
                # so check mean and var with float32 gpu threshold
                if self.dtype == "float16" and j > 0:
                    atol = 1e-5
                    rtol = 1e-5
                elif self.dtype == "bfloat16" and j > 0:
                    atol = 5e-3
                    rtol = 5e-3
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
            # compare with eager_desire
            np.testing.assert_allclose(
                self.fwd_desire[i],
                fwd_actual[i][0],
                rtol=rtol,
                atol=atol,
                err_msg=f"Check diff failed with fwd_eager:{self.places[i]}",
            )

            for j in range(len(self.static_rev_desire[i])):
                # TODO: fix the diff between cpu and gpu grad is large in original op
                # now use larger threshold when testing cpu grads to bypass cpu grad test
                if self.special_threshold is not None and j <= 1:
                    atol = self.special_threshold[i]
                    rtol = self.special_threshold[i]
                else:
                    atol = self.threshold_list[i][0]
                    rtol = self.threshold_list[i][0]

                max_abs_diff = np.max(
                    np.abs(self.static_rev_desire[i][j] - rev_actual[i][j])
                )

                np.testing.assert_allclose(
                    self.static_rev_desire[i][j],
                    rev_actual[i][j],
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"Check diff failed of place:{self.places[i]}, output: {vars_name[j + 3]}",
                )

            # TODO: fix the diff between cpu and gpu grad is large in original op
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
            net = PrimNet(
                self.groups,
                self.num_channels,
                scale_,
                bias_,
                self.epsilon,
                self.data_format,
            )
            net = apply_to_static(net, False)
            output = net(input_)
            grad = paddle.grad(output, input_)
            fwd_actual.append(
                convert_uint16_to_float(output.numpy())
                if self.dtype == "bfloat16"
                else output.numpy()
            )
            rev_actual.append(
                convert_uint16_to_float(grad[0].numpy())
                if self.dtype == "bfloat16"
                else grad[0].numpy()
            )

        for i in range(len(self.places)):
            atol = self.threshold_list[i][1]
            rtol = self.threshold_list[i][1]
            np.testing.assert_allclose(
                self.fwd_desire[i],
                fwd_actual[i],
                rtol=rtol,
                atol=atol,
                err_msg='%s jit fwd' % self.places[i],
            )

            # TODO: fix the diff between cpu and gpu grad is large in original op
            # now use larger threshold when testing cpu grads to bypass cpu grad test
            if self.special_threshold is not None:
                atol = self.special_threshold[i]
                rtol = self.special_threshold[i]

            np.testing.assert_allclose(
                self.rev_desire[i],
                rev_actual[i],
                rtol=rtol,
                atol=atol,
                err_msg='%s jit rev' % self.places[i],
            )

    def test_jit_comp_with_cinn(self):
        fwd_actual = []
        rev_actual = []
        for place in self.places:
            if not isinstance(place, base.CUDAPlace):
                continue
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
            net = PrimNet(
                self.groups,
                self.num_channels,
                scale_,
                bias_,
                self.epsilon,
                self.data_format,
            )
            # failed in cinn test
            net = apply_to_static(net, True)
            output = net(input_)
            grad = paddle.grad(output, input_)
            fwd_actual.append(
                convert_uint16_to_float(output.numpy())
                if self.dtype == "bfloat16"
                else output.numpy()
            )
            rev_actual.append(
                convert_uint16_to_float(grad[0].numpy())
                if self.dtype == "bfloat16"
                else grad[0].numpy()
            )

        i = 0
        for place in self.places:
            if not isinstance(place, base.CUDAPlace):
                continue
            atol = self.threshold_list[i][2]
            rtol = self.threshold_list[i][2]
            np.testing.assert_allclose(
                self.fwd_desire[i],
                fwd_actual[i],
                rtol=rtol,  # mean of uniform distribution, scale for avoid random failed
                atol=atol,
                err_msg='%s jit_cinn fwd' % self.places[i],
            )
            # TODO: fix the diff between cpu and gpu grad is large in original op
            # now use larger threshold when testing cpu grads to bypass cpu grad test
            if self.special_threshold is not None:
                atol = self.special_threshold[i]
                rtol = self.special_threshold[i]
            np.testing.assert_allclose(
                self.rev_desire[i],
                rev_actual[i],
                rtol=rtol,  # mean of uniform distribution, scale for avoid random failed
                atol=atol,
                err_msg='%s jit_cinn rev' % self.places[i],
            )
            i += 1


if __name__ == '__main__':
    unittest.main()
