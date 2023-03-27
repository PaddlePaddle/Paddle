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
from eager_op_test import (
    OpTest,
    convert_float_to_uint16,
    paddle_static_guard,
    skip_check_grad_ci,
)
from testsuite import create_op

import paddle
from paddle import fluid
from paddle.fluid import core


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
            with fluid.program_guard(fluid.Program(), fluid.Program()):

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
            'X': OpTest.np_dtype_to_fluid_dtype(input),
            'Scale': OpTest.np_dtype_to_fluid_dtype(scale),
            'Bias': OpTest.np_dtype_to_fluid_dtype(bias),
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
        inputs_to_check = set(['X', 'Scale', 'Bias'])
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
        self.check_grad_with_place(place, set(['X', 'Scale', 'Bias']), 'Y')
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                set(['X', 'Scale', 'Bias']),
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
        self.check_grad_with_place(place, set(['X', 'Scale', 'Bias']), 'Y')

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
        self.check_grad_with_place(place, set(['X', 'Scale', 'Bias']), 'Y')

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
            exe = fluid.Executor(place)
            results = exe.run(
                fluid.default_main_program(),
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

        with fluid.dygraph.guard():
            tensor_1 = fluid.dygraph.to_variable(input)
            tensor_1.stop_gradient = False
            groupNorm = paddle.nn.GroupNorm(num_channels=32, num_groups=4)
            ret1 = groupNorm(tensor_1)
            ret1.backward()
            tensor_eager_1 = fluid.dygraph.to_variable(input)
            tensor_eager_1.stop_gradient = False
            groupNorm_eager = paddle.nn.GroupNorm(num_channels=32, num_groups=4)
            ret2 = groupNorm_eager(tensor_eager_1)
            ret2.backward()
            self.assertEqual(
                (tensor_1.grad.numpy() == tensor_eager_1.grad.numpy()).all(),
                True,
            )


class TestGroupNormEager_fp32(unittest.TestCase):
    def test_dygraph_api(self):
        self.dtype = np.float32
        self.shape = (8, 32, 32)
        input = np.random.random(self.shape).astype(self.dtype)

        with fluid.dygraph.guard():
            tensor_1 = fluid.dygraph.to_variable(input)
            tensor_1.stop_gradient = False
            groupNorm = paddle.nn.GroupNorm(num_channels=32, num_groups=4)
            ret1 = groupNorm(tensor_1)
            ret1.backward()
            tensor_eager_1 = fluid.dygraph.to_variable(input)
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

        with fluid.dygraph.guard():
            tensor_1 = fluid.dygraph.to_variable(input)
            tensor_1.stop_gradient = False
            groupNorm = paddle.nn.GroupNorm(num_channels=32, num_groups=4)
            ret1 = groupNorm(tensor_1)
            ret1.backward()
            tensor_eager_1 = fluid.dygraph.to_variable(input)
            tensor_eager_1.stop_gradient = False
            groupNorm_eager = paddle.nn.GroupNorm(num_channels=32, num_groups=4)
            ret2 = groupNorm_eager(tensor_eager_1)
            ret2.backward()
            self.assertEqual(
                (tensor_1.grad.numpy() == tensor_eager_1.grad.numpy()).all(),
                True,
            )


if __name__ == '__main__':
    unittest.main()
