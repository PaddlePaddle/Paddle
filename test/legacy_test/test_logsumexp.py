#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16, get_device_place

import paddle
from paddle.base import core


def ref_logsumexp(x, axis=None, keepdim=False, reduce_all=False):
    if isinstance(axis, int):
        axis = (axis,)
    elif isinstance(axis, list):
        axis = tuple(axis)
    if reduce_all:
        axis = None
    out = np.log(np.exp(x).sum(axis=axis, keepdims=keepdim))
    return out


def logsumexp_wrapper(x, axis=None, keepdim=False, allreduce=False):
    if allreduce:
        return paddle.logsumexp(x, None, keepdim)
    return paddle.logsumexp(x, axis, keepdim)


def logsumexp_op_grad(x, axis=None, keepdim=False, reduce_all=False):
    paddle.disable_static()
    tensor_x = paddle.to_tensor(x)
    tensor_x.stop_gradient = False
    out = logsumexp_wrapper(tensor_x, axis, keepdim, reduce_all)
    grad = paddle.grad(out, [tensor_x])
    x_grad = grad[0].numpy()
    paddle.enable_static()
    return x_grad


def logsumexp_ref_grad(x):
    sum = np.exp(x).sum()
    return np.exp(x) / sum


class TestLogsumexp(OpTest):
    def setUp(self):
        self.op_type = 'logsumexp'
        self.prim_op_type = "prim"
        self.python_api = logsumexp_wrapper
        self.public_python_api = logsumexp_wrapper
        self.shape = [2, 3, 4, 5]
        self.dtype = 'float64'
        self.axis = [-1]
        self.keepdim = False
        self.reduce_all = False
        self.set_attrs()

        np.random.seed(10)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = ref_logsumexp(x, self.axis, self.keepdim, self.reduce_all)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {
            'axis': self.axis,
            'keepdim': self.keepdim,
            'reduce_all': self.reduce_all,
        }
        self.user_defined_grads = None
        self.user_defined_grad_outputs = None
        self.set_attrs_addition()

    def set_attrs(self):
        pass

    def set_attrs_addition(self):
        pass

    def test_check_output(self):
        self.check_output(
            check_pir=True,
            check_prim_pir=True,
        )

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            ['Out'],
            user_defined_grads=self.user_defined_grads,
            user_defined_grad_outputs=self.user_defined_grad_outputs,
            check_pir=True,
            check_prim_pir=True,
        )

    def calc_grad(self):
        dy = np.ones(1, dtype=self.dtype)
        x = self.inputs['X']
        y = self.outputs['Out']
        return dy * np.exp(x - y)


class TestLogsumexp_ZeroDim(TestLogsumexp):
    def set_attrs(self):
        self.shape = []
        self.axis = []


class TestLogsumexp_shape(TestLogsumexp):
    def set_attrs(self):
        self.shape = [4, 5, 6]


class TestLogsumexp_axis(TestLogsumexp):
    def set_attrs(self):
        self.axis = [0, -1]


class TestLogsumexp_axis_all(TestLogsumexp):
    def set_attrs(self):
        self.axis = [0, 1, 2, 3]

    def set_attrs_addition(self):
        if paddle.base.core.is_compiled_with_rocm():
            self.user_defined_grads = [self.calc_grad()]
            self.user_defined_grad_outputs = [np.ones(1, dtype=self.dtype)]


class TestLogsumexp_keepdim(TestLogsumexp):
    def set_attrs(self):
        self.keepdim = True


class TestLogsumexp_reduce_all(TestLogsumexp):
    def set_attrs(self):
        self.reduce_all = True

    def set_attrs_addition(self):
        if paddle.base.core.is_compiled_with_rocm():
            self.user_defined_grads = [self.calc_grad()]
            self.user_defined_grad_outputs = [np.ones(1, dtype=self.dtype)]


class TestLogsumexp_FP32(TestLogsumexp):
    def set_attrs(self):
        self.dtype = 'float32'

    def test_check_grad(self):
        self.__class__.dtype = self.dtype
        x_grad = logsumexp_op_grad(self.inputs['X'])
        ref_x_grad = logsumexp_ref_grad(self.inputs['X'])
        np.testing.assert_allclose(x_grad, ref_x_grad, rtol=1e-08, atol=1e-08)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestLogsumexp_FP16(TestLogsumexp):
    def set_attrs(self):
        self.dtype = 'float16'

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(
            place,
            check_pir=True,
            check_prim_pir=True,
        )

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            check_pir=True,
            check_prim_pir=True,
        )

    def set_attrs_addition(self):
        pass


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestLogsumexpBF16Op(TestLogsumexp):
    def setUp(self):
        self.op_type = 'logsumexp'
        self.prim_op_type = "prim"
        self.python_api = logsumexp_wrapper
        self.public_python_api = logsumexp_wrapper
        self.dtype = np.uint16
        self.shape = [2, 3, 4, 5]
        self.axis = [-1]
        self.keepdim = False
        self.reduce_all = False
        self.set_attrs()
        x = np.random.uniform(-1, 1, self.shape).astype(np.float64)
        out = ref_logsumexp(x, self.axis, self.keepdim, self.reduce_all)
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.outputs = {'Out': convert_float_to_uint16(out)}
        self.attrs = {
            'axis': self.axis,
            'keepdim': self.keepdim,
            'reduce_all': self.reduce_all,
        }
        self.set_attrs_addition()

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(
            place,
            check_pir=True,
            check_prim_pir=True,
        )

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            check_pir=True,
            check_prim_pir=True,
        )

    def set_attrs(self):
        pass

    def set_attrs_addition(self):
        pass


class TestLogsumexpError(unittest.TestCase):
    def test_errors(self):
        with paddle.static.program_guard(paddle.static.Program()):
            self.assertRaises(TypeError, paddle.logsumexp, 1)
            x1 = paddle.static.data(name='x1', shape=[120], dtype="bool")
            self.assertRaises(TypeError, paddle.logsumexp, x1)


class TestLogsumexpAPI(unittest.TestCase):
    def setUp(self):
        self.shape = [2, 3, 4, 5]
        self.x = np.random.uniform(-1, 1, self.shape).astype(np.float32)
        self.place = get_device_place()

    def api_case(self, axis=None, keepdim=False):
        out_ref = ref_logsumexp(self.x, axis, keepdim)
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.shape)
            out = paddle.logsumexp(x, axis, keepdim)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x}, fetch_list=[out])
        np.testing.assert_allclose(res[0], out_ref, rtol=1e-05)

        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        out = paddle.logsumexp(x, axis, keepdim)
        np.testing.assert_allclose(out.numpy(), out_ref, rtol=1e-05)
        paddle.enable_static()

    def test_api(self):
        self.api_case()
        self.api_case(2)
        self.api_case([-1])
        self.api_case([2, -3])
        self.api_case((0, 1, -1))
        self.api_case(keepdim=True)

    def test_alias(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        out1 = paddle.logsumexp(x)
        out2 = paddle.tensor.logsumexp(x)
        out3 = paddle.tensor.math.logsumexp(x)
        out_ref = ref_logsumexp(self.x)
        for out in [out1, out2, out3]:
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=1e-05)
        paddle.enable_static()


class TestLogsumexp_ZeroSize(OpTest):
    def setUp(self):
        self.op_type = 'logsumexp'
        self.python_api = logsumexp_wrapper
        self.public_python_api = logsumexp_wrapper
        self.dtype = 'float64'
        self.shape = [2, 3, 0]
        self.axis = [-1]  # out return shape [2, 3], value -inf
        self.keepdim = False
        self.reduce_all = False
        self.set_attrs()

        np.random.seed(10)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = ref_logsumexp(x, self.axis, self.keepdim, self.reduce_all)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {
            'axis': self.axis,
            'keepdim': self.keepdim,
            'reduce_all': self.reduce_all,
        }

    def set_attrs(self):
        pass

    def test_check_output(self):
        self.check_output(
            check_pir=True,
        )

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            ['Out'],
            check_pir=True,
        )


class TestLogsumexp_ZeroSize2(TestLogsumexp_ZeroSize):
    def set_attrs(self):
        self.shape = [2, 3, 0]
        self.axis = [1]  # out return shape [2, 0]


class TestLogsumexpOutAndParamDecorator(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x_shape = [2, 3, 4]
        self.axis = 1
        self.x_np = np.random.rand(*self.x_shape).astype(np.float32)

        self.apis = [
            paddle.logsumexp,
            paddle.special.logsumexp,
        ]
        self.test_types = [
            # "decorator1",
            # "decorator2",
            "out",
            # "out_decorator",
        ]

    def do_test(self, api, test_type):
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        out = paddle.empty((2, 3), dtype='float32')
        out.stop_gradient = False

        if test_type == 'raw':
            result = api(x, axis=self.axis)
            result.mean().backward()
            return result, x.grad
        elif test_type == 'decorator1':
            result = api(x, axis=self.axis)
            result.mean().backward()
            return result, x.grad
        elif test_type == 'decorator2':
            result = api(input=x, axis=self.axis)
            result.mean().backward()
            return result, x.grad
        elif test_type == 'out':
            api(x, axis=self.axis, out=out)
            out.mean().backward()
            return out, x.grad
        elif test_type == 'out_decorator':
            api(input=x, axis=self.axis, out=out)
            out.mean().backward()
            return out, x.grad
        else:
            raise ValueError(f"Unknown test type: {test_type}")

    def test_logsumexp_out(self):
        out_std, grad_std = self.do_test(paddle.logsumexp, 'raw')
        for test_type in self.test_types:
            out, grad = self.do_test(paddle.logsumexp, test_type)
            np.testing.assert_allclose(out.numpy(), out_std.numpy(), rtol=1e-20)
            np.testing.assert_allclose(
                grad.numpy(), grad_std.numpy(), rtol=1e-20
            )


class TestLogsumexpAPI_Compatibility(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.shape = [5, 6]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_input = np.random.randint(0, 8, self.shape).astype(self.dtype)
        self.np_ref_out = ref_logsumexp(
            self.np_input, axis=[0, 1], keepdim=True, reduce_all=True
        )

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_input)
        paddle_dygraph_out = []
        # Position args (args)
        out1 = paddle.logsumexp(x, [0, 1], True)
        paddle_dygraph_out.append(out1)
        # Key words args (kwargs) for paddle
        out2 = paddle.logsumexp(x=x, axis=[0, 1], keepdim=True)
        paddle_dygraph_out.append(out2)
        # Key words args for torch
        out3 = paddle.logsumexp(input=x, dim=[0, 1], keepdim=True)
        paddle_dygraph_out.append(out3)
        # Combined args and kwargs
        out4 = paddle.logsumexp(x, dim=[0, 1], keepdim=True)
        paddle_dygraph_out.append(out4)
        # Tensor method args
        out5 = x.logsumexp([0, 1], True)
        paddle_dygraph_out.append(out5)
        # Tensor method kwargs
        out6 = x.logsumexp(dim=[0, 1], keepdim=True)
        paddle_dygraph_out.append(out6)
        # Test out
        out7 = paddle.empty([])
        paddle.logsumexp(x, [0, 1], True, out=out7)
        paddle_dygraph_out.append(out7)
        # Numpy reference  out
        ref_out = self.np_ref_out
        # Check
        for out in paddle_dygraph_out:
            np.testing.assert_allclose(ref_out, out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            # Position args (args)
            out1 = paddle.logsumexp(x, [0, 1], True)
            # Key words args (kwargs) for paddle
            out2 = paddle.logsumexp(x=x, axis=[0, 1], keepdim=True)
            # Key words args for torch
            out3 = paddle.logsumexp(input=x, dim=[0, 1], keepdim=True)
            # Combined args and kwargs
            out4 = paddle.logsumexp(x, dim=[0, 1], keepdim=True)
            # Tensor method args
            out5 = x.logsumexp([0, 1], True)
            # Tensor method kwargs
            out6 = x.logsumexp(dim=[0, 1], keepdim=True)
            # Do not support out in static
            # out7 = paddle.empty([])
            exe = paddle.base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={"x": self.np_input},
                fetch_list=[out1, out2, out3, out4, out5, out6],
            )
            ref_out = self.np_ref_out
            for out in fetches:
                np.testing.assert_allclose(out, ref_out)


if __name__ == '__main__':
    unittest.main()
