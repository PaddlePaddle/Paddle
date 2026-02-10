#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, get_device_place, is_custom_device

import paddle


def ref_std(x, axis=None, unbiased=True, keepdim=False):
    ddof = 1 if unbiased else 0
    if isinstance(axis, int):
        axis = (axis,)
    if axis is not None:
        axis = tuple(axis)
    return np.std(x, axis=axis, ddof=ddof, keepdims=keepdim)


class TestStdAPI(unittest.TestCase):
    def setUp(self):
        self.dtype = 'float64'
        self.shape = [1, 3, 4, 10]
        self.axis = [1, 3]
        self.keepdim = False
        self.unbiased = True
        self.set_attrs()
        self.x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.place = get_device_place()

    def set_attrs(self):
        pass

    def static(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.shape, self.dtype)
            out = paddle.std(x, self.axis, self.unbiased, self.keepdim)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x}, fetch_list=[out])
        return res[0]

    def dygraph(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        out = paddle.std(x, self.axis, self.unbiased, self.keepdim)
        paddle.enable_static()
        return out.numpy()

    def test_api(self):
        out_ref = ref_std(self.x, self.axis, self.unbiased, self.keepdim)
        out_dygraph = self.dygraph()
        out_static = self.static()
        for out in [out_dygraph, out_static]:
            np.testing.assert_allclose(out_ref, out, rtol=1e-05)
            self.assertTrue(np.equal(out_ref.shape, out.shape).all())


class TestStdAPI2(OpTest):
    def setUp(self):
        self.python_api = paddle.std
        self.op_type = "std"
        self.prim_op_type = "prim"
        self.init_dtype_type()
        self.attrs = {
            'axis': self.axis,
            'unbiased': self.unbiased,
            'keepdim': self.keepdim,
        }
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = ref_std(
            x, axis=self.axis, unbiased=self.unbiased, keepdim=self.keepdim
        )
        self.inputs = {'x': x}
        self.outputs = {'out': out}

        def std_wrapper(x):
            return paddle.std(
                x, axis=self.axis, unbiased=self.unbiased, keepdim=self.keepdim
            )

        self.python_api = std_wrapper
        self.public_python_api = std_wrapper

    def init_dtype_type(self):
        self.dtype = 'float64'
        self.shape = [1, 3, 4, 10]
        self.axis = [1, 3]
        self.keepdim = False
        self.unbiased = True

    def test_check_output(self):
        self.check_output_with_place(
            paddle.CPUPlace(),
            check_prim=True,
            check_pir=True,
            check_symbol_infer=True,
            check_prim_pir=True,
        )
        if paddle.is_compiled_with_cuda():
            self.check_output_with_place(
                paddle.CUDAPlace(0),
                check_prim=True,
                check_pir=True,
                check_symbol_infer=True,
                check_prim_pir=True,
            )

    def test_check_grad_normal(self):
        self.check_grad_with_place(
            paddle.CPUPlace(),
            ['x'],
            'out',
            check_prim=False,
            check_pir=True,
            check_prim_pir=False,
        )
        if paddle.core.is_compiled_with_cuda():
            self.check_grad_with_place(
                paddle.CUDAPlace(0),
                ['x'],
                'out',
                check_prim=False,
                check_pir=True,
                check_prim_pir=False,
            )


class TestStdAPI_dtype(TestStdAPI):
    def set_attrs(self):
        self.dtype = 'float32'


class TestStdAPI_axis_int(TestStdAPI):
    def set_attrs(self):
        self.axis = 2


class TestStdAPI_axis_list(TestStdAPI):
    def set_attrs(self):
        self.axis = [1, 2]


class TestStdAPI_axis_tuple(TestStdAPI):
    def set_attrs(self):
        self.axis = (1, 3)


class TestStdAPI_keepdim(TestStdAPI):
    def set_attrs(self):
        self.keepdim = False


class TestStdAPI_unbiased(TestStdAPI):
    def set_attrs(self):
        self.unbiased = False


class TestStdAPI_alias(unittest.TestCase):
    def test_alias(self):
        paddle.disable_static()
        x = paddle.to_tensor(np.array([10, 12], 'float32'))
        out1 = paddle.std(x).numpy()
        out2 = paddle.tensor.std(x).numpy()
        out3 = paddle.tensor.stat.std(x).numpy()
        np.testing.assert_allclose(out1, out2, rtol=1e-05)
        np.testing.assert_allclose(out1, out3, rtol=1e-05)
        paddle.enable_static()


class TestStdAPI_Compatibility(unittest.TestCase):
    def setUp(self):
        np.random.seed(2026)
        self.dtype = 'float32'
        self.shape = [1, 3, 4, 10]
        self.x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.place = get_device_place()

    def test_dygraph_compatibility(self):
        paddle.disable_static()
        x = paddle.tensor(self.x)
        # input arg
        out1_1 = paddle.std(x=x)
        out1_2 = paddle.std(input=x)
        np.testing.assert_allclose(out1_1.numpy(), out1_2.numpy(), rtol=1e-05)
        # dim arg
        out2_1 = paddle.std(x, axis=3)
        out2_2 = paddle.std(x, dim=3)
        np.testing.assert_allclose(out2_1.numpy(), out2_2.numpy(), rtol=1e-05)
        # out arg
        out3_1 = paddle.empty([])
        out3_2 = paddle.std(x, out=out3_1)
        np.testing.assert_allclose(out3_1.numpy(), out3_2.numpy(), rtol=1e-05)
        paddle.enable_static()

    def test_static_compatibility(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('x', self.shape, self.dtype)
            # input arg
            out1_1 = paddle.std(x=x)
            out1_2 = paddle.std(input=x)
            # dim arg
            out2_1 = paddle.std(x, axis=3)
            out2_2 = paddle.std(x, dim=3)
            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={'x': self.x}, fetch_list=[out1_1, out1_2, out2_1, out2_2]
            )
        np.testing.assert_allclose(res[0], res[1], rtol=1e-05)
        np.testing.assert_allclose(res[2], res[3], rtol=1e-05)


class TestStdAPI_Correction(unittest.TestCase):
    def setUp(self):
        np.random.seed(2026)
        self.dtype = 'float32'
        self.shape = [1, 3, 4, 10]
        self.set_attrs()
        self.x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        if self.axis:
            axis = tuple(self.axis)
            self.ref_out = np.std(self.x, axis, ddof=self.correction)
        else:
            self.ref_out = np.std(self.x, ddof=self.correction)
        self.place = get_device_place()

    def set_attrs(self):
        self.correction = 1
        self.axis = None

    def test_dygraph_correction(self):
        paddle.disable_static()
        x = paddle.tensor(self.x)
        if self.axis:
            out = paddle.std(x, self.axis, correction=self.correction)
        else:
            out = paddle.std(x, correction=self.correction)
        np.testing.assert_allclose(out.numpy(), self.ref_out, rtol=1e-05)
        paddle.enable_static()

    def test_static_correction(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('x', self.shape, self.dtype)
            if self.axis:
                out = paddle.std(x, self.axis, correction=self.correction)
            else:
                out = paddle.std(x, correction=self.correction)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': self.x}, fetch_list=[out])
        np.testing.assert_allclose(res[0], self.ref_out, rtol=1e-05)


class TestStdAPI_Correction2(TestStdAPI_Correction):
    def set_attrs(self):
        self.correction = 2
        self.axis = None


class TestStdAPI_CorrectionFloat(TestStdAPI_Correction):
    def set_attrs(self):
        self.correction = 1.5
        self.axis = None


class TestStdAPI_CorrectionWithAxis(TestStdAPI_Correction):
    def set_attrs(self):
        self.correction = 0
        self.axis = [1, 2]


class TestStdError(unittest.TestCase):
    def test_error(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', [2, 3, 4], 'int32')
            self.assertRaises(TypeError, paddle.std, x)


class Testfp16Std(unittest.TestCase):
    def test_fp16_with_gpu(self):
        paddle.enable_static()
        if paddle.base.core.is_compiled_with_cuda() or is_custom_device():
            place = get_device_place()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                input = np.random.random([12, 14]).astype("float16")
                x = paddle.static.data(
                    name="x", shape=[12, 14], dtype="float16"
                )

                y = paddle.std(x)

                exe = paddle.static.Executor(place)
                res = exe.run(
                    paddle.static.default_main_program(),
                    feed={
                        "x": input,
                    },
                    fetch_list=[y],
                )


class TestStdAPI_ZeroSize1(unittest.TestCase):
    def init_data(self):
        self.x_shape = [0]
        self.dtype = 'float64'
        self.expact_out = np.nan
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(self.dtype)

    def test_zerosize(self):
        self.init_data()
        paddle.disable_static()
        x = paddle.to_tensor(np.random.random(self.x_shape))
        out1 = paddle.std(x).numpy()
        np.testing.assert_allclose(out1, self.expact_out, equal_nan=True)
        paddle.enable_static()

    def test_static_zero(self):
        paddle.enable_static()
        self.init_data()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.x_shape, self.dtype)
            out = paddle.std(x)
            exe = paddle.static.Executor(paddle.CPUPlace())
            res = exe.run(feed={'X': self.x}, fetch_list=[out])
            np.testing.assert_allclose(self.expact_out, res[0], rtol=1e-05)
        paddle.disable_static()


class TestStdAPI_UnBiased1(unittest.TestCase):
    def init_data(self):
        self.x_shape = [1]
        # x = torch.randn([1])
        # res= torch.std(x,correction=0)     Here, res is 0.
        self.expect_out = 0.0

    def test_api(self):
        self.init_data()
        paddle.disable_static()
        x = paddle.to_tensor(np.random.random(self.x_shape))
        out1 = paddle.std(x, unbiased=False).numpy()
        np.testing.assert_allclose(out1, self.expect_out, equal_nan=True)
        paddle.enable_static()


class TestStdAPI_UnBiased2(unittest.TestCase):
    def init_data(self):
        self.x_shape = [1]
        # x = torch.randn([1])
        # res= torch.std(x,correction=1)     Here, res is 0.
        self.expect_out = np.nan

    def test_api(self):
        self.init_data()
        paddle.disable_static()
        x = paddle.to_tensor(np.random.random(self.x_shape))
        out1 = paddle.std(x, unbiased=True).numpy()
        np.testing.assert_allclose(out1, self.expect_out, equal_nan=True)
        paddle.enable_static()


class TestVarAPI_Backward1(unittest.TestCase):
    def test_api(self):
        paddle.disable_static()
        self.shape = []
        self.axis = []
        self.x = np.random.uniform(-1, 1, self.shape).astype('float64')
        paddle.set_device(paddle.CPUPlace())

        out_ref = ref_std(self.x, self.axis, True, False)
        x = paddle.to_tensor(self.x)
        x.stop_gradient = False
        out = paddle.std(x, self.axis, True, False)

        out.sum().backward()
        paddle.enable_static()


class TestVarAPI_Backward2(unittest.TestCase):
    def test_api(self):
        paddle.disable_static()
        self.shape = [2]
        self.axis = []
        self.x = np.random.uniform(-1, 1, self.shape).astype('float64')
        paddle.set_device(paddle.CPUPlace())

        out_ref = ref_std(self.x, self.axis, True, False)
        x = paddle.to_tensor(self.x)
        x.stop_gradient = False
        out = paddle.std(x, self.axis, True, False)

        out.sum().backward()
        paddle.enable_static()


class TestStdAPI_Backward_ZeroSize1(unittest.TestCase):
    def test_api(self):
        paddle.disable_static()
        self.shape = [1, 3, 0, 10]
        self.axis = [1, 3]
        self.x = np.random.uniform(-1, 1, self.shape).astype('float64')
        paddle.set_device(paddle.CPUPlace())

        out_ref = ref_std(self.x, self.axis, True, False)
        x = paddle.to_tensor(self.x)
        x.stop_gradient = False
        out = paddle.std(x, self.axis, True, False)

        out.sum().backward()
        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
