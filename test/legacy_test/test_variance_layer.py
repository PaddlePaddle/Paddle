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

import paddle


def ref_var(x, axis=None, unbiased=True, keepdim=False):
    ddof = 1 if unbiased else 0
    if isinstance(axis, int):
        axis = (axis,)
    if axis is not None:
        axis = tuple(axis)
    return np.var(x, axis=axis, ddof=ddof, keepdims=keepdim)


class TestVarAPI(unittest.TestCase):
    def setUp(self):
        self.dtype = 'float64'
        self.shape = [1, 3, 4, 10]
        self.axis = [1, 3]
        self.keepdim = False
        self.unbiased = True
        self.set_attrs()
        self.x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.base.core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def set_attrs(self):
        pass

    def static(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.shape, self.dtype)
            out = paddle.var(x, self.axis, self.unbiased, self.keepdim)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x}, fetch_list=[out])
        return res[0]

    def dygraph(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        out = paddle.var(x, self.axis, self.unbiased, self.keepdim)
        paddle.enable_static()
        return out.numpy()

    def test_api(self):
        out_ref = ref_var(self.x, self.axis, self.unbiased, self.keepdim)
        out_dygraph = self.dygraph()

        np.testing.assert_allclose(out_ref, out_dygraph, rtol=1e-05)
        self.assertTrue(np.equal(out_ref.shape, out_dygraph.shape).all())

        def test_static_or_pir_mode():
            out_static = self.static()
            np.testing.assert_allclose(out_ref, out_static, rtol=1e-05)
            self.assertTrue(np.equal(out_ref.shape, out_static.shape).all())

        test_static_or_pir_mode()


class TestVarAPI_dtype(TestVarAPI):
    def set_attrs(self):
        self.dtype = 'float32'


class TestVarAPI_axis_int(TestVarAPI):
    def set_attrs(self):
        self.axis = 2


class TestVarAPI_axis_list(TestVarAPI):
    def set_attrs(self):
        self.axis = [1, 2]


class TestVarAPI_axis_tuple(TestVarAPI):
    def set_attrs(self):
        self.axis = (1, 3)


class TestVarAPI_keepdim(TestVarAPI):
    def set_attrs(self):
        self.keepdim = False


class TestVarAPI_unbiased(TestVarAPI):
    def set_attrs(self):
        self.unbiased = False


class TestVarAPI_alias(unittest.TestCase):
    def test_alias(self):
        paddle.disable_static()
        x = paddle.to_tensor(np.array([10, 12], 'float32'))
        out1 = paddle.var(x).numpy()
        out2 = paddle.tensor.var(x).numpy()
        out3 = paddle.tensor.stat.var(x).numpy()
        np.testing.assert_allclose(out1, out2, rtol=1e-05)
        np.testing.assert_allclose(out1, out3, rtol=1e-05)
        paddle.enable_static()


class TestVarError(unittest.TestCase):

    def test_error(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', [2, 3, 4], 'int32')
            self.assertRaises(TypeError, paddle.var, x)


class TestVarAPI_ZeroSize(unittest.TestCase):
    def init_data(self):
        self.x_shape = [10, 0]

    def test_zerosize(self):
        self.init_data()
        paddle.disable_static()
        x = paddle.to_tensor(np.random.random(self.x_shape))
        out1 = paddle.var(x).numpy()
        out2 = np.var(x.numpy())
        np.testing.assert_allclose(out1, out2, equal_nan=True)
        paddle.enable_static()


class TestVarAPI_ZeroSize1(unittest.TestCase):
    def init_data(self):
        self.x_shape = []
        # x = torch.tensor([])
        # res= torch.var(x)     Here, res is nan
        self.expact_out = np.nan

    def test_zerosize(self):
        self.init_data()
        paddle.disable_static()
        x = paddle.to_tensor(np.random.random(self.x_shape))
        out1 = paddle.var(x).numpy()
        np.testing.assert_allclose(out1, self.expact_out, equal_nan=True)
        paddle.enable_static()


class TestVarAPI_UnBiased1(unittest.TestCase):
    def init_data(self):
        self.x_shape = [1]
        # x = torch.randn([1])
        # res= torch.var(x,correction=0)     Here, res is 0.
        self.expact_out = 0.0

    def test_api(self):
        self.init_data()
        paddle.disable_static()
        x = paddle.to_tensor(np.random.random(self.x_shape))
        out1 = paddle.var(x, unbiased=False).numpy()
        np.testing.assert_allclose(out1, self.expact_out, equal_nan=True)
        paddle.enable_static()


class TestVarAPI_UnBiased2(unittest.TestCase):
    def init_data(self):
        self.x_shape = [1]
        # x = torch.randn([1])
        # res= torch.var(x,correction=1)     Here, res is 0.
        self.expact_out = np.nan

    def test_api(self):
        self.init_data()
        paddle.disable_static()
        x = paddle.to_tensor(np.random.random(self.x_shape))
        out1 = paddle.var(x, unbiased=True).numpy()
        np.testing.assert_allclose(out1, self.expact_out, equal_nan=True)
        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
