# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


class TestAminmaxAPI(unittest.TestCase):
    def setUp(self):
        self.init_case()
        self.place = paddle.CPUPlace()

    def init_case(self):
        self.x_np = np.array(
            [[0.2, 0.3, 0.5, 0.9], [0.1, 0.2, 0.6, 0.7]], dtype='float64'
        )
        self.shape = [2, 4]
        self.dtype = 'float64'
        self.axis = None
        self.keepdim = False

    def test_dygraph(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x_np, dtype=self.dtype, stop_gradient=False)
        min_val, max_val = paddle.aminmax(x, self.axis, self.keepdim)

        expected_min = np.amin(self.x_np, axis=self.axis, keepdims=self.keepdim)
        expected_max = np.amax(self.x_np, axis=self.axis, keepdims=self.keepdim)

        np.testing.assert_allclose(min_val.numpy(), expected_min, rtol=1e-05)
        np.testing.assert_allclose(max_val.numpy(), expected_max, rtol=1e-05)
        paddle.enable_static()

    def test_static(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name='x', dtype=self.dtype, shape=self.shape)
            min_val, max_val = paddle.aminmax(x, self.axis, self.keepdim)

            exe = paddle.static.Executor(self.place)
            fetches = exe.run(
                main,
                feed={'x': self.x_np},
                fetch_list=[min_val, max_val],
            )

            expected_min = np.amin(
                self.x_np, axis=self.axis, keepdims=self.keepdim
            )
            expected_max = np.amax(
                self.x_np, axis=self.axis, keepdims=self.keepdim
            )

            np.testing.assert_allclose(fetches[0], expected_min, rtol=1e-05)
            np.testing.assert_allclose(fetches[1], expected_max, rtol=1e-05)

    def test_dygraph_gradient(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x_np, dtype=self.dtype, stop_gradient=False)
        min_val, max_val = paddle.aminmax(x, self.axis, self.keepdim)
        loss = min_val.sum() + max_val.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        paddle.enable_static()


class TestAminmaxAPI_Axis0(TestAminmaxAPI):
    def init_case(self):
        self.x_np = np.array(
            [[0.2, 0.3, 0.5, 0.9], [0.1, 0.2, 0.6, 0.7]], dtype='float64'
        )
        self.shape = [2, 4]
        self.dtype = 'float64'
        self.axis = 0
        self.keepdim = False


class TestAminmaxAPI_Axis1(TestAminmaxAPI):
    def init_case(self):
        self.x_np = np.array(
            [[0.2, 0.3, 0.5, 0.9], [0.1, 0.2, 0.6, 0.7]], dtype='float64'
        )
        self.shape = [2, 4]
        self.dtype = 'float64'
        self.axis = 1
        self.keepdim = False


class TestAminmaxAPI_Keepdim(TestAminmaxAPI):
    def init_case(self):
        self.x_np = np.array(
            [[0.2, 0.3, 0.5, 0.9], [0.1, 0.2, 0.6, 0.7]], dtype='float64'
        )
        self.shape = [2, 4]
        self.dtype = 'float64'
        self.axis = 1
        self.keepdim = True


class TestAminmaxAPI_AxisNeg(TestAminmaxAPI):
    def init_case(self):
        self.x_np = np.array(
            [[0.2, 0.3, 0.5, 0.9], [0.1, 0.2, 0.6, 0.7]], dtype='float64'
        )
        self.shape = [2, 4]
        self.dtype = 'float64'
        self.axis = -1
        self.keepdim = False


class TestAminmaxAPI_3D(TestAminmaxAPI):
    def init_case(self):
        self.x_np = np.random.rand(2, 3, 4).astype('float64')
        self.shape = [2, 3, 4]
        self.dtype = 'float64'
        self.axis = 1
        self.keepdim = False


class TestAminmaxAPI_Float32(TestAminmaxAPI):
    def init_case(self):
        self.x_np = np.random.rand(3, 5).astype('float32')
        self.shape = [3, 5]
        self.dtype = 'float32'
        self.axis = None
        self.keepdim = False


class TestAminmaxAPI_ZeroDim(TestAminmaxAPI):
    def init_case(self):
        self.x_np = np.array(0.5).astype('float64')
        self.shape = []
        self.dtype = 'float64'
        self.axis = None
        self.keepdim = False


class TestAminmaxAPI_DuplicateValues(TestAminmaxAPI):
    """Test with multiple equal min/max elements (gradient distribution)."""

    def init_case(self):
        self.x_np = np.array(
            [[0.9, 0.1, 0.1, 0.9], [0.1, 0.9, 0.5, 0.1]], dtype='float64'
        )
        self.shape = [2, 4]
        self.dtype = 'float64'
        self.axis = None
        self.keepdim = False


class TestAminmaxCompatibility(unittest.TestCase):
    """Test paddle/torch keyword alias compatibility."""

    def setUp(self):
        self.np_x = np.array(
            [[0.2, 0.3, 0.5, 0.9], [0.1, 0.2, 0.6, 0.7]], dtype='float64'
        )

    def test_dygraph_aliases(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        min1, max1 = paddle.aminmax(x)
        min2, max2 = paddle.aminmax(x=x)
        min3, max3 = paddle.aminmax(input=x)
        min4, max4 = paddle.aminmax(x, axis=0)
        min5, max5 = paddle.aminmax(x, dim=0)

        ref_min = np.amin(self.np_x)
        ref_max = np.amax(self.np_x)
        for min_val in [min1, min2, min3]:
            np.testing.assert_allclose(ref_min, min_val.numpy())
        for max_val in [max1, max2, max3]:
            np.testing.assert_allclose(ref_max, max_val.numpy())

        ref_min_ax0 = np.amin(self.np_x, axis=0)
        ref_max_ax0 = np.amax(self.np_x, axis=0)
        for min_val in [min4, min5]:
            np.testing.assert_allclose(ref_min_ax0, min_val.numpy())
        for max_val in [max4, max5]:
            np.testing.assert_allclose(ref_max_ax0, max_val.numpy())

        paddle.enable_static()

    def test_static(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[2, 4], dtype='float64')
            min1, max1 = paddle.aminmax(x)
            min2, max2 = paddle.aminmax(x=x)
            min3, max3 = paddle.aminmax(input=x)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[min1, max1, min2, max2, min3, max3],
            )

            ref_min = np.amin(self.np_x)
            ref_max = np.amax(self.np_x)
            for i in range(0, len(fetches), 2):
                np.testing.assert_allclose(fetches[i], ref_min)
                np.testing.assert_allclose(fetches[i + 1], ref_max)


if __name__ == '__main__':
    unittest.main()
