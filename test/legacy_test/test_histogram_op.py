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

import paddle
from paddle import base


class TestHistogramOpAPI(unittest.TestCase):
    """Test histogram api."""

    def test_static_graph(self):
        startup_program = paddle.static.Program()
        train_program = paddle.static.Program()
        with paddle.static.program_guard(train_program, startup_program):
            inputs = paddle.static.data(
                name='input', dtype='int64', shape=[2, 3]
            )
            output = paddle.histogram(inputs, bins=5, min=1, max=5)
            place = base.CPUPlace()
            if base.core.is_compiled_with_cuda():
                place = base.CUDAPlace(0)
            exe = base.Executor(place)
            img = np.array([[2, 4, 2], [2, 5, 4]]).astype(np.int64)
            res = exe.run(feed={'input': img}, fetch_list=[output])
            actual = np.array(res[0])
            expected = np.array([0, 3, 0, 2, 1]).astype(np.int64)
            self.assertTrue(
                (actual == expected).all(),
                msg='histogram output is wrong, out =' + str(actual),
            )

    def test_dygraph(self):
        with base.dygraph.guard():
            inputs_np = np.array([[2, 4, 2], [2, 5, 4]]).astype(np.int64)
            inputs = paddle.to_tensor(inputs_np)
            actual = paddle.histogram(inputs, bins=5, min=1, max=5)
            expected = np.array([0, 3, 0, 2, 1]).astype(np.int64)
            self.assertTrue(
                (actual.numpy() == expected).all(),
                msg='histogram output is wrong, out =' + str(actual.numpy()),
            )

            inputs_np = np.array([[2, 4, 2], [2, 5, 4]]).astype(np.int64)
            inputs = paddle.to_tensor(inputs_np)
            actual = paddle.histogram(inputs, bins=5, min=1, max=5)
            self.assertTrue(
                (actual.numpy() == expected).all(),
                msg='histogram output is wrong, out =' + str(actual.numpy()),
            )


class TestHistogramOpError(unittest.TestCase):
    """Test histogram op error."""

    def run_network(self, net_func):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            net_func()
            exe = base.Executor()
            exe.run(main_program)

    def test_bins_error(self):
        """Test bins should be greater than or equal to 1."""

        def net_func():
            input_value = paddle.tensor.fill_constant(
                shape=[3, 4], dtype='float32', value=3.0
            )
            paddle.histogram(input=input_value, bins=-1, min=1, max=5)

        with self.assertRaises(ValueError):
            self.run_network(net_func)

    def test_min_max_error(self):
        """Test max must be larger or equal to min."""

        def net_func():
            input_value = paddle.tensor.fill_constant(
                shape=[3, 4], dtype='float32', value=3.0
            )
            paddle.histogram(input=input_value, bins=1, min=5, max=1)

        with self.assertRaises(ValueError):
            self.run_network(net_func)

    def test_min_max_range_error(self):
        """Test range of min, max is not finite"""

        def net_func():
            input_value = paddle.tensor.fill_constant(
                shape=[3, 4], dtype='float32', value=3.0
            )
            paddle.histogram(input=input_value, bins=1, min=-np.inf, max=5)

        with self.assertRaises(ValueError):
            self.run_network(net_func)

    def test_input_range_error(self):
        """Test range of input is out of bound"""

        def net_func():
            input_value = paddle.to_tensor(
                [
                    -7095538316670326452,
                    -6102192280439741006,
                    2040176985344715288,
                    -6276983991026997920,
                    -6570715756420355710,
                    -5998045007776667296,
                    -6763099356862306438,
                    3166073479842736625,
                ],
                dtype=paddle.int64,
            )
            paddle.histogram(input=input_value, bins=1, min=0, max=0)

        with self.assertRaises(ValueError):
            self.run_network(net_func)

    def test_type_errors(self):
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(
                TypeError, paddle.histogram, 1, bins=5, min=1, max=5
            )
            # The input type must be 'int32', 'int64', 'float32', 'float64'
            x_bool = paddle.static.data(
                name='x_bool', shape=[4, 3], dtype='bool'
            )
            self.assertRaises(
                TypeError, paddle.histogram, x_bool, bins=5, min=1, max=5
            )


class TestHistogram(unittest.TestCase):
    """Test histogram api."""

    def setUp(self):
        self.init_test_case()
        self.input_np = np.random.uniform(
            low=0.0, high=20.0, size=self.in_shape
        ).astype(np.float32)
        self.weight_np = np.random.uniform(
            low=0.0, high=1.0, size=self.in_shape
        ).astype(np.float32)

    def init_test_case(self):
        self.in_shape = (10, 12)
        self.bins = 5
        self.min = 1
        self.max = 5
        self.density = False
        self.is_weight = True

    def test_static_graph(self):
        startup_program = paddle.static.Program()
        train_program = paddle.static.Program()
        with paddle.static.program_guard(train_program, startup_program):
            inputs = paddle.static.data(
                name='input', dtype='float32', shape=self.in_shape
            )
            if self.is_weight:
                weight = paddle.static.data(
                    name='weight', dtype='float32', shape=self.in_shape
                )
                output = paddle.histogram(
                    inputs,
                    bins=self.bins,
                    min=self.min,
                    max=self.max,
                    weight=weight,
                    density=self.density,
                )
            else:
                output = paddle.histogram(
                    inputs,
                    bins=self.bins,
                    min=self.min,
                    max=self.max,
                    density=self.density,
                )
            place = base.CPUPlace()
            if base.core.is_compiled_with_cuda():
                place = base.CUDAPlace(0)
            exe = base.Executor(place)
            if self.is_weight:
                res = exe.run(
                    feed={
                        'input': self.input_np,
                        'weight': self.weight_np,
                    },
                    fetch_list=[output],
                )
            else:
                res = exe.run(
                    feed={'input': self.input_np}, fetch_list=[output]
                )

            actual = np.array(res[0])
            Out, _ = np.histogram(
                self.input_np,
                bins=self.bins,
                range=(self.min, self.max),
                density=self.density,
                weights=self.weight_np if self.is_weight else None,
            )
            np.testing.assert_allclose(actual, Out, rtol=1e-58, atol=1e-5)

    def test_dygraph(self):
        with base.dygraph.guard():
            inputs_np = np.random.uniform(
                low=0.0, high=20.0, size=self.in_shape
            ).astype(np.float32)
            inputs = paddle.to_tensor(inputs_np)
            weight_np = np.random.uniform(
                low=0.0, high=1.0, size=self.in_shape
            ).astype(np.float32)
            weight = paddle.to_tensor(weight_np)
            actual = paddle.histogram(
                inputs,
                bins=5,
                min=1,
                max=5,
                weight=weight if self.is_weight else None,
                density=self.density,
            )
            Out, _ = np.histogram(
                inputs_np,
                bins=5,
                range=(1, 5),
                weights=weight_np if self.is_weight else None,
                density=self.density,
            )
            np.testing.assert_allclose(
                actual.numpy(), Out, rtol=1e-58, atol=1e-5
            )


class TestHistogramOpAPIWithDensity(TestHistogram):
    def init_test_case(self):
        self.in_shape = (10, 12)
        self.bins = 5
        self.min = 1
        self.max = 5
        self.density = True
        self.is_weight = False


class TestHistogramOpAPIWithWeight(TestHistogram):
    def init_test_case(self):
        self.in_shape = (10, 12)
        self.bins = 5
        self.min = 1
        self.max = 5
        self.density = False
        self.is_weight = True


class TestHistogramOpAPIWithWeightAndDensity(TestHistogram):
    def init_test_case(self):
        self.in_shape = (10, 12)
        self.bins = 5
        self.min = 1
        self.max = 5
        self.density = True
        self.is_weight = True


class TestHistogramOpAPIWithFloat32(TestHistogram):
    def init_test_case(self):
        self.in_shape = (10, 12)
        self.bins = 5
        self.min = 1
        self.max = 5
        self.density = False
        self.is_weight = False


class TestHistogramOp_ZeroDim(TestHistogram):
    def init_test_case(self):
        self.in_shape = []
        self.bins = 5
        self.min = 1
        self.max = 5
        self.density = False
        self.is_weight = False


class TestHistogramOpAPIWithFloatminMax(TestHistogram):
    def init_test_case(self):
        self.in_shape = (10, 12)
        self.bins = 4
        self.min = 2.2
        self.max = 4.5
        self.density = False
        self.is_weight = False


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
