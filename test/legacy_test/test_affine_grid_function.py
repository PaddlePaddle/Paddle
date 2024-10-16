# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.base.dygraph as dg
import paddle.nn.functional as F
from paddle import base


class AffineGridTestCase(unittest.TestCase):
    def __init__(
        self,
        methodName='runTest',
        theta_shape=(20, 2, 3),
        output_shape=[20, 2, 5, 7],
        align_corners=True,
        dtype="float32",
        invalid_theta=False,
        variable_output_shape=False,
    ):
        super().__init__(methodName)

        self.theta_shape = theta_shape
        self.output_shape = output_shape
        self.align_corners = align_corners
        self.dtype = dtype
        self.invalid_theta = invalid_theta
        self.variable_output_shape = variable_output_shape

    def setUp(self):
        self.theta = np.random.randn(*(self.theta_shape)).astype(self.dtype)

    def base_layer(self, place):
        paddle.enable_static()
        with base.unique_name.guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                theta_var = paddle.static.data(
                    "input", self.theta_shape, dtype=self.dtype
                )
                y_var = paddle.nn.functional.affine_grid(
                    theta_var, self.output_shape
                )
                feed_dict = {"input": self.theta}
                exe = paddle.static.Executor(place)
                (y_np,) = exe.run(
                    paddle.static.default_main_program(),
                    feed=feed_dict,
                    fetch_list=[y_var],
                )
                return y_np

    def functional(self, place):
        paddle.enable_static()
        with base.unique_name.guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                theta_var = paddle.static.data(
                    "input", self.theta_shape, dtype=self.dtype
                )
                y_var = F.affine_grid(
                    theta_var,
                    self.output_shape,
                    align_corners=self.align_corners,
                )
                feed_dict = {"input": self.theta}
                exe = paddle.static.Executor(place)
                (y_np,) = exe.run(
                    paddle.static.default_main_program(),
                    feed=feed_dict,
                    fetch_list=[y_var],
                )
                return y_np

    def test_static_api(self):
        place = base.CPUPlace()
        paddle.enable_static()
        with base.unique_name.guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                align_corners = True
                theta_var = paddle.static.data(
                    "input", self.theta_shape, dtype=self.dtype
                )
                y_var = paddle.nn.functional.affine_grid(
                    theta_var, self.output_shape
                )
                y_var2 = F.affine_grid(
                    theta_var,
                    self.output_shape,
                    align_corners=align_corners,
                )
                feed_dict = {"input": self.theta}
                exe = paddle.static.Executor(place)
                (y_np, y_np2) = exe.run(
                    paddle.static.default_main_program(),
                    feed=feed_dict,
                    fetch_list=[y_var, y_var2],
                )
                np.testing.assert_array_almost_equal(y_np, y_np2)

    def paddle_dygraph_layer(self):
        paddle.disable_static()
        theta_var = (
            paddle.to_tensor(self.theta)
            if not self.invalid_theta
            else "invalid"
        )
        output_shape = (
            paddle.to_tensor(self.output_shape)
            if self.variable_output_shape
            else self.output_shape
        )
        y_var = F.affine_grid(
            theta_var, output_shape, align_corners=self.align_corners
        )
        y_np = y_var.numpy()
        return y_np

    def _test_equivalence(self, place):
        place = base.CPUPlace()
        result1 = self.base_layer(place)
        result2 = self.functional(place)
        result3 = self.paddle_dygraph_layer()
        if self.align_corners:
            np.testing.assert_array_almost_equal(result1, result2)
        np.testing.assert_array_almost_equal(result2, result3)

    def runTest(self):
        place = base.CPUPlace()
        self._test_equivalence(place)

        if base.core.is_compiled_with_cuda():
            place = base.CUDAPlace(0)
            self._test_equivalence(place)


class AffineGridErrorTestCase(AffineGridTestCase):
    def runTest(self):
        place = base.CPUPlace()
        with dg.guard(place):
            with self.assertRaises(TypeError):
                self.paddle_dygraph_layer()


def add_cases(suite):
    suite.addTest(AffineGridTestCase(methodName='runTest'))
    suite.addTest(AffineGridTestCase(methodName='runTest', align_corners=True))

    suite.addTest(AffineGridTestCase(methodName='runTest', align_corners=False))
    suite.addTest(
        AffineGridTestCase(methodName='runTest', variable_output_shape=True)
    )

    suite.addTest(
        AffineGridTestCase(
            methodName='runTest',
            theta_shape=(20, 2, 3),
            output_shape=[20, 1, 7, 7],
            align_corners=True,
        )
    )


def add_error_cases(suite):
    suite.addTest(
        AffineGridErrorTestCase(methodName='runTest', output_shape="not_valid")
    )
    suite.addTest(
        AffineGridErrorTestCase(methodName='runTest', invalid_theta=True)
    )  # to test theta not variable error checking


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    add_cases(suite)
    add_error_cases(suite)
    return suite


if __name__ == '__main__':
    unittest.main()
