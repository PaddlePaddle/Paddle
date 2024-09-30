#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base import core


class LinalgLstsqTestCase(unittest.TestCase):
    def setUp(self):
        self.devices = ["cpu"]
        self.init_config()
        if core.is_compiled_with_cuda() and self.driver == "gels":
            self.devices.append("gpu")
        self.generate_input()
        self.generate_output()
        np.random.seed(2022)

    def init_config(self):
        self.dtype = 'float64'
        self.rcond = 1e-15
        self.driver = "gelsd"
        self._input_shape_1 = (5, 4)
        self._input_shape_2 = (5, 3)

    def generate_input(self):
        self._input_data_1 = np.random.random(self._input_shape_1).astype(
            self.dtype
        )
        self._input_data_2 = np.random.random(self._input_shape_2).astype(
            self.dtype
        )

    def generate_output(self):
        if len(self._input_shape_1) == 2:
            out = np.linalg.lstsq(
                self._input_data_1, self._input_data_2, rcond=self.rcond
            )
            self._output_solution = out[0]
            self._output_residuals = out[1]
            self._output_rank = out[2]
            self._output_sg_values = out[3]
        elif len(self._input_shape_1) == 3:
            self._output_solution = []
            self._output_residuals = []
            self._output_rank = []
            self._output_sg_values = []
            for i in range(self._input_shape_1[0]):
                out = np.linalg.lstsq(
                    self._input_data_1[i],
                    self._input_data_2[i],
                    rcond=self.rcond,
                )
                self._output_solution.append(out[0])
                self._output_residuals.append(out[1])
                self._output_rank.append(out[2])
                self._output_sg_values.append(out[3])

    def test_eager_dygraph(self):
        paddle.disable_static()
        for dev in self.devices:
            paddle.set_device(dev)
            place = paddle.CPUPlace() if dev == "cpu" else paddle.CUDAPlace(0)
            x = paddle.to_tensor(
                self._input_data_1, place=place, dtype=self.dtype
            )
            y = paddle.to_tensor(
                self._input_data_2, place=place, dtype=self.dtype
            )
            results = paddle.linalg.lstsq(
                x, y, rcond=self.rcond, driver=self.driver
            )
            self._result_solution = results[0].numpy()
            self._result_residuals = results[1].numpy()
            self._result_rank = results[2].numpy()
            self._result_sg_values = results[3].numpy()
            self.assert_np_close()

    def test_static(self):
        paddle.enable_static()
        for dev in self.devices:
            paddle.set_device(dev)
            place = base.CPUPlace() if dev == "cpu" else base.CUDAPlace(0)
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(
                    name="x",
                    shape=self._input_shape_1,
                    dtype=self._input_data_1.dtype,
                )
                y = paddle.static.data(
                    name="y",
                    shape=self._input_shape_2,
                    dtype=self._input_data_2.dtype,
                )
                results = paddle.linalg.lstsq(
                    x, y, rcond=self.rcond, driver=self.driver
                )
                exe = base.Executor(place)
                fetches = exe.run(
                    feed={"x": self._input_data_1, "y": self._input_data_2},
                    fetch_list=[results],
                )
                self._result_solution = fetches[0]
                self._result_residuals = fetches[1]
                self._result_rank = fetches[2]
                self._result_sg_values = fetches[3]
                self.assert_np_close()

    def assert_np_close(self):
        if len(self._input_shape_1) == 2:
            np.testing.assert_allclose(
                self._result_solution, self._output_solution, rtol=1e-3
            )
            if (
                self._input_shape_1[-2] > self._input_shape_1[-1]
                and self._output_rank == self._input_shape_1[-1]
            ):
                np.testing.assert_allclose(
                    self._result_residuals, self._output_residuals, rtol=1e-5
                )
            if self.driver in ("gelsy", "gelsd", "gelss"):
                np.testing.assert_allclose(
                    self._result_rank, self._output_rank, rtol=1e-5
                )
            if self.driver in ("gelsd", "gelss"):
                np.testing.assert_allclose(
                    self._result_sg_values, self._output_sg_values, rtol=1e-5
                )
        else:
            for i in range(len(self._output_solution)):
                np.testing.assert_allclose(
                    self._result_solution[i],
                    self._output_solution[i],
                    rtol=1e-3,
                )
                if (
                    self._input_shape_1[-2] > self._input_shape_1[-1]
                    and self._output_rank[i] == self._input_shape_1[-1]
                ):
                    np.testing.assert_allclose(
                        self._result_residuals[i],
                        self._output_residuals[i],
                        rtol=1e-5,
                    )
                if self.driver in ("gelsy", "gelsd", "gelss"):
                    np.testing.assert_allclose(
                        self._result_rank[i], self._output_rank[i], rtol=1e-5
                    )
                if self.driver in ("gelsd", "gelss"):
                    np.testing.assert_allclose(
                        self._result_sg_values[i],
                        self._output_sg_values[i],
                        rtol=1e-5,
                    )


class LinalgLstsqTestCase1(LinalgLstsqTestCase):
    def init_config(self):
        self.dtype = 'float32'
        self.rcond = 1e-15
        self.driver = "gels"
        self._input_shape_1 = (9, 9)
        self._input_shape_2 = (9, 5)


class LinalgLstsqTestCase2(LinalgLstsqTestCase):
    def init_config(self):
        self.dtype = 'float64'
        self.rcond = 1e-15
        self.driver = "gels"
        self._input_shape_1 = (5, 10)
        self._input_shape_2 = (5, 8)


class LinalgLstsqTestCase3(LinalgLstsqTestCase):
    def init_config(self):
        self.dtype = 'float64'
        self.rcond = 1e-15
        self.driver = "gels"
        self._input_shape_1 = (10, 7, 3)
        self._input_shape_2 = (10, 7, 6)


class LinalgLstsqTestCaseRcond(LinalgLstsqTestCase):
    def init_config(self):
        self.dtype = 'float64'
        self.rcond = 1e-7
        self.driver = "gelsd"
        self._input_shape_1 = (3, 2)
        self._input_shape_2 = (3, 3)


class LinalgLstsqTestCaseGelsFloat32(LinalgLstsqTestCase):
    def init_config(self):
        self.dtype = 'float32'
        self.rcond = None
        self.driver = "gels"
        self._input_shape_1 = (10, 5)
        self._input_shape_2 = (10, 8)


class LinalgLstsqTestCaseGelsFloat64(LinalgLstsqTestCase):
    def init_config(self):
        self.dtype = 'float32'
        self.rcond = None
        self.driver = "gels"
        self._input_shape_1 = (3, 2, 8)
        self._input_shape_2 = (3, 2, 15)


class LinalgLstsqTestCaseGelssFloat64(LinalgLstsqTestCase):
    def init_config(self):
        self.dtype = 'float64'
        self.rcond = None
        self.driver = "gelss"
        self._input_shape_1 = (5, 5)
        self._input_shape_2 = (5, 1)


class LinalgLstsqTestCaseGelsyFloat32(LinalgLstsqTestCase):
    def init_config(self):
        self.dtype = 'float32'
        self.rcond = 1e-15
        self.driver = "gelsy"
        self._input_shape_1 = (8, 2)
        self._input_shape_2 = (8, 10)


class LinalgLstsqTestCaseBatch1(LinalgLstsqTestCase):
    def init_config(self):
        self.dtype = 'float32'
        self.rcond = 1e-15
        self.driver = "gelss"
        self._input_shape_1 = (2, 3, 10)
        self._input_shape_2 = (2, 3, 4)


class LinalgLstsqTestCaseBatch2(LinalgLstsqTestCase):
    def init_config(self):
        self.dtype = 'float64'
        self.rcond = 1e-15
        self.driver = "gels"
        self._input_shape_1 = (10, 8, 6)
        self._input_shape_2 = (10, 8, 10)


class LinalgLstsqTestCaseLarge1(LinalgLstsqTestCase):
    def init_config(self):
        self.dtype = 'float64'
        self.rcond = 1e-15
        self.driver = "gelsd"
        self._input_shape_1 = (200, 100)
        self._input_shape_2 = (200, 50)


class LinalgLstsqTestCaseLarge2(LinalgLstsqTestCase):
    def init_config(self):
        self.dtype = 'float64'
        self.rcond = 1e-15
        self.driver = "gelss"
        self._input_shape_1 = (50, 600)
        self._input_shape_2 = (50, 300)


class TestLinalgLstsqAPIError(unittest.TestCase):
    def setUp(self):
        pass

    def test_api_errors(self):
        def test_x_bad_shape():
            x = paddle.to_tensor(np.random.random(size=(5)), dtype=np.float32)
            y = paddle.to_tensor(
                np.random.random(size=(5, 15)), dtype=np.float32
            )
            out = paddle.linalg.lstsq(x, y, driver='gelsy')

        def test_y_bad_shape():
            x = paddle.to_tensor(
                np.random.random(size=(5, 10)), dtype=np.float32
            )
            y = paddle.to_tensor(np.random.random(size=(5)), dtype=np.float32)
            out = paddle.linalg.lstsq(x, y, driver='gelsy')

        def test_shape_dismatch():
            x = paddle.to_tensor(
                np.random.random(size=(5, 10)), dtype=np.float32
            )
            y = paddle.to_tensor(
                np.random.random(size=(4, 15)), dtype=np.float32
            )
            out = paddle.linalg.lstsq(x, y, driver='gelsy')

        self.assertRaises(ValueError, test_x_bad_shape)
        self.assertRaises(ValueError, test_y_bad_shape)
        self.assertRaises(ValueError, test_shape_dismatch)


if __name__ == '__main__':
    unittest.main()
