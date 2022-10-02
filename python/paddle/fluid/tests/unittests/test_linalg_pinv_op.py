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
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
from op_test import OpTest, skip_check_grad_ci
from gradient_checker import grad_check
from decorator_helper import prog_scope


class LinalgPinvTestCase(unittest.TestCase):

    def setUp(self):
        self.init_config()
        self.generate_input()
        self.generate_output()
        self.places = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def generate_input(self):
        self._input_shape = (5, 5)
        np.random.seed(123)
        self._input_data = np.random.random(self._input_shape).astype(
            self.dtype)

    def generate_output(self):
        self._output_data = np.linalg.pinv(self._input_data, \
            rcond=self.rcond, hermitian=self.hermitian)

    def init_config(self):
        self.dtype = 'float64'
        self.rcond = 1e-15
        self.hermitian = False

    def test_dygraph(self):
        for place in self.places:
            paddle.disable_static(place)
            x = paddle.to_tensor(self._input_data, place=place)
            out = paddle.linalg.pinv(x,
                                     rcond=self.rcond,
                                     hermitian=self.hermitian).numpy()
            if (np.abs(out - self._output_data) < 1e-6).any():
                pass
            else:
                print("EXPECTED: \n", self._output_data)
                print("GOT     : \n", out)
                raise RuntimeError("Check PINV dygraph Failed")

    def test_static(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for place in places:
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                x = paddle.fluid.data(name="input",
                                      shape=self._input_shape,
                                      dtype=self._input_data.dtype)
                out = paddle.linalg.pinv(x,
                                         rcond=self.rcond,
                                         hermitian=self.hermitian)
                exe = fluid.Executor(place)
                fetches = exe.run(fluid.default_main_program(),
                                  feed={"input": self._input_data},
                                  fetch_list=[out])
                if (np.abs(fetches[0] - self._output_data) < 1e-6).any():
                    pass
                else:
                    print("EXPECTED: \n", self._output_data)
                    print("GOT     : \n", fetches[0])
                    raise RuntimeError("Check PINV static Failed")

    def test_grad(self):
        for place in self.places:
            x = paddle.to_tensor(self._input_data,
                                 place=place,
                                 stop_gradient=False)
            out = paddle.linalg.pinv(x,
                                     rcond=self.rcond,
                                     hermitian=self.hermitian)
            try:
                out.backward()
                x_grad = x.grad
                # print(x_grad)
            except:
                raise RuntimeError("Check PINV Grad Failed")


class LinalgPinvTestCase1(LinalgPinvTestCase):

    def generate_input(self):
        self._input_shape = (4, 5)
        np.random.seed(123)
        self._input_data = np.random.random(self._input_shape).astype(
            self.dtype)


class LinalgPinvTestCase2(LinalgPinvTestCase):

    def generate_input(self):
        self._input_shape = (5, 4)
        np.random.seed(123)
        self._input_data = np.random.random(self._input_shape).astype(
            self.dtype)


class LinalgPinvTestCaseBatch1(LinalgPinvTestCase):

    def generate_input(self):
        self._input_shape = (3, 5, 5)
        np.random.seed(123)
        self._input_data = np.random.random(self._input_shape).astype(
            self.dtype)


class LinalgPinvTestCaseBatch2(LinalgPinvTestCase):

    def generate_input(self):
        self._input_shape = (3, 4, 5)
        np.random.seed(123)
        self._input_data = np.random.random(self._input_shape).astype(
            self.dtype)


class LinalgPinvTestCaseBatch3(LinalgPinvTestCase):

    def generate_input(self):
        self._input_shape = (3, 5, 4)
        np.random.seed(123)
        self._input_data = np.random.random(self._input_shape).astype(
            self.dtype)


class LinalgPinvTestCaseBatch4(LinalgPinvTestCase):

    def generate_input(self):
        self._input_shape = (3, 6, 5, 4)
        np.random.seed(123)
        self._input_data = np.random.random(self._input_shape).astype(
            self.dtype)


class LinalgPinvTestCaseBatchBig(LinalgPinvTestCase):

    def generate_input(self):
        self._input_shape = (2, 200, 300)
        np.random.seed(123)
        self._input_data = np.random.random(self._input_shape).astype(
            self.dtype)


class LinalgPinvTestCaseFP32(LinalgPinvTestCase):

    def generate_input(self):
        self._input_shape = (3, 5, 5)
        np.random.seed(123)
        self._input_data = np.random.random(self._input_shape).astype(
            self.dtype)

    def init_config(self):
        self.dtype = 'float32'
        self.rcond = 1e-15
        self.hermitian = False


class LinalgPinvTestCaseRcond(LinalgPinvTestCase):

    def generate_input(self):
        self._input_shape = (3, 5, 5)
        np.random.seed(123)
        self._input_data = np.random.random(self._input_shape).astype(
            self.dtype)

    def init_config(self):
        self.dtype = 'float64'
        self.rcond = 1e-10
        self.hermitian = False


class LinalgPinvTestCaseHermitian1(LinalgPinvTestCase):

    def generate_input(self):
        self._input_shape = (5, 5)
        np.random.seed(123)
        x = np.random.random(self._input_shape).astype(self.dtype) + \
            1J * np.random.random(self._input_shape).astype(self.dtype)
        self._input_data = x + x.transpose().conj()

    def init_config(self):
        self.dtype = 'float64'
        self.rcond = 1e-15
        self.hermitian = True


class LinalgPinvTestCaseHermitian2(LinalgPinvTestCase):

    def generate_input(self):
        self._input_shape = (3, 5, 5)
        np.random.seed(123)
        x = np.random.random(self._input_shape).astype(self.dtype) + \
            1J * np.random.random(self._input_shape).astype(self.dtype)
        self._input_data = x + x.transpose((0, 2, 1)).conj()

    def init_config(self):
        self.dtype = 'float64'
        self.rcond = 1e-15
        self.hermitian = True


class LinalgPinvTestCaseHermitian3(LinalgPinvTestCase):

    def generate_input(self):
        self._input_shape = (3, 5, 5)
        np.random.seed(123)
        x = np.random.random(self._input_shape).astype(self.dtype) + \
            1J * np.random.random(self._input_shape).astype(self.dtype)
        self._input_data = x + x.transpose((0, 2, 1)).conj()

    def init_config(self):
        self.dtype = 'float32'
        self.rcond = 1e-15
        self.hermitian = True


class LinalgPinvTestCaseHermitian4(LinalgPinvTestCase):

    def generate_input(self):
        self._input_shape = (5, 5)
        np.random.seed(123)
        x = np.random.random(self._input_shape).astype(self.dtype)
        self._input_data = x + x.transpose()

    def init_config(self):
        self.dtype = 'float64'
        self.rcond = 1e-15
        self.hermitian = True


class LinalgPinvTestCaseHermitian5(LinalgPinvTestCase):

    def generate_input(self):
        self._input_shape = (3, 5, 5)
        np.random.seed(123)
        x = np.random.random(self._input_shape).astype(self.dtype)
        self._input_data = x + x.transpose((0, 2, 1))

    def init_config(self):
        self.dtype = 'float64'
        self.rcond = 1e-15
        self.hermitian = True


class LinalgPinvTestCaseHermitianFP32(LinalgPinvTestCase):

    def generate_input(self):
        self._input_shape = (3, 5, 5)
        np.random.seed(123)
        x = np.random.random(self._input_shape).astype(self.dtype)
        self._input_data = x + x.transpose((0, 2, 1))

    def init_config(self):
        self.dtype = 'float32'
        self.rcond = 1e-15
        self.hermitian = True


if __name__ == '__main__':
    unittest.main()
