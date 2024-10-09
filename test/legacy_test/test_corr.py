# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import numpy as np

import paddle
from paddle import base


def numpy_corr(np_arr, rowvar=True, dtype='float64'):
    # np.corrcoef support parameter 'dtype' since 1.20
    if np.lib.NumpyVersion(np.__version__) < "1.20.0":
        return np.corrcoef(np_arr, rowvar=rowvar)
    return np.corrcoef(np_arr, rowvar=rowvar, dtype=dtype)


class Corr_Test(unittest.TestCase):
    def setUp(self):
        self.shape = [4, 5]

    def test_tensor_corr_default(self):
        typelist = ['float64', 'float32']
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not base.core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if base.core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for idx, p in enumerate(places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')

            for dtype in typelist:
                np_arr = np.random.rand(*self.shape).astype(dtype)
                tensor = paddle.to_tensor(np_arr, place=p)
                corr = paddle.linalg.corrcoef(tensor)
                np_corr = numpy_corr(np_arr, rowvar=True, dtype=dtype)
                if dtype == 'float32':
                    np.testing.assert_allclose(
                        np_corr, corr.numpy(), rtol=1e-05, atol=1e-05
                    )
                else:
                    np.testing.assert_allclose(
                        np_corr, corr.numpy(), rtol=1e-05
                    )

    def test_tensor_corr_rowvar(self):
        typelist = ['float64', 'float32']
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not base.core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if base.core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))

        for idx, p in enumerate(places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')

            for dtype in typelist:
                np_arr = np.random.rand(*self.shape).astype(dtype)
                tensor = paddle.to_tensor(np_arr, place=p)
                corr = paddle.linalg.corrcoef(tensor, rowvar=False)
                np_corr = numpy_corr(np_arr, rowvar=False, dtype=dtype)
                if dtype == 'float32':
                    np.testing.assert_allclose(
                        np_corr, corr.numpy(), rtol=1e-05, atol=1e-05
                    )
                else:
                    np.testing.assert_allclose(
                        np_corr, corr.numpy(), rtol=1e-05
                    )


# Input(x) only support N-D (1<=N<=2) tensor
class Corr_Test2(Corr_Test):
    def setUp(self):
        self.shape = [10]


class Corr_Test3(Corr_Test):
    def setUp(self):
        self.shape = [4, 5]


# Input(x) only support N-D (1<=N<=2) tensor
class Corr_Test4(unittest.TestCase):
    def setUp(self):
        self.shape = [2, 5, 2]

    def test_errors(self):
        def test_err():
            np_arr = np.random.rand(*self.shape).astype('float64')
            tensor = paddle.to_tensor(np_arr)
            covrr = paddle.linalg.corrcoef(tensor)

        self.assertRaises(ValueError, test_err)


# test unsupported complex input
class Corr_Comeplex_Test(unittest.TestCase):
    def setUp(self):
        self.dtype = 'complex128'

    def test_errors(self):
        paddle.enable_static()
        x1 = paddle.static.data(name=self.dtype, shape=[2], dtype=self.dtype)
        self.assertRaises(TypeError, paddle.linalg.corrcoef, x=x1)
        paddle.disable_static()


class Corr_Test5(Corr_Comeplex_Test):
    def setUp(self):
        self.dtype = 'complex64'


if __name__ == '__main__':
    unittest.main()
