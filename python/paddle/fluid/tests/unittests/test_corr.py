# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
import unittest
import numpy as np
import six
import paddle
import warnings


def numpy_corr(np_arr, rowvar=True):
    return np.corrcoef(np_arr, rowvar=rowvar)


class Corr_Test(unittest.TestCase):
    def setUp(self):
        self.shape = [20, 10]

    def test_tensor_corr_default(self):
        typelist = ['float64']
        places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for idx, p in enumerate(places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')

            for dtype in typelist:
                np_arr = np.random.rand(*self.shape).astype(dtype)
                tensor = paddle.to_tensor(np_arr, place=p)
                corr = paddle.linalg.corrcoef(tensor)
                np_corr = numpy_corr(np_arr, rowvar=True)
                self.assertTrue(np.allclose(np_corr, corr.numpy()))

    def test_tensor_corr_rowvar(self):
        typelist = ['float64']
        places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        for idx, p in enumerate(places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')

            for dtype in typelist:
                np_arr = np.random.rand(*self.shape).astype(dtype)
                tensor = paddle.to_tensor(np_arr, place=p)
                corr = paddle.linalg.corrcoef(tensor, rowvar=False)
                np_corr = numpy_corr(np_arr, rowvar=False)
                self.assertTrue(np.allclose(np_corr, corr.numpy()))



class Corr_Test2(Corr_Test):
    def setUp(self):
        self.shape = [10]


# Input(x) only support N-D (1<=N<=2) tensor
class Corr_Test3(unittest.TestCase):
    def setUp(self):
        self.shape = [2, 5, 10]

    def test_errors(self):
        def test_err():
            np_arr = np.random.rand(*self.shape).astype('float64')
            tensor = paddle.to_tensor(np_arr)
            covrr = paddle.linalg.corrcoef(tensor)

        self.assertRaises(ValueError, test_err)


class Corr_Test4(unittest.TestCase):
    def setUp(self):
        self.shape = [2, 2, 5, 10]

    def test_errors(self):
        def test_err():
            np_arr = np.random.rand(*self.shape).astype('float64')
            tensor = paddle.to_tensor(np_arr)
            corr = paddle.linalg.corrcoef(tensor)

        self.assertRaises(ValueError, test_err)


class Corr_Test5(unittest.TestCase):
    def setUp(self):
        self.shape = [2, 5, 10, 6, 7]

    def test_errors(self):
        def test_err():
            np_arr = np.random.rand(*self.shape).astype('float64')
            tensor = paddle.to_tensor(np_arr)
            corr = paddle.linalg.corrcoef(tensor)

        self.assertRaises(ValueError, test_err)


if __name__ == '__main__':
    unittest.main()
