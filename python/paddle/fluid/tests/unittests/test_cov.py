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


def numpy_cov(np_arr, rowvar=True, ddof=1, fweights=None, aweights=None):
    return np.cov(np_arr,
                  rowvar=rowvar,
                  ddof=int(ddof),
                  fweights=fweights,
                  aweights=aweights)


class Cov_Test(unittest.TestCase):
    def setUp(self):
        self.shape = [20, 10]
        self.weightshape = [10]

    def test_tensor_cov_default(self):
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
                cov = paddle.linalg.cov(tensor,
                                        rowvar=True,
                                        ddof=True,
                                        fweights=None,
                                        aweights=None)
                np_cov = numpy_cov(
                    np_arr, rowvar=True, ddof=1, fweights=None, aweights=None)
                self.assertTrue(np.allclose(np_cov, cov.numpy()))

    def test_tensor_cov_rowvar(self):
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
                cov = paddle.linalg.cov(tensor,
                                        rowvar=False,
                                        ddof=True,
                                        fweights=None,
                                        aweights=None)
                np_cov = numpy_cov(
                    np_arr, rowvar=False, ddof=1, fweights=None, aweights=None)
                self.assertTrue(np.allclose(np_cov, cov.numpy()))

    def test_tensor_cov_ddof(self):
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
                cov = paddle.linalg.cov(tensor,
                                        rowvar=True,
                                        ddof=False,
                                        fweights=None,
                                        aweights=None)
                np_cov = numpy_cov(
                    np_arr, rowvar=True, ddof=0, fweights=None, aweights=None)
                self.assertTrue(np.allclose(np_cov, cov.numpy()))

    def test_tensor_cov_fweights(self):
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
                np_fw = np.random.randint(
                    10, size=self.weightshape).astype('int32')
                tensor = paddle.to_tensor(np_arr, place=p)
                fweights = paddle.to_tensor(np_fw, place=p)
                cov = paddle.linalg.cov(tensor,
                                        rowvar=True,
                                        ddof=True,
                                        fweights=fweights,
                                        aweights=None)
                np_cov = numpy_cov(
                    np_arr, rowvar=True, ddof=1, fweights=np_fw, aweights=None)
                self.assertTrue(np.allclose(np_cov, cov.numpy()))

    def test_tensor_cov_aweights(self):
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
                np_aw = np.random.randint(
                    10, size=self.weightshape).astype('int32')
                tensor = paddle.to_tensor(np_arr, place=p)
                aweights = paddle.to_tensor(np_aw, place=p)
                cov = paddle.linalg.cov(tensor,
                                        rowvar=True,
                                        ddof=True,
                                        fweights=None,
                                        aweights=aweights)
                np_cov = numpy_cov(
                    np_arr, rowvar=True, ddof=1, fweights=None, aweights=np_aw)
                self.assertTrue(np.allclose(np_cov, cov.numpy()))

    def test_tensor_cov_weights(self):
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
                np_fw = np.random.randint(
                    10, size=self.weightshape).astype('int64')
                np_aw = np.random.rand(*self.weightshape).astype('float64')
                tensor = paddle.to_tensor(np_arr, place=p)
                fweights = paddle.to_tensor(np_fw, place=p)
                aweights = paddle.to_tensor(np_aw, place=p)
                cov = paddle.linalg.cov(tensor,
                                        rowvar=True,
                                        ddof=True,
                                        fweights=fweights,
                                        aweights=aweights)
                np_cov = numpy_cov(
                    np_arr, rowvar=True, ddof=1, fweights=np_fw, aweights=np_aw)
                self.assertTrue(np.allclose(np_cov, cov.numpy()))


class Cov_Test2(Cov_Test):
    def setUp(self):
        self.shape = [10]
        self.weightshape = [10]


# Input(x) only support N-D (1<=N<=2) tensor
class Cov_Test3(unittest.TestCase):
    def setUp(self):
        self.shape = [2, 5, 10]
        self.fweightshape = [10]
        self.aweightshape = [10]
        self.fw_s = 1.
        self.aw_s = 1.

    def test_errors(self):
        def test_err():
            np_arr = np.random.rand(*self.shape).astype('float64')
            np_fw = self.fw_s * np.random.rand(
                *self.fweightshape).astype('int32')
            np_aw = self.aw_s * np.random.rand(
                *self.aweightshape).astype('float64')
            tensor = paddle.to_tensor(np_arr)
            fweights = paddle.to_tensor(np_fw)
            aweights = paddle.to_tensor(np_aw)
            cov = paddle.linalg.cov(tensor,
                                    rowvar=True,
                                    ddof=True,
                                    fweights=fweights,
                                    aweights=aweights)

        self.assertRaises(ValueError, test_err)


#Input(fweights) only support N-D (N<=1) tensor
class Cov_Test4(Cov_Test3):
    def setUp(self):
        self.shape = [5, 10]
        self.fweightshape = [2, 10]
        self.aweightshape = [10]
        self.fw_s = 1.
        self.aw_s = 1.


#The number of Input(fweights) should equal to x's dim[1]
class Cov_Test5(Cov_Test3):
    def setUp(self):
        self.shape = [5, 10]
        self.fweightshape = [5]
        self.aweightshape = [10]
        self.fw_s = 1.
        self.aw_s = 1.


#The value of Input(fweights) cannot be negtive
class Cov_Test6(Cov_Test3):
    def setUp(self):
        self.shape = [5, 10]
        self.fweightshape = [10]
        self.aweightshape = [10]
        self.fw_s = -1.
        self.aw_s = 1.


#Input(aweights) only support N-D (N<=1) tensor
class Cov_Test7(Cov_Test3):
    def setUp(self):
        self.shape = [5, 10]
        self.fweightshape = [10]
        self.aweightshape = [2, 10]
        self.fw_s = 1.
        self.aw_s = 1.


#The number of Input(aweights) should equal to x's dim[1]
class Cov_Test8(Cov_Test3):
    def setUp(self):
        self.shape = [5, 10]
        self.fweightshape = [10]
        self.aweightshape = [5]
        self.fw_s = 1.
        self.aw_s = 1.


#The value of Input(aweights) cannot be negtive
class Cov_Test9(Cov_Test3):
    def setUp(self):
        self.shape = [5, 10]
        self.fweightshape = [10]
        self.aweightshape = [10]
        self.fw_s = 1.
        self.aw_s = -1.


if __name__ == '__main__':
    unittest.main()
