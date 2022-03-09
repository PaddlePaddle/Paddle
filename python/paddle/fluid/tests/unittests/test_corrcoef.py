# -*- coding: UTF-8 -*-

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


def numpy_corrcoef(np_arr, rowvar=True, ddof=1):
    return np.corrcoef(np_arr,
                  rowvar=rowvar,
                  ddof=int(ddof))


class Corrcoef_Test(unittest.TestCase):
    def setUp(self):
        self.shape = [20, 10]

    def test_tensor_corrcoef_default(self):
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
                corr = paddle.linalg.corrcoef(tensor,
                                        rowvar=True,
                                        ddof=True)
                np_corr = numpy_corrcoef(
                    np_arr, rowvar=True, ddof=1)
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
                corr = paddle.linalg.corrcoef(tensor,
                                        rowvar=False,
                                        ddof=True)
                np_corr = numpy_corrcoef(
                    np_arr, rowvar=False, ddof=1)
                self.assertTrue(np.allclose(np_corr, corr.numpy()))

    def test_tensor_corr_ddof(self):
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
                corr = paddle.linalg.corrcoef(tensor,
                                        rowvar=True,
                                        ddof=False)
                np_corr = numpy_corrcoef(
                    np_arr, rowvar=True, ddof=0)
                self.assertTrue(np.allclose(np_corr, corr.numpy()))

class Corrcoef_Test2(Corrcoef_Test):
    def setUp(self):
        self.shape = [10]

if __name__ == '__main__':
    unittest.main()