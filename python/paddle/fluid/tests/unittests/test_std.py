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
import paddle.tensor as tensor
import unittest
import numpy as np
import paddle


class TensorStd(unittest.TestCase):
    def setUp(self):
        self.shape = [10, 10, 2, 2]
        self.unbiased = True
        self.axis = None
        self.keepdim = True
        self.use_gpu = False
        np.random.seed(123)

    def numpy_res(self, x_array):
        if self.axis is not None and isinstance(self.axis, list):
            axis = tuple(self.axis)
        elif self.axis is not None and isinstance(self.axis, int):
            axis = tuple([self.axis])
        else:
            axis = self.axis
        if self.unbiased:
            var_num = x_array.var(axis=axis, keepdims=self.keepdim)
            shape = x_array.shape
            reduce_dims = []
            if axis is None:
                reduce_dims = shape
            else:
                reduce_dims = [shape[i] for i in axis]
            for i in range(len(reduce_dims)):
                if reduce_dims[i] < 0:
                    reduce_dims[i] = len(shape) + reduce_dims[i]
            count = 1
            for dim in reduce_dims:
                count *= dim
            np_res = np.sqrt(var_num * count / (count - 1))
        else:
            np_res = np.std(x_array, axis=axis, keepdims=self.keepdim)
        return np_res

    def test_main(self):
        place = fluid.CUDAPlace(
            0) if self.use_gpu and fluid.core.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        x = fluid.data(name='x', shape=self.shape, dtype='float32')
        out = tensor.std(x,
                         unbiased=self.unbiased,
                         axis=self.axis,
                         keepdim=self.keepdim)

        prog = fluid.default_main_program()
        x_array = np.random.random(self.shape).astype('float32')
        np_res = self.numpy_res(x_array)
        res = exe.run(fluid.default_main_program(),
                      fetch_list=[out],
                      feed={'x': x_array})
        res = res[0]
        self.assertEqual(np_res.shape, res.shape)
        self.assertTrue(np.allclose(np_res, res, rtol=1e-6, atol=0))

    def test_dygraph(self):
        class Net(fluid.Layer):
            def __init__(self, axis=None, unbiased=True, keepdim=False):
                super(Net, self).__init__()
                self.axis = axis
                self.unbiased = unbiased
                self.keepdim = keepdim

            def forward(self, x):
                x = tensor.std(x,
                               axis=self.axis,
                               unbiased=self.unbiased,
                               keepdim=self.keepdim)
                return x

        x_array = np.random.random(self.shape).astype('float32')
        with fluid.dygraph.guard():
            x_std = Net(unbiased=self.unbiased,
                        axis=self.axis,
                        keepdim=self.keepdim)
            dy_ret = x_std(fluid.dygraph.to_variable(x_array))
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        np_res = self.numpy_res(x_array)
        self.assertTrue(np.allclose(np_res, dy_ret_value, rtol=1e-6, atol=0))


class TensorStd1(TensorStd):
    def setUp(self):
        self.shape = [10, 10, 2, 2]
        self.unbiased = False
        self.axis = [1, 2, 3]
        self.keepdim = False
        self.use_gpu = False


class TensorStd2(TensorStd):
    def setUp(self):
        self.shape = [10, 10, 2, 2]
        self.unbiased = True
        self.axis = 0
        self.keepdim = True
        self.use_gpu = False


class TensorStd3(TensorStd):
    def setUp(self):
        self.shape = [10, 10, 2, 2]
        self.unbiased = False
        self.axis = 1
        self.keepdim = False
        self.use_gpu = False


class TensorStd4(TensorStd):
    def setUp(self):
        self.shape = [10, 10, 2, 2]
        self.unbiased = False
        self.axis = None
        self.keepdim = True
        self.use_gpu = True


if __name__ == '__main__':
    unittest.main()
