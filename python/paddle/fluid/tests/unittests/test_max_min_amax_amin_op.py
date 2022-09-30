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
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard
from op_test import OpTest

paddle.enable_static()


class TestMaxMinAmaxAminAPI(unittest.TestCase):

    def setUp(self):
        self.init_case()
        self.cal_np_out_and_gradient()
        self.place = fluid.CUDAPlace(
            0) if core.is_compiled_with_cuda() else fluid.CPUPlace()

    def init_case(self):
        self.x_np = np.array([[0.2, 0.3, 0.5, 0.9], [0.1, 0.2, 0.6, 0.7]])
        self.shape = [2, 4]
        self.dtype = 'float64'
        self.axis = 0
        self.keepdim = False

    # If there are multiple minimum or maximum elements, max/min/amax/amin is non-derivable,
    # its gradient check is not supported by unittest framework,
    # thus we calculate the gradient by numpy function.
    def cal_np_out_and_gradient(self):

        def _cal_np_out_and_gradient(func):
            if func is 'amax':
                out = np.amax(self.x_np, axis=self.axis, keepdims=self.keepdim)
            elif func is 'amin':
                out = np.amin(self.x_np, axis=self.axis, keepdims=self.keepdim)
            elif func is 'max':
                out = np.max(self.x_np, axis=self.axis, keepdims=self.keepdim)
            elif func is 'min':
                out = np.min(self.x_np, axis=self.axis, keepdims=self.keepdim)
            else:
                print('This unittest only test amax/amin/max/min, but now is',
                      func)
            self.np_out[func] = out
            grad = np.zeros(self.shape)
            out_b = np.broadcast_to(out.view(), self.shape)
            grad[self.x_np == out_b] = 1
            if func in ['amax', 'amin']:
                grad_sum = grad.sum(self.axis).reshape(out.shape)
                grad_b = np.broadcast_to(grad_sum, self.shape)
                grad /= grad_sum

            self.np_grad[func] = grad

        self.np_out = dict()
        self.np_grad = dict()
        _cal_np_out_and_gradient('amax')
        _cal_np_out_and_gradient('amin')
        _cal_np_out_and_gradient('max')
        _cal_np_out_and_gradient('min')

    def _choose_paddle_func(self, func, x):
        if func is 'amax':
            out = paddle.amax(x, self.axis, self.keepdim)
        elif func is 'amin':
            out = paddle.amin(x, self.axis, self.keepdim)
        elif func is 'max':
            out = paddle.max(x, self.axis, self.keepdim)
        elif func is 'min':
            out = paddle.min(x, self.axis, self.keepdim)
        else:
            print('This unittest only test amax/amin/max/min, but now is', func)
        return out

    # We check the output between paddle API and numpy in static graph.
    def test_static_graph(self):

        def _test_static_graph(func):
            startup_program = fluid.Program()
            train_program = fluid.Program()
            with fluid.program_guard(startup_program, train_program):
                x = fluid.data(name='input', dtype=self.dtype, shape=self.shape)
                x.stop_gradient = False
                out = self._choose_paddle_func(func, x)

                exe = fluid.Executor(self.place)
                res = exe.run(fluid.default_main_program(),
                              feed={'input': self.x_np},
                              fetch_list=[out])
                self.assertTrue((np.array(res[0]) == self.np_out[func]).all())

        _test_static_graph('amax')
        _test_static_graph('amin')
        _test_static_graph('max')
        _test_static_graph('min')

    # As dygraph is easy to compute gradient, we check the gradient between
    # paddle API and numpy in dygraph.
    def test_dygraph(self):

        def _test_dygraph(func):
            paddle.disable_static()
            x = paddle.to_tensor(self.x_np,
                                 dtype=self.dtype,
                                 stop_gradient=False)
            out = self._choose_paddle_func(func, x)
            grad_tensor = paddle.ones_like(x)
            paddle.autograd.backward([out], [grad_tensor], True)

            np.testing.assert_allclose(self.np_out[func],
                                       out.numpy(),
                                       rtol=1e-05)
            np.testing.assert_allclose(self.np_grad[func], x.grad, rtol=1e-05)
            paddle.enable_static()

        _test_dygraph('amax')
        _test_dygraph('amin')
        _test_dygraph('max')
        _test_dygraph('min')


    # test two minimum or maximum elements
class TestMaxMinAmaxAminAPI2(TestMaxMinAmaxAminAPI):

    def init_case(self):
        self.x_np = np.array([[0.2, 0.3, 0.9, 0.9], [0.1, 0.1, 0.6, 0.7]])
        self.shape = [2, 4]
        self.dtype = 'float64'
        self.axis = None
        self.keepdim = False


# test different axis
class TestMaxMinAmaxAminAPI3(TestMaxMinAmaxAminAPI):

    def init_case(self):
        self.x_np = np.array([[0.2, 0.3, 0.9, 0.9], [0.1, 0.1, 0.6, 0.7]])
        self.shape = [2, 4]
        self.dtype = 'float64'
        self.axis = 0
        self.keepdim = False


# test keepdim = True
class TestMaxMinAmaxAminAPI4(TestMaxMinAmaxAminAPI):

    def init_case(self):
        self.x_np = np.array([[0.2, 0.3, 0.9, 0.9], [0.1, 0.1, 0.6, 0.7]])
        self.shape = [2, 4]
        self.dtype = 'float64'
        self.axis = 1
        self.keepdim = True


# test axis is tuple
class TestMaxMinAmaxAminAPI5(TestMaxMinAmaxAminAPI):

    def init_case(self):
        self.x_np = np.array([[[1, 2], [3, 4]], [[5, 6], [7,
                                                          8]]]).astype(np.int32)
        self.shape = [2, 2, 2]
        self.dtype = 'int32'
        self.axis = (0, 1)
        self.keepdim = False


# test multiple minimum or maximum elements
class TestMaxMinAmaxAminAPI6(TestMaxMinAmaxAminAPI):

    def init_case(self):
        self.x_np = np.array([[0.2, 0.9, 0.9, 0.9], [0.9, 0.9, 0.2, 0.2]])
        self.shape = [2, 4]
        self.dtype = 'float64'
        self.axis = None
        self.keepdim = False


if __name__ == '__main__':
    unittest.main()
