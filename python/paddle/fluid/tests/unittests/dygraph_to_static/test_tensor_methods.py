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

from __future__ import print_function

import numpy
import paddle
import unittest


@paddle.jit.to_static
def tensor_clone(x):
    x = paddle.to_tensor(x)
    y = x.clone()
    return y


class TestTensorClone(unittest.TestCase):

    def _run(self, to_static):
        prog_trans = paddle.jit.ProgramTranslator()
        prog_trans.enable(to_static)
        x = paddle.ones([1, 2, 3])
        return tensor_clone(x).numpy()

    def test_tensor_clone(self):
        dygraph_res = self._run(to_static=False)
        static_res = self._run(to_static=True)
        self.assertTrue(numpy.allclose(dygraph_res, static_res),
                        msg='dygraph res is {}\nstatic_res is {}'.format(
                            dygraph_res, static_res))


@paddle.jit.to_static
def tensor_numpy(x):
    x = paddle.to_tensor(x)
    x.clear_gradient()
    return x


class TestTensorDygraphOnlyMethodError(unittest.TestCase):

    def _run(self, to_static):
        prog_trans = paddle.jit.ProgramTranslator()
        prog_trans.enable(to_static)
        x = paddle.zeros([2, 2])
        y = tensor_numpy(x)
        return y.numpy()

    def test_to_static_numpy_report_error(self):
        dygraph_res = self._run(to_static=False)
        with self.assertRaises(AssertionError):
            static_res = self._run(to_static=True)


@paddle.jit.to_static
def tensor_badreturn_0(x):
    a = paddle.to_tensor([1.0, 2.0, 3.0], dtype="int64")
    return a

@paddle.jit.to_static
def tensor_badreturn_1(x):
    paddle.set_default_dtype("float64")
    a = paddle.to_tensor([1.0, 2.0, 3.0])
    return a 


class TestToTensorReturnVal(unittest.TestCase):

    def _run(self, to_static):
        prog_trans = paddle.jit.ProgramTranslator()
        prog_trans.enable(to_static)
        x = paddle.to_tensor([3])
        out0 = tensor_badreturn_0(x)
        out1 = tensor_badreturn_1(x)
        return out0, out1
    
    def test_to_tensor_badreturn(self):
        dygraph_res = self._run(to_static=False)
        x = paddle.to_tensor([3])
        self.assertTrue(dygraph_res[0].dtype == tensor_badreturn_0(x).dtype,
                        msg='to_static dtype is {}, orig dtype is {}'.format(
                            dygraph_res[0].dtype, tensor_badreturn_0(x).dtype))
        self.assertTrue(dygraph_res[1].dtype == tensor_badreturn_1(x).dtype,
                        msg='to_static dtype is {}, orig dtype is {}'.format(
                            dygraph_res[0].dtype, tensor_badreturn_1(x).dtype))
        




if __name__ == '__main__':
    unittest.main()
