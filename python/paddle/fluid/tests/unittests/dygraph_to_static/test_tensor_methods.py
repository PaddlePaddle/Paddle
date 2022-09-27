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

import numpy as np
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
        np.testing.assert_allclose(dygraph_res, static_res, rtol=1e-05)


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
def tensor_item(x):
    x = paddle.to_tensor(x)
    y = x.clone()
    return y.item()


class TestTensorItem(unittest.TestCase):

    def _run(self, to_static):
        prog_trans = paddle.jit.ProgramTranslator()
        prog_trans.enable(to_static)
        x = paddle.ones([1])
        if to_static:
            return tensor_item(x).numpy()
        return tensor_item(x)

    def test_tensor_clone(self):
        dygraph_res = self._run(to_static=False)
        static_res = self._run(to_static=True)
        np.testing.assert_allclose(dygraph_res, static_res)


@paddle.jit.to_static
def tensor_size(x):
    x = paddle.to_tensor(x)
    x = paddle.reshape(x, paddle.shape(x))  # dynamic shape
    y = x.size
    return y


class TestTensorSize(unittest.TestCase):

    def _run(self, to_static):
        prog_trans = paddle.jit.ProgramTranslator()
        prog_trans.enable(to_static)
        x = paddle.ones([1, 2, 3])
        if to_static == False:
            return tensor_size(x)
        return tensor_size(x).numpy()

    def test_tensor_clone(self):
        dygraph_res = self._run(to_static=False)
        static_res = self._run(to_static=True)
        np.testing.assert_allclose(dygraph_res, static_res, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
