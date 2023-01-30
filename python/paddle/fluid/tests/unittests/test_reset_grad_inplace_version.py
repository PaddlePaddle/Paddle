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
<<<<<<< HEAD
import unittest

import numpy as np

import paddle
from paddle import _legacy_C_ops
=======
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle import _C_ops, _legacy_C_ops
from paddle.fluid import framework
from paddle.fluid.framework import _test_eager_guard
import unittest
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

paddle.set_device('cpu')


# Test 1
def clear_grad_test_0(w, a):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    @paddle.no_grad()
    def warp(*_):
        assert w.grad is not None
        _legacy_C_ops.scale_(w.grad, 'scale', 0.5)
        w._reset_grad_inplace_version(True)

    return warp


class TestInplaceAndClearGradient(unittest.TestCase):
<<<<<<< HEAD
    def test_inplace_n_clear_grad(self):
=======

    def func_test(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        input_data = np.ones([1, 1])
        w = paddle.to_tensor(input_data, 'float32', stop_gradient=False)

        _clear_grad = clear_grad_test_0(w, a="1")
        w._register_backward_hook(_clear_grad)
        for i in range(2):
            print(" Step: ", i)
            out0 = _legacy_C_ops.scale(w, 'scale', 0.1)
<<<<<<< HEAD
            out = _legacy_C_ops.matmul_v2(
                out0, w, 'trans_x', False, 'trans_y', False
            )
            out.backward()
        assert w.grad[0] == 0.15


# Test 2
class Counter:
=======
            out = _legacy_C_ops.matmul_v2(out0, w, 'trans_x', False, 'trans_y',
                                          False)
            out.backward()
        assert w.grad[0] == 0.15

    def test(self):
        with _test_eager_guard():
            self.func_test()
        self.func_test()


# Test 2
class Counter:

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self):
        self.num_calls = 0
        self.step = 0


def clear_grad_test_1(w, c):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    @paddle.no_grad()
    def warp(*_):
        assert w.grad is not None
        if c.step == 1:
            w.grad.scale_(scale=0.5)
            w._reset_grad_inplace_version(True)

        c.num_calls += 1

    return warp


class TestInplaceClearGradAccumulation(unittest.TestCase):
<<<<<<< HEAD
    def test_inplace_clear_grad_accum(self):
=======

    def func_test(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        input_data = np.ones([1, 1])
        w = paddle.to_tensor(input_data, 'float32', stop_gradient=False)
        c = Counter()

        _clear_grad = clear_grad_test_1(w, c)
        w._register_backward_hook(_clear_grad)

        for c.step in range(5):
            out0 = _legacy_C_ops.scale(w, 'scale', 0.1)
<<<<<<< HEAD
            out = _legacy_C_ops.matmul_v2(
                out0, w, 'trans_x', False, 'trans_y', False
            )
=======
            out = _legacy_C_ops.matmul_v2(out0, w, 'trans_x', False, 'trans_y',
                                          False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            out.backward()

            if c.step == 1:
                w.clear_gradient(False)

            assert c.num_calls == 1
            c.num_calls = 0

<<<<<<< HEAD

class TestInplaceClearGradAccumulationAlt(unittest.TestCase):
    def test_inplace_clear_grad_accum(self):
=======
    def test(self):
        with _test_eager_guard():
            self.func_test()
        self.func_test()


class TestInplaceClearGradAccumulationAlt(unittest.TestCase):

    def func_test(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        input_data = np.ones([1, 1])
        w = paddle.to_tensor(input_data, 'float32', stop_gradient=False)
        out = _legacy_C_ops.scale(w, 'scale', 0.1)
        out.backward()

        w.grad.scale_(scale=0.5)
        w._reset_grad_inplace_version(False)

        assert w.grad._inplace_version() == 1

<<<<<<< HEAD
=======
    def test(self):
        with _test_eager_guard():
            self.func_test()
        self.func_test()

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

if __name__ == '__main__':
    unittest.main()
