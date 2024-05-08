#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# Note:
# 0D Tensor indicates that the tensor's dimension is 0
# 0D Tensor's shape is always [], numel is 1
# which can be created by paddle.rand([])

import unittest

import numpy as np
from decorator_helper import prog_scope

import paddle

# Use to test zero-dim of Sundry API, which is unique and can not be classified
# with others. It can be implemented here flexibly.


class TestSundryAPIStatic(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.exe = paddle.static.Executor()

    def assertShapeEqual(self, out, target_tuple):
        if not paddle.framework.in_pir_mode():
            out_shape = list(out.shape)
        else:
            out_shape = out.shape
        self.assertEqual(out_shape, target_tuple)

    @prog_scope()
    def test_create_global_var(self):
        zero_dim_var = paddle.static.create_global_var(
            shape=[], value=0.5, dtype='float32'
        )
        self.assertEqual(zero_dim_var.shape, ())
        prog = paddle.static.default_startup_program()
        res = self.exe.run(prog, fetch_list=[zero_dim_var])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], 0.5)

    @prog_scope()
    def test_setitem(self):
        # NOTE(zoooo0820): __setitem__ has gradient problem in static graph.
        # To solve this, we may not support __setitem__ in static graph.
        # These unit tests will delete soon.

        # case1: all axis have a scalar indice
        x = paddle.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5))
        x.stop_gradient = False
        out = x * 2
        out = paddle.static.setitem(out, (1, 2, 3, 4), 10)
        paddle.static.append_backward(out.sum())
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x.grad_name])

        self.assertEqual(out.shape, x.shape)
        np.testing.assert_allclose(res[0][1, 2, 3, 4], np.array(10))
        self.assertEqual(res[1].shape, (2, 3, 4, 5))
        x_grad_expected = np.ones((2, 3, 4, 5)) * 2
        x_grad_expected[1, 2, 3, 4] = 0
        np.testing.assert_allclose(res[1], x_grad_expected)

        # case2: 0-D Tensor indice in some axis
        # NOTE(zoooo0820): Now, int/slice with 0-D Tensor will still be
        # treated as combined indexing, which is not support backward.
        # There should have more test cases such as out[1, indice, :] = 0.5 when this
        # problem is fixed.
        x = paddle.randn((2, 3, 4, 5))
        x.stop_gradient = False
        indice = paddle.full([], 1, dtype='int32')
        out = x * 1
        out = paddle.static.setitem(out, (indice, indice), 0.5)
        paddle.static.append_backward(out.sum())
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x.grad_name])

        self.assertEqual(out.shape, x.shape)
        np.testing.assert_allclose(res[0][1, 1], np.ones((4, 5)) * 0.5)
        x_grad_expected = np.ones((2, 3, 4, 5))
        x_grad_expected[1, 1] = 0
        np.testing.assert_allclose(res[1], x_grad_expected)

        # case3：0-D Tensor indice in some axis, value is a Tensor
        # and there is broadcast
        x = paddle.randn((2, 3, 4, 5))
        x.stop_gradient = False
        v = paddle.ones((4, 5), dtype='float32') * 5
        v.stop_gradient = False
        indice = paddle.full([], 1, dtype='int32')
        out = x * 1
        out = paddle.static.setitem(out, indice, v)
        paddle.static.append_backward(out.sum())
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x.grad_name, v.grad_name])

        self.assertEqual(out.shape, x.shape)
        np.testing.assert_allclose(res[0][1], np.ones((3, 4, 5)) * 5)
        x_grad_expected = np.ones((2, 3, 4, 5))
        x_grad_expected[1] = 0
        np.testing.assert_allclose(res[1], x_grad_expected)

    @prog_scope()
    def test_static_auc(self):
        x = paddle.full(shape=[3, 2], fill_value=0.25)
        y = paddle.full(shape=[3], fill_value=1, dtype="int64")
        out = paddle.static.auc(input=x, label=y)[0]

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[out],
        )

        self.assertEqual(res[0].shape, ())

    @prog_scope()
    def test_static_nn_prelu(self):
        x1 = paddle.full([], 1.0, 'float32')
        x1.stop_gradient = False
        out1 = paddle.static.nn.prelu(x1, 'all')
        grad_list = paddle.static.append_backward(
            out1.sum(), parameter_list=[x1, out1]
        )
        (_, x1_grad), (_, out1_grad) = grad_list

        prog = paddle.static.default_main_program()
        self.exe.run(paddle.static.default_startup_program())
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                x1_grad,
                out1_grad,
            ],
        )

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        np.testing.assert_allclose(res[0], np.array(1))
        np.testing.assert_allclose(res[1], np.array(1))


if __name__ == "__main__":
    unittest.main()
