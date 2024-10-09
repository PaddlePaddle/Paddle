#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
    def test_numel(self):
        # 1) x is 0D
        x = paddle.full([], 0.5)
        out = paddle.numel(x)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        self.assertEqual(res[0].shape, ())
        np.testing.assert_array_equal(res[0], np.array(1))

        # 2) x is ND
        x = paddle.full([3, 5], 0.5)
        out = paddle.numel(x)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        self.assertEqual(res[0].shape, ())
        np.testing.assert_array_equal(res[0], np.array(15))

    @prog_scope()
    def test_rank(self):
        # 1) x is 0D
        x = paddle.full([], 0.5)
        out = paddle.rank(x)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        self.assertEqual(res[0].shape, ())
        np.testing.assert_array_equal(res[0], np.array(0))

        # 1) x is ND
        x = paddle.full([3, 5], 0.5)
        out = paddle.rank(x)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        self.assertEqual(res[0].shape, ())
        np.testing.assert_array_equal(res[0], np.array(2))

    @prog_scope()
    def test_shape(self):
        x = paddle.full([], 0.5)
        out = paddle.shape(x)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        np.testing.assert_array_equal(res[0], np.array([]))
        self.assertEqual(res[0].shape, (0,))

    def test_broadcast_tensors(self):
        # 1) x is 0D, y is 0D
        x1 = paddle.full([], 2.0)
        x1.stop_gradient = False
        x2 = paddle.full([], 2.0)
        x2.stop_gradient = False
        out1, out2 = paddle.broadcast_tensors([x1, x2])

        self.assertShapeEqual(out1, [])
        self.assertShapeEqual(out2, [])

        # 2) x is ND , y is 0D
        x1 = paddle.full([2, 3], 2.0)
        x1.stop_gradient = False
        x2 = paddle.full([], 2.0)
        x2.stop_gradient = False
        out1, out2 = paddle.broadcast_tensors([x1, x2])

        self.assertShapeEqual(out1, [2, 3])
        self.assertShapeEqual(out2, [2, 3])

        # 3) x is 0D , y is ND
        x1 = paddle.full([], 2.0)
        x1.stop_gradient = False
        x2 = paddle.full([2, 3], 2.0)
        x2.stop_gradient = False
        out1, out2 = paddle.broadcast_tensors([x1, x2])

        self.assertShapeEqual(out1, [2, 3])
        self.assertShapeEqual(out2, [2, 3])

    @prog_scope()
    def test_to_tensor(self):
        out1 = paddle.to_tensor(1)
        out2 = paddle.to_tensor(2.5)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out1, out2])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], 1)
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[1], 2.5)

    @prog_scope()
    def test_matmul(self):
        # 1) no transpose
        x = paddle.randn([10])
        x.stop_gradient = False
        y = paddle.randn([10])
        y.stop_gradient = False
        out = paddle.matmul(x, y)
        grad_list = paddle.static.append_backward(out, parameter_list=[x, y])
        (_, x_grad), (_, y_grad) = grad_list

        self.assertShapeEqual(out, [])

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x_grad, y_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (10,))
        self.assertEqual(res[2].shape, (10,))

        # 2) transpose x and y
        x = paddle.randn([10])
        x.stop_gradient = False
        y = paddle.randn([10])
        y.stop_gradient = False
        out = paddle.matmul(x, y, True, True)
        grad_list = paddle.static.append_backward(out, parameter_list=[x, y])
        (_, x_grad), (_, y_grad) = grad_list

        self.assertShapeEqual(out, [])

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x_grad, y_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (10,))
        self.assertEqual(res[2].shape, (10,))

    @prog_scope()
    def test_linalg_slogdet(self):
        # 2-D input
        x = paddle.randn([3, 3])
        x.stop_gradient = False
        out = paddle.linalg.slogdet(x)
        _, x_grad = paddle.static.append_backward(
            out.sum(), parameter_list=[x]
        )[0]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x_grad])
        self.assertEqual(res[0].shape, (2,))
        self.assertEqual(res[1].shape, (3, 3))

        # 3-D input
        x1 = paddle.randn([3, 3, 3])
        x1.stop_gradient = False
        out1 = paddle.linalg.slogdet(x1)
        _, x1_grad = paddle.static.append_backward(
            out1.sum(), parameter_list=[x1]
        )[0]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out1, x1_grad])
        self.assertEqual(res[0].shape, (2, 3))
        self.assertEqual(res[1].shape, (3, 3, 3))

    @prog_scope()
    def test_multi_dot(self):
        a = paddle.randn([4])
        a.stop_gradient = False
        b = paddle.randn([4, 5])
        b.stop_gradient = False
        c = paddle.randn([5])
        c.stop_gradient = False

        out = paddle.linalg.multi_dot([a, b, c])
        grad_list = paddle.static.append_backward(
            out.sum(), parameter_list=[a, b, c]
        )
        (_, a_grad), (_, b_grad), (_, c_grad) = grad_list
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, a_grad, b_grad, c_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (4,))
        self.assertEqual(res[2].shape, (4, 5))
        self.assertEqual(res[3].shape, (5,))

    @prog_scope()
    def test_cov(self):
        xt_1 = paddle.randn((12,))
        xt_1.stop_gradient = False
        out = paddle.linalg.cov(xt_1)
        _, xt_1_grad = paddle.static.append_backward(
            out, parameter_list=[xt_1]
        )[0]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, xt_1_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (12,))

    @prog_scope()
    def test_corrcoef(self):
        x = paddle.randn((12,))
        x.stop_gradient = False
        out = paddle.linalg.corrcoef(x)
        _, x_grad = paddle.static.append_backward(out, parameter_list=[x])[0]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (12,))

    @prog_scope()
    def test_det(self):
        xt_1 = paddle.randn((3, 3))
        xt_1.stop_gradient = False

        out = paddle.linalg.det(xt_1)
        _, xt_1_grad = paddle.static.append_backward(
            out.sum(), parameter_list=[xt_1]
        )[0]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, xt_1_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (3, 3))

    @prog_scope()
    def test_dist(self):
        x = paddle.to_tensor([[3, 3], [3, 3]], dtype="float32")
        y = paddle.to_tensor([[3, 3], [3, 1]], dtype="float32")
        x.stop_gradient = False
        y.stop_gradient = False
        out = paddle.dist(x, y)
        (_, x_grad), (_, y_grad) = paddle.static.append_backward(
            out, parameter_list=[x, y]
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x_grad, y_grad])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (2, 2))
        self.assertEqual(res[2].shape, (2, 2))
        np.testing.assert_array_equal(res[0], np.array(2).astype(np.float32))

    @prog_scope()
    def test_linalg_norm1(self):
        # 1D input, p = fro ,axis = None, using reduceInferMeta
        x_1 = paddle.arange(24, dtype="float32") - 12
        x_1.stop_gradient = False
        out_1 = paddle.linalg.norm(x_1)
        grad_list = paddle.static.append_backward(out_1, parameter_list=[x_1])
        ((_, x_1_grad),) = grad_list

        prog = paddle.static.default_main_program()

        res = self.exe.run(prog, fetch_list=[out_1, x_1_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (24,))

    @prog_scope()
    def test_linalg_norm2(self):
        # 1D input, p = 1 ,axis = None,
        # using p_norm, as_vector = True
        x_2 = paddle.arange(24, dtype="float32") - 12
        x_2.stop_gradient = False
        out_2 = paddle.linalg.norm(x_2, p=1)
        ((_, x_2_grad),) = paddle.static.append_backward(
            out_2.sum(), parameter_list=[x_2]
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_2, x_2_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (24,))

    @prog_scope()
    def test_linalg_norm3(self):
        # 1D input, p = 1 ,axis = 0,
        # using p_norm, as_vector = False
        x_2_p = paddle.arange(24, dtype="float32") - 12
        x_2_p.stop_gradient = False
        out_2_p = paddle.linalg.norm(x_2_p, p=1, axis=0)
        ((_, x_2_p_grad),) = paddle.static.append_backward(
            out_2_p.sum(), parameter_list=[x_2_p]
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_2_p, x_2_p_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (24,))

    @prog_scope()
    def test_linalg_norm4(self):
        # 1D input, p = fro ,axis = 0,
        # using p_norm, as_vector = False
        x_2_fro = paddle.arange(24, dtype="float32") - 12
        x_2_fro.stop_gradient = False
        out_2_fro = paddle.linalg.norm(x_2_fro, p="fro", axis=0)
        ((_, x_2_fro_grad),) = paddle.static.append_backward(
            out_2_fro.sum(), parameter_list=[x_2_fro]
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_2_fro, x_2_fro_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (24,))

    @prog_scope()
    def test_linalg_norm5(self):
        # 2D input, p = 1, axis = [0, 1]
        # using p_matrix_norm, depends on paddle.sum
        x_3 = paddle.arange(24, dtype="float32").reshape([4, 6])
        x_3.stop_gradient = False
        out_3 = paddle.linalg.norm(x_3, p=1, axis=[0, 1])
        ((_, x_3_grad),) = paddle.static.append_backward(
            out_3.sum(), parameter_list=[x_3]
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_3, x_3_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (4, 6))

    @prog_scope()
    def test_linalg_norm6(self):
        # 2D input, p = 1, axis = None
        # using p_matrix_norm, depends on paddle.sum
        x_4 = paddle.arange(24, dtype="float32").reshape([4, 6])
        x_4.stop_gradient = False
        out_4 = paddle.linalg.norm(x_4)
        ((_, x_4_grad),) = paddle.static.append_backward(
            out_4.sum(), parameter_list=[x_4]
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_4, x_4_grad])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (4, 6))

    @prog_scope()
    def test_linalg_norm7(self):
        # 2D input, p = inf, axis = None
        x_5 = paddle.arange(24, dtype="float32").reshape([4, 6])
        x_5.stop_gradient = False
        out_5 = paddle.linalg.norm(x_5)
        ((_, x_5_grad),) = paddle.static.append_backward(
            out_5.sum(), parameter_list=[x_5]
        )
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_5, x_5_grad])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (4, 6))

    @prog_scope()
    def test_linalg_norm8(self):
        # 2D input, p = -inf, axis = [0, 1]
        x_6 = paddle.arange(24, dtype="float32").reshape([4, 6])
        x_6.stop_gradient = False
        out_6 = paddle.linalg.norm(x_6, p=-float("inf"), axis=[0, 1])
        ((_, x_6_grad),) = paddle.static.append_backward(
            out_6.sum(), parameter_list=[x_6]
        )
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_6, x_6_grad])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (4, 6))

    @prog_scope()
    def test_linalg_cond(self):
        # use paddle.sum
        x = paddle.to_tensor([[1.0, 0, -1], [0, 1, 0], [1, 0, 1]])
        x.stop_gradient = False
        out = paddle.linalg.cond(x)
        _, x_grad = paddle.static.append_backward(out, parameter_list=[x])[0]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (3, 3))

        # p = fro : use paddle.sum
        x2 = paddle.to_tensor([[1.0, 0, -1], [0, 1, 0], [1, 0, 1]])
        x2.stop_gradient = False
        out_fro = paddle.linalg.cond(x2, p='fro')
        grad_list = paddle.static.append_backward(out_fro, parameter_list=[x2])
        ((_, x2_grad),) = grad_list

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_fro, x2_grad])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (3, 3))

        # p = nuc : use paddle.sum
        x3 = paddle.to_tensor([[1.0, 0, -1], [0, 1, 0], [1, 0, 1]])
        x3.stop_gradient = False
        out_nuc = paddle.linalg.cond(x3, p='nuc')
        _, x3_grad = paddle.static.append_backward(
            out_nuc, parameter_list=[x3]
        )[0]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_nuc, x3_grad])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (3, 3))

        # p in (-1, 1) : use paddle.sum
        x4 = paddle.to_tensor([[1.0, 0, -1], [0, 1, 0], [1, 0, 1]])
        x4.stop_gradient = False
        out_1 = paddle.linalg.cond(x4, p=1)
        _, x4_grad = paddle.static.append_backward(out_1, parameter_list=[x4])[
            0
        ]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_1, x4_grad])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (3, 3))

        x5 = paddle.to_tensor([[1.0, 0, -1], [0, 1, 0], [1, 0, 1]])
        x5.stop_gradient = False
        out_minus_1 = paddle.linalg.cond(x5, p=-1)
        ((_, x5_grad),) = paddle.static.append_backward(
            out_minus_1, parameter_list=[x5]
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_minus_1, x5_grad])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (3, 3))

        # p in (-2, 2) depends on paddle.sum
        x6 = paddle.to_tensor([[1.0, 0, -1], [0, 1, 0], [1, 0, 1]])
        x6.stop_gradient = False
        out_2 = paddle.linalg.cond(x6, p=2)
        ((_, x6_grad),) = paddle.static.append_backward(
            out_2, parameter_list=[x6]
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_2, x6_grad])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (3, 3))

        # p in (-inf, inf):use paddle.sum
        x8 = paddle.to_tensor([[1.0, 0, -1], [0, 1, 0], [1, 0, 1]])
        x8.stop_gradient = False
        out_inf = paddle.linalg.cond(x8, p=float("inf"))
        ((_, x8_grad),) = paddle.static.append_backward(
            out_inf, parameter_list=[x8]
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_inf, x8_grad])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (3, 3))

        # depends on paddle.sum
        a = paddle.randn([2, 4, 4])
        a.stop_gradient = False
        a_cond_fro = paddle.linalg.cond(a, p='fro')
        ((_, a_grad),) = paddle.static.append_backward(
            a_cond_fro.sum(), parameter_list=[a]
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[a_cond_fro, a_grad])

        self.assertEqual(res[0].shape, (2,))
        self.assertEqual(res[1].shape, (2, 4, 4))

    @prog_scope()
    def test_trace(self):
        x = paddle.to_tensor([[3, 2], [1, 9]], dtype="float32")
        x.stop_gradient = False
        out = paddle.trace(x)
        _, x_grad = paddle.static.append_backward(out, parameter_list=[x])[0]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x_grad])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (2, 2))
        np.testing.assert_allclose(res[0], np.array(12))


if __name__ == "__main__":
    unittest.main()
