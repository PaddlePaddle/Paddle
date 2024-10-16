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

import sys
import unittest

import numpy as np

sys.path.append("../../legacy_test")
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
    def test_allclose(self):
        # 1) x is 0D
        x = paddle.full([], 0.5)
        y = paddle.full([], 0.6)
        out = paddle.allclose(x, y)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        self.assertEqual(res[0].shape, ())
        self.assertFalse(res[0])

        # 2) x is ND
        x = paddle.full([2, 3], 0.5)
        y = paddle.full([2, 3], 0.6)
        out = paddle.allclose(x, y)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        self.assertEqual(res[0].shape, ())
        self.assertFalse(res[0])

    @prog_scope()
    def test_equal_all(self):
        # 1) x is 0D
        x = paddle.full([], 0.5)
        y = paddle.full([], 0.6)
        out = paddle.equal_all(x, y)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        self.assertEqual(res[0].shape, ())
        self.assertFalse(res[0])

        # 2) x is ND
        x = paddle.full([2, 3], 0.5)
        y = paddle.full([2, 3], 0.6)
        out = paddle.equal_all(x, y)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        self.assertEqual(res[0].shape, ())
        self.assertFalse(res[0])

    @prog_scope()
    def test_where(self):
        x1 = paddle.full([], 1, 'float32')
        x2 = paddle.full([], 2, 'float32')
        x1.stop_gradient = False
        x2.stop_gradient = False
        out = paddle.where(x1 > x2, x1, x2)
        loss = paddle.mean(out)
        grad_list = paddle.static.append_backward(
            loss, parameter_list=[out, x1, x2]
        )
        (_, out_grad), (_, x1_grad), (_, x2_grad) = grad_list

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            feed={},
            fetch_list=[out, out_grad, x1_grad, x2_grad],
        )

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], 2)
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[2], 0)
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[3], 1)

    @prog_scope()
    def test_atan2(self):
        x1 = paddle.full([], 0, 'float32')
        x2 = paddle.full([], 2, 'float32')
        x1.stop_gradient = False
        x2.stop_gradient = False
        out = paddle.atan2(x1, x2)
        paddle.static.append_backward(out)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, feed={}, fetch_list=[out])

        self.assertEqual(res[0].shape, ())

    @prog_scope()
    def test_interpolate(self):
        from paddle.nn.functional import interpolate

        input_x = paddle.rand([2, 3, 6, 6])
        input_x.stop_gradient = False

        output_size = [
            paddle.full([], 12, dtype="int32"),
            paddle.full([], 12, dtype="int32"),
        ]

        out1 = interpolate(
            x=input_x, size=output_size, mode="bilinear", align_corners=False
        )
        _, input_x_grad = paddle.static.append_backward(
            out1.sum(), parameter_list=[input_x]
        )[0]
        prog = paddle.static.default_main_program()
        res1 = self.exe.run(prog, feed={}, fetch_list=[out1, input_x_grad])

        scale_1 = paddle.full([], 2)
        out2 = interpolate(
            x=input_x,
            scale_factor=scale_1,
            mode="bilinear",
            align_corners=False,
        )
        _, input_x_grad = paddle.static.append_backward(
            out2.sum(), parameter_list=[input_x]
        )[0]
        prog = paddle.static.default_main_program()
        res2 = self.exe.run(prog, feed={}, fetch_list=[out2, input_x_grad])

        self.assertEqual(res1[0].shape, (2, 3, 12, 12))
        self.assertEqual(res1[1].shape, (2, 3, 6, 6))
        self.assertEqual(res2[0].shape, (2, 3, 12, 12))
        self.assertEqual(res2[1].shape, (2, 3, 6, 6))

    @prog_scope()
    def test_upsample(self):
        from paddle.nn.functional import upsample

        input_x = paddle.rand([2, 3, 6, 6])
        input_x.stop_gradient = False

        output_size = [
            paddle.full([], 12, dtype="int32"),
            paddle.full([], 12, dtype="int32"),
        ]

        out1 = upsample(
            x=input_x, size=output_size, mode="bilinear", align_corners=False
        )
        _, input_x_grad = paddle.static.append_backward(
            out1.sum(), parameter_list=[input_x]
        )[0]
        prog = paddle.static.default_main_program()
        res1 = self.exe.run(prog, feed={}, fetch_list=[out1, input_x_grad])

        self.assertEqual(res1[0].shape, (2, 3, 12, 12))
        self.assertEqual(res1[1].shape, (2, 3, 6, 6))

    @prog_scope()
    def test_unstack(self):
        x1 = paddle.full([1], 0, 'float32')
        x1.stop_gradient = False
        out1 = paddle.unstack(x1, 0)
        out1 = paddle.add_n(out1)
        _, x1_grad = paddle.static.append_backward(out1, parameter_list=[x1])[0]
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, feed={}, fetch_list=[out1, x1_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (1,))

        x2 = paddle.full([2], 2, 'float32')
        x2.stop_gradient = False
        out2 = paddle.unstack(x2, 0)
        out2_sum = paddle.add_n(out2)
        _, x2_grad = paddle.static.append_backward(
            out2_sum, parameter_list=[x2]
        )[0]
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, feed={}, fetch_list=[out2_sum, x2_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (2,))

    @prog_scope()
    def test_unbind(self):
        x1 = paddle.full([1], 0, 'float32')
        x1.stop_gradient = False
        out1 = paddle.unbind(x1, 0)
        out1 = paddle.add_n(out1)
        _, x1_grad = paddle.static.append_backward(out1, parameter_list=[x1])[0]
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, feed={}, fetch_list=[out1, x1_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (1,))

        x2 = paddle.full([2], 2, 'float32')
        x2.stop_gradient = False
        out2 = paddle.unbind(x2, 0)
        out2_sum = paddle.add_n(out2)
        _, x2_grad = paddle.static.append_backward(
            out2_sum, parameter_list=[x2]
        )[0]
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, feed={}, fetch_list=[out2_sum, x2_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (2,))

    @prog_scope()
    def test_masked_select(self):
        x = paddle.rand([])
        x.stop_gradient = False
        mask = paddle.full([], True, dtype='bool')
        y = paddle.masked_select(x, mask)
        grad_list = paddle.static.append_backward(
            y.sum(), parameter_list=[y, x]
        )
        (_, y_grad), (_, x_grad) = grad_list

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x, y, y_grad, x_grad])
        self.assertEqual(res[1].shape, (1,))
        self.assertEqual(res[1], res[0])
        self.assertEqual(res[2].shape, (1,))
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[3], 1)

    @prog_scope()
    def test_squeeze(self):
        x1 = paddle.full([], 2)
        x1.stop_gradient = False
        out1 = paddle.squeeze(x1, axis=0)
        _, x1_grad = paddle.static.append_backward(
            out1.sum(), parameter_list=[x1]
        )[0]

        x2 = paddle.full([], 3)
        x3 = paddle.full([], 0, dtype='int32')
        x2.stop_gradient = False
        out2 = paddle.squeeze(x2, axis=x3)
        _, x2_grad = paddle.static.append_backward(
            out2.sum(), parameter_list=[x2]
        )[0]

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                x1_grad,
                x2_grad,
            ],
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())

    @prog_scope()
    def test_unsqueeze(self):
        x1 = paddle.full([], 2)
        x1.stop_gradient = False
        out1 = paddle.unsqueeze(x1, axis=0)
        _, x1_grad = paddle.static.append_backward(
            out1.sum(), parameter_list=[x1]
        )[0]

        x2 = paddle.full([], 3)
        x3 = paddle.full([], 0, dtype='int32')
        x2.stop_gradient = False
        out2 = paddle.unsqueeze(x2, axis=x3)
        _, x2_grad = paddle.static.append_backward(
            out2.sum(), parameter_list=[x2]
        )[0]

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                x1_grad,
                x2_grad,
            ],
        )
        self.assertEqual(res[0].shape, (1,))
        self.assertEqual(res[1].shape, (1,))
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())

    @prog_scope()
    def test_t(self):
        x = paddle.full([], 2.0)
        x.stop_gradient = False
        out = paddle.t(x)
        grad_list = paddle.static.append_backward(out, parameter_list=[out, x])

        prog = paddle.static.default_main_program()
        if paddle.framework.in_pir_mode():
            res = self.exe.run(
                prog,
                feed={},
                fetch_list=[out, grad_list[0][1], grad_list[1][1]],
            )
        else:
            res = self.exe.run(
                prog, feed={}, fetch_list=[out, out.grad_name, x.grad_name]
            )

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())

    @prog_scope()
    def test_static_data(self):
        x1 = paddle.static.data(name="x1", shape=[])
        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            feed={
                "x1": np.array(1.0, dtype='float32'),
            },
            fetch_list=[x1],
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], np.array(1.0))

        x2 = paddle.static.data(name="x2", shape=[])
        x3 = paddle.static.data(name="x3", shape=[])
        y = x2 + x3
        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            feed={
                "x1": np.array(1.0, dtype='float32'),
                "x2": 100.5,
                "x3": 200.5,
            },
            fetch_list=[y],
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], 301.0)

    @prog_scope()
    def test_prelu(self):
        x1 = paddle.full([], 1.0, 'float32')
        x1.stop_gradient = False
        w1 = paddle.to_tensor([0.25], dtype='float32')
        out1 = paddle.nn.functional.prelu(x1, w1)
        (_, out1_grad), (_, x1_grad) = paddle.static.append_backward(
            out1.sum(), parameter_list=[out1, x1]
        )

        x2 = paddle.full([], 1.0, 'float32')
        x2.stop_gradient = False
        w2 = paddle.full([], 0.25, dtype='float32')
        out2 = paddle.nn.functional.prelu(x2, w2)
        (_, out2_grad), (_, x2_grad) = paddle.static.append_backward(
            out2.sum(), parameter_list=[out2, x2]
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                x1_grad,
                x2_grad,
                out1_grad,
                out2_grad,
            ],
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[4].shape, ())
        self.assertEqual(res[5].shape, ())

    @prog_scope()
    def test_while_loop(self):
        def cond(i, x):
            return paddle.less_than(i, eleven)

        def body(i, x):
            x = x + i
            i = i + 1
            return [i, x]

        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, paddle.static.Program()):
            i = paddle.static.data(name='i', shape=[], dtype='float32')
            i.stop_gradient = False
            i.persistable = True
            eleven = paddle.full([], 11, 'float32')
            x = paddle.static.data(name='x', shape=[], dtype='float32')
            x.stop_gradient = False
            x.persistable = True
            out_i, out_x = paddle.static.nn.while_loop(cond, body, [i, x])
            grad_list = paddle.static.append_backward(out_x)

        feed = {
            'i': np.array(1.0, dtype='float32'),
            'x': np.array(0.0, dtype='float32'),
        }
        if paddle.framework.in_pir_mode():
            fetch_list = [out_i, out_x]
            for _, g in grad_list:
                fetch_list.append(g)
            res = self.exe.run(
                main_program,
                feed=feed,
                fetch_list=fetch_list,
            )
        else:
            res = self.exe.run(
                main_program,
                feed=feed,
                fetch_list=[out_i.name, out_x.name, i.grad_name, x.grad_name],
            )

        self.assertEqual(res[0].shape, ())
        np.testing.assert_allclose(res[0], np.array(11))
        self.assertEqual(res[1].shape, ())
        np.testing.assert_allclose(res[1], np.array(55))
        self.assertEqual(res[2].shape, ())
        np.testing.assert_allclose(res[2], np.array(10))
        self.assertEqual(res[3].shape, ())
        np.testing.assert_allclose(res[3], np.array(1.0))


if __name__ == "__main__":
    unittest.main()
