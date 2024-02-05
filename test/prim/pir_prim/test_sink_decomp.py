# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn.functional as F
from paddle.autograd.ir_backward import grad
from paddle.base import core
from paddle.decomposition import decompose

paddle.enable_static()


class TestPrimMode(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [8, 16, 32, 64]
        self.shape_y = [8, 16, 32, 64]
        self.x = np.random.random(self.shape_x).astype("float32")
        self.y = np.random.random(self.shape_y).astype("float32")
        self.prog = None

    def base_net(self, flag=None):
        if flag == "forward":
            core._set_prim_forward_enabled(True)
        elif flag == "backward":
            core._set_prim_backward_enabled(True)
        elif flag == "all":
            core._set_prim_all_enabled(True)
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.static.data('x', self.shape_x, dtype='float32')
            y = paddle.static.data('y', self.shape_y, dtype='float32')
            x.stop_gradient = False
            y.stop_gradient = False
            divide_out = paddle.divide(x, y)
            sum_out = paddle.mean(divide_out, axis=0)
            [new_out] = decompose(main_program, [sum_out])
            gradients = grad(new_out, (x, y))

            exe = paddle.static.Executor()
            [fwd, dx, dy] = exe.run(
                feed={'x': self.x, 'y': self.y}, fetch_list=[new_out, gradients]
            )

        whole_ops = [op.name() for op in main_program.global_block().ops]
        self.prog = main_program
        if flag == "forward":
            core._set_prim_forward_enabled(False)
            assert (
                'pd_op.mean' not in whole_ops
                and 'pd_op.divide_grad' in whole_ops
            )
        elif flag == "backward":
            core._set_prim_backward_enabled(False)
            assert (
                'pd_op.mean' in whole_ops
                and 'pd_op.divide_grad' not in whole_ops
            )
        elif flag == "all":
            core._set_prim_all_enabled(False)
            assert (
                'pd_op.mean' not in whole_ops
                and 'pd_op.divide_grad' not in whole_ops
            )
        else:
            assert (
                'pd_op.mean' in whole_ops and 'pd_op.divide_grad' in whole_ops
            )
        return fwd, dx, dy

    def test_prim_forward(self):
        res_ref = self.base_net()
        res = self.base_net("forward")
        for ref, actual in zip(res_ref, res):
            np.testing.assert_equal(ref, actual)

    def test_prim_backward(self):
        res_ref = self.base_net()
        res = self.base_net("backward")
        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(ref, actual, rtol=1e-6)

    def test_prim_all(self):
        res_ref = self.base_net()
        res = self.base_net("all")
        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(ref, actual, rtol=1e-6)

    def test_has_decomp(self):
        _ = self.base_net()
        for op in self.prog.global_block().ops:
            if op.name() == "pd_op.divide":
                self.assertEqual(core.has_decomp(op), False)
            if op.name() == "pd_op.mean":
                self.assertEqual(core.has_decomp(op), True)


class TestReluSink(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [8, 16, 32, 64]
        self.x = np.random.random(self.shape_x).astype("float32")
        self.prog = None

    def base_net(self, flag=None):
        if flag == "forward":
            core._set_prim_forward_enabled(True)
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.static.data('x', self.shape_x, dtype='float32')
            x.stop_gradient = False
            sum_out = F.relu(x)
            [new_out] = decompose(main_program, [sum_out])
            gradients = grad(new_out, x)

            exe = paddle.static.Executor()
            [fwd, dx] = exe.run(
                feed={'x': self.x}, fetch_list=[new_out, gradients]
            )

        whole_ops = [op.name() for op in main_program.global_block().ops]
        self.prog = main_program
        if flag == "forward":
            core._set_prim_forward_enabled(False)
            assert 'pd_op.relu' not in whole_ops
        else:
            assert 'pd_op.relu' in whole_ops
        return fwd, dx

    def test_relu_forward(self):
        res_ref = self.base_net()
        res = self.base_net("forward")
        for ref, actual in zip(res_ref, res):
            np.testing.assert_equal(ref, actual)


class TestGeluSink(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [8, 16, 32, 64]
        self.x = np.random.random(self.shape_x).astype("float32")
        self.prog = None

    def base_net(self, approximate=True, flag=None):
        if flag == "forward":
            core._set_prim_forward_enabled(True)
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.static.data('x', self.shape_x, dtype='float32')
            x.stop_gradient = False
            sum_out = F.gelu(x, approximate=approximate)
            [new_out] = decompose(main_program, [sum_out])
            gradients = grad(new_out, x)

            exe = paddle.static.Executor()
            [fwd, dx] = exe.run(
                feed={'x': self.x}, fetch_list=[new_out, gradients]
            )

        whole_ops = [op.name() for op in main_program.global_block().ops]
        self.prog = main_program
        if flag == "forward":
            core._set_prim_forward_enabled(False)
            assert 'pd_op.gelu' not in whole_ops
        else:
            assert 'pd_op.gelu' in whole_ops
        return fwd, dx

    def test_gelu_forward_true(self):
        res_ref = self.base_net(approximate=True)
        res = self.base_net(approximate=True, flag="forward")
        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(ref, actual, rtol=1e-6)

    def test_gelu_approximate_false(self):
        res_ref = self.base_net(approximate=False)
        res = self.base_net(approximate=False, flag="forward")
        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(ref, actual, rtol=1e-6)


class TestHardSwishSink(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [8, 16, 32, 64]
        self.x = np.random.random(self.shape_x).astype("float32")
        self.prog = None

    def base_net(self, flag=None):
        if flag == "forward":
            core._set_prim_forward_enabled(True)
        elif flag == "backward":
            core._set_prim_backward_enabled(True)
        elif flag == "all":
            core._set_prim_all_enabled(True)
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.static.data('x', self.shape_x, dtype='float32')
            x.stop_gradient = False
            sum_out = F.hardswish(x)
            [new_out] = decompose(main_program, [sum_out])
            gradients = grad(new_out, x)

            exe = paddle.static.Executor()
            [fwd, dx] = exe.run(
                feed={'x': self.x}, fetch_list=[new_out, gradients]
            )

        whole_ops = [op.name() for op in main_program.global_block().ops]
        self.prog = main_program
        if flag == "forward":
            core._set_prim_forward_enabled(False)
            assert 'pd_op.hardswish' not in whole_ops
        elif flag == "backward":
            core._set_prim_backward_enabled(False)
            assert (
                'pd_op.hardswish' in whole_ops
                and 'pd_op.hardswish_grad' not in whole_ops
            )
        elif flag == "all":
            core._set_prim_all_enabled(False)
            assert (
                'pd_op.hardswish' not in whole_ops
                and 'pd_op.hardswish_grad' not in whole_ops
            )
        else:
            assert (
                'pd_op.hardswish' in whole_ops
                and 'pd_op.hardswish_grad' in whole_ops
            )
        return fwd, dx

    def test_prim_forward(self):
        res_ref = self.base_net()
        res = self.base_net("forward")
        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(ref, actual, rtol=1e-3, atol=1e-3)

    def test_prim_backward(self):
        res_ref = self.base_net()
        res = self.base_net("backward")
        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(ref, actual, rtol=1e-3, atol=1e-3)

    def test_prim_all(self):
        res_ref = self.base_net()
        res = self.base_net("all")
        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(ref, actual, rtol=1e-3, atol=1e-3)


class TestEmbeddingSink(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [3, 2]
        self.shape_w = [10, 3]
        self.x = np.array([[3, 4, -1], [2, -2, 4]], dtype=int).reshape(
            self.shape_x
        )
        self.w = np.array(
            [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                [9, 10, 11],
                [12, 13, 14],
                [15, 16, 17],
                [18, 19, 20],
                [21, 22, 23],
                [24, 25, 26],
                [27, 28, 29],
            ],
            dtype=np.float32,
        )
        self.prog = None

    def base_net(self, flag=None):
        if flag == "forward":
            core._set_prim_forward_enabled(True)
        elif flag == "backward":
            core._set_prim_backward_enabled(True)
        elif flag == "all":
            core._set_prim_all_enabled(True)
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.static.data('x', self.shape_x, dtype=int)
            w = paddle.static.data('w', self.shape_w, dtype='float32')
            x.stop_gradient = False
            w.stop_gradient = False
            sum_out = F.embedding(x, w)
            [new_out] = decompose(main_program, [sum_out])
            gradients = grad(new_out, (x, w))

            print(main_program)

            exe = paddle.static.Executor()
            [fwd, dx, dy] = exe.run(
                feed={'x': self.x, 'w': self.w}, fetch_list=[new_out, gradients]
            )

        whole_ops = [op.name() for op in main_program.global_block().ops]
        self.prog = main_program
        if flag == "forward":
            core._set_prim_forward_enabled(False)
            assert 'pd_op.embedding' not in whole_ops
        elif flag == "backward":
            core._set_prim_backward_enabled(False)
            assert 'pd_op.embedding' in whole_ops
        elif flag == "all":
            core._set_prim_all_enabled(False)
            assert 'pd_op.embedding' not in whole_ops
        else:
            assert 'pd_op.embedding' in whole_ops
        return fwd, dx, dy

    def test_prim_forward(self):
        # res_ref = self.base_net()
        res = self.base_net("forward")
        # for ref, actual in zip(res_ref, res):
        #     np.testing.assert_allclose(ref, actual, rtol=1e-3, atol=1e-3)


def embedding_test():
    np.random.seed(2023)
    shape_x = [3, 1]
    x = np.array([[3, 4, 1]], dtype=int).reshape(shape_x)
    x = paddle.to_tensor(x)
    w = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
            [12, 13, 14],
            [15, 16, 17],
            [18, 19, 20],
            [21, 22, 23],
            [24, 25, 26],
            [27, 28, 29],
        ],
        dtype=np.float32,
    )
    w = paddle.to_tensor(w)

    index = paddle.to_tensor([[2]])
    print(index.shape)

    w = paddle.put_along_axis(w, index, 0.0, axis=0)
    print(w)

    x = x.reshape([-1, 1])
    print(x.shape)

    out = paddle.gather(w, x)
    out = out.reshape([3, 1, -1])
    print(out)


if __name__ == "__main__":
    unittest.main()
    # embedding_test()
