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

import os
import pathlib
import sys
import unittest

import numpy as np

import paddle
from paddle import base
from paddle.autograd.ir_backward import grad as ir_grad
from paddle.framework import core

sys.path.append(
    str(pathlib.Path(__file__).resolve().parents[2] / 'legacy_test')
)
from utils import static_guard


class IfNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, cond):
        if cond:
            x = x + 1
            out1 = paddle.mean(x)
            out2 = paddle.mean(y)
        else:
            y = y + 1
            out1 = paddle.mean(x)
            out2 = paddle.mean(y)
        return out1, out2


class WhileNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        while paddle.all(x < y):
            x = x + 1
            out = paddle.mean(y**2)
        return out


class WhileAndIfNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        while paddle.all(x < y):
            if paddle.all(x + 1 < y):
                x = x + 0.5
                out = paddle.mean(y**2)
            else:
                x = x + 0.6
                out = paddle.mean(paddle.nn.functional.softmax(y))
        return out


class TestPrimControlFlowIf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core._set_prim_forward_enabled(False)

    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [8, 16, 32]
        self.shape_y = [8, 16, 32]
        self.x_np = np.random.random(self.shape_x).astype("float32")
        self.y_np = np.random.random(self.shape_y).astype("float32")
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            self.places.append(paddle.CPUPlace())
        if paddle.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def get_control_if_res(self, x, y, cond, prim_forward=False):
        if prim_forward:
            core._set_prim_forward_enabled(True)
        net = IfNet()
        net = paddle.jit.to_static(net, full_graph=True)
        out1, out2 = net(x, y, cond)
        core._set_prim_forward_enabled(False)
        return out1, out2

    def test_decompose_if_true(self):
        for place in self.places:
            if isinstance(place, paddle.base.CPUPlace):
                paddle.set_device("cpu")
            elif isinstance(place, paddle.base.CUDAPlace):
                paddle.set_device("gpu")
            x = paddle.to_tensor(self.x_np, dtype="float32")
            y = paddle.to_tensor(self.y_np, dtype="float32")
            cond = paddle.full(shape=[1], fill_value=1)
            out1_baseline, out2_baseline = self.get_control_if_res(
                x, y, cond, prim_forward=False
            )
            out1, out2 = self.get_control_if_res(x, y, cond, prim_forward=True)
            np.testing.assert_allclose(out1_baseline, out1, rtol=1e-6, atol=0)
            np.testing.assert_allclose(out2_baseline, out2, rtol=1e-6, atol=0)

    def test_decompose_if_false(self):
        for place in self.places:
            if isinstance(place, paddle.base.CPUPlace):
                paddle.set_device("cpu")
            elif isinstance(place, paddle.base.CUDAPlace):
                paddle.set_device("gpu")
            x = paddle.to_tensor(self.x_np, dtype="float32")
            y = paddle.to_tensor(self.y_np, dtype="float32")
            cond = paddle.full(shape=[1], fill_value=0)
            out1_baseline, out2_baseline = self.get_control_if_res(
                x, y, cond, prim_forward=False
            )
            out1, out2 = self.get_control_if_res(x, y, cond, prim_forward=True)
            np.testing.assert_allclose(out1_baseline, out1, rtol=1e-6, atol=0)
            np.testing.assert_allclose(out2_baseline, out2, rtol=1e-6, atol=0)

    @classmethod
    def tearDownClass(cls):
        core._set_prim_forward_enabled(False)


class TestPrimControlFlowWhile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core._set_prim_forward_enabled(False)

    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [8, 16, 32]
        self.shape_y = [8, 16, 32]
        self.x_np = np.random.random(self.shape_x).astype("float32")
        self.y_np = np.random.random(self.shape_y).astype("float32") + 3
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            self.places.append(paddle.CPUPlace())
        if paddle.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def get_control_while_res(self, x, y, prim_forward=False):
        if prim_forward:
            core._set_prim_forward_enabled(True)
        net = WhileNet()
        net = paddle.jit.to_static(net, full_graph=True)
        out = net(x, y)
        core._set_prim_forward_enabled(False)
        return out

    def test_decompose_while(self):
        for place in self.places:
            if isinstance(place, paddle.base.CPUPlace):
                paddle.set_device("cpu")
            elif isinstance(place, paddle.base.CUDAPlace):
                paddle.set_device("gpu")
            x = paddle.to_tensor(self.x_np, dtype="float32")
            y = paddle.to_tensor(self.y_np, dtype="float32")
            out_baseline = self.get_control_while_res(x, y, prim_forward=False)
            out = self.get_control_while_res(x, y, prim_forward=True)
            np.testing.assert_allclose(out_baseline, out, rtol=1e-6, atol=0)

    @classmethod
    def tearDownClass(cls):
        core._set_prim_forward_enabled(False)


class TestPrimControlFlowWhileAndIf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core._set_prim_forward_enabled(False)

    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [8, 16, 32]
        self.shape_y = [8, 16, 32]
        self.x_np = np.random.random(self.shape_x).astype("float32")
        self.y_np = np.random.random(self.shape_y).astype("float32") + 3
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            self.places.append(paddle.CPUPlace())
        if paddle.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def get_control_flow_res(self, x, y, prim_forward=False):
        if prim_forward:
            core._set_prim_forward_enabled(True)
        net = WhileAndIfNet()
        net = paddle.jit.to_static(net, full_graph=True)
        out = net(x, y)
        core._set_prim_forward_enabled(False)
        return out

    def test_decompose_while_and_if(self):
        for place in self.places:
            if isinstance(place, paddle.base.CPUPlace):
                paddle.set_device("cpu")
            elif isinstance(place, paddle.base.CUDAPlace):
                paddle.set_device("gpu")
            x = paddle.to_tensor(self.x_np, dtype="float32")
            y = paddle.to_tensor(self.y_np, dtype="float32")
            out_baseline = self.get_control_flow_res(x, y, prim_forward=False)
            out = self.get_control_flow_res(x, y, prim_forward=True)
            np.testing.assert_allclose(out_baseline, out, rtol=1e-6, atol=0)

    @classmethod
    def tearDownClass(cls):
        core._set_prim_forward_enabled(False)


class TestPrimControlFlowWhileBackward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core._set_prim_all_enabled(False)

    def setUp(self):
        np.random.seed(2023)
        self.shape_i = [1]
        self.shape_x = [1]
        self.i_np = np.random.random(self.shape_i).astype("float32")
        self.x_np = np.random.random(self.shape_x).astype("float32")

    def cond(self, i, x):
        return i < 3

    def body(self, i, x):
        x = paddle.pow(x, i)
        i = i + 1
        return [i, x]

    def get_while_prim_grad_res(self):
        core._set_prim_all_enabled(True)
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            i = paddle.static.data(name='i', shape=[1], dtype='float32')
            i.stop_gradient = False
            i.persistable = True
            x = paddle.static.data(name='x', shape=[1], dtype='float32')
            x.stop_gradient = False
            x.persistable = True

            out = paddle.static.nn.while_loop(self.cond, self.body, [i, x])
            [new_out] = paddle.decomposition.decomp.decompose(
                main_program, [out[1]]
            )
            out_grad = ir_grad(new_out, [x])

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        out_grad = exe.run(
            main_program,
            feed={'i': self.i_np, 'x': self.x_np},
            fetch_list=[out_grad],
        )
        core._set_prim_all_enabled(False)
        return main_program, out_grad[0]

    def get_while_grad_res(self):
        core._set_prim_all_enabled(False)
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            i = paddle.static.data(name='i', shape=[1], dtype='float32')
            i.stop_gradient = False
            i.persistable = True
            x = paddle.static.data(name='x', shape=[1], dtype='float32')
            x.stop_gradient = False
            x.persistable = True

            out = paddle.static.nn.while_loop(self.cond, self.body, [i, x])
            out_grad = ir_grad(out, [x])

        place = (
            base.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)
        out_grad = exe.run(
            main_program,
            feed={'i': self.i_np, 'x': self.x_np},
            fetch_list=[out_grad],
        )
        return main_program, out_grad[0]

    def test_while_loop_backward2(self):
        with static_guard():
            program_origin, out_grad_baseline = self.get_while_grad_res()
            program_prim, out_grad = self.get_while_prim_grad_res()
        np.testing.assert_allclose(
            out_grad_baseline, out_grad, rtol=1e-6, atol=0
        )
        assert len(
            program_origin.global_block().ops[-1].as_while_op().body().ops
        ) != len(program_prim.global_block().ops[-1].as_while_op().body().ops)

    @classmethod
    def tearDownClass(cls):
        core._set_prim_all_enabled(False)


if __name__ == "__main__":
    unittest.main()
