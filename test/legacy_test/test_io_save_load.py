#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import tempfile
import unittest

import paddle
from paddle import base, static
from paddle.base import core


class TestSaveLoadAPIError(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.save_dir = os.path.join(self.temp_dir.name, "fake_dir")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_get_valid_program_error(self):
        # case 1: CompiledProgram no program
        graph = core.Graph(core.ProgramDesc())
        compiled_program = base.CompiledProgram(graph)
        with self.assertRaises(TypeError):
            paddle.static.io._get_valid_program(compiled_program)

        # case 2: main_program type error
        with self.assertRaises(TypeError):
            paddle.static.io._get_valid_program("program")

    def test_load_vars_error(self):
        place = base.CPUPlace()
        exe = base.Executor(place)
        # case 1: main_program type error when vars None
        with self.assertRaises(TypeError):
            static.io.load_vars(
                executor=exe, dirname=self.save_dir, main_program="program"
            )

        # case 2: main_program type error when vars not None
        with self.assertRaises(TypeError):
            static.io.load_vars(
                executor=exe,
                dirname=self.save_dir,
                main_program="program",
                vars="vars",
            )


class TestSaveInferenceModelAPIError(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_useless_feeded_var_names(self):
        start_prog = base.Program()
        main_prog = base.Program()
        with base.program_guard(main_prog, start_prog):
            x = paddle.static.data(name='x', shape=[10, 16], dtype='float32')
            y = paddle.static.data(name='y', shape=[10, 16], dtype='float32')
            z = paddle.static.nn.fc(x, 4)

        exe = base.Executor(base.CPUPlace())
        exe.run(start_prog)
        with self.assertRaisesRegex(
            ValueError, "not involved in the target_vars calculation"
        ):
            paddle.static.io.save_inference_model(
                path_prefix=os.path.join(self.temp_dir.name, 'model'),
                feed_vars=[x, y],
                fetch_vars=[z],
                executor=exe,
                program=main_prog,
            )


class TestWhenTrainWithNoGrad(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_when_train_with_no_grad(self):
        paddle.disable_static()
        net = paddle.nn.Linear(1024, 1)
        net = paddle.jit.to_static(net, full_graph=True)
        x = paddle.rand([1024], 'float32')
        x.stop_gradient = False
        out = net(x)
        out.backward()
        x_grad = x.grad.mean()
        x.clear_grad()

        # jit.save
        save_path = os.path.join(self.temp_dir.name, 'train_with_no_grad')
        paddle.jit.save(net, save_path)

        # test eval mode
        net1 = paddle.jit.load(save_path)
        net1.eval()

        with paddle.no_grad():
            out1 = net1(x)
            self.assertEqual(out, out1)

        # test train mode
        net2 = paddle.jit.load(save_path)
        net2.train()
        out2 = net2(x)
        out2.backward()
        self.assertEqual(out, out2)
        x_grad2 = x.grad.mean()
        if paddle.framework.in_pir_mode():
            self.assertEqual(x_grad, x_grad2)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
