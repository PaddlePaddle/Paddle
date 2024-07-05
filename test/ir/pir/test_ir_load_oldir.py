# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle


class TestALoadPdmodelTranslatePir(unittest.TestCase):
    def setUp(self):
        paddle.seed(2022)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.save_path = os.path.join(self.temp_dir.name, 'saveload')
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_inference_model(self):
        paddle.enable_static()
        if paddle.framework.in_pir_mode():
            return
        np_x = np.random.randn(9, 10, 11).astype('float32')
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
            x = paddle.static.data(shape=np_x.shape, name='x', dtype=np_x.dtype)
            linear = paddle.nn.Linear(np_x.shape[-1], np_x.shape[-1])
            linear_out = linear(x)
            relu_out = paddle.nn.functional.relu(linear_out)
            axis = paddle.full([1], 2, dtype='int64')
            out = paddle.cumsum(relu_out, axis=axis)
            loss = paddle.mean(out)
            sgd = paddle.optimizer.SGD(learning_rate=0.0)
            sgd.minimize(paddle.mean(out))

            exe = paddle.static.Executor(self.place)
            exe.run(startup_prog)
            out_old = exe.run(feed={'x': np_x}, fetch_list=[out])

            # run infer
            paddle.static.save_inference_model(
                self.save_path, [x], [out], exe, program=main_prog
            )

            exe = paddle.static.Executor(self.place)

            load_program, _, _ = paddle.static.load_inference_model(
                self.save_path, exe
            )

        with paddle.pir_utils.IrGuard():
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(load_program, startup_prog):
                exe.run(startup_prog)
                out_new = exe.run(load_program, feed={'x': np_x}, fetch_list=[])
                np.testing.assert_allclose(out_old, out_new)

        load_program = paddle.load(
            self.save_path + '.pdmodel',
        )

        with paddle.pir_utils.IrGuard():
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(load_program, startup_prog):
                exe.run(startup_prog)
                out_new = exe.run(load_program, feed={'x': np_x}, fetch_list=[])
                np.testing.assert_allclose(out_old, out_new)


class TestJitSaveOp(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, "pir_save_load")
        paddle.disable_static()
        linear = paddle.nn.Linear(10, 10)
        path = os.path.join(self.model_path, "linear")

        paddle.jit.save(
            linear,
            path,
            input_spec=[paddle.static.InputSpec([10, 10], 'float32', 'x')],
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_with_pir(self):
        paddle.enable_static()
        if paddle.framework.in_pir_mode():
            return
        place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

        exe = paddle.static.Executor(place)

        [
            inference_program,
            feed_target_names,
            fetch_targets,
        ] = paddle.static.io.load_inference_model(
            self.model_path,
            executor=exe,
            model_filename="linear.pdmodel",
            params_filename="linear.pdiparams",
        )


if __name__ == '__main__':
    unittest.main()
