#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from test_eager_deletion_padding_rnn import PaddingRNNTestBase, RNNConfig

import paddle
import paddle.fluid as fluid


class TestExecutor(unittest.TestCase):
=======
from __future__ import print_function

import unittest

import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
from test_eager_deletion_padding_rnn import RNNConfig, PaddingRNNTestBase


class TestExecutor(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_mul(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
<<<<<<< HEAD
            a = paddle.static.data(name='a', shape=[-1, 784], dtype='float32')
            b = paddle.static.data(name='b', shape=[784, 100], dtype='float32')
            a.desc.set_need_check_feed(False)
            b.desc.set_need_check_feed(False)
            output = paddle.matmul(x=a, y=b)
=======
            a = fluid.layers.data(name='a', shape=[784], dtype='float32')
            b = fluid.layers.data(name='b',
                                  shape=[784, 100],
                                  dtype='float32',
                                  append_batch_size=False)
            output = fluid.layers.mul(x=a, y=b)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        # Compute with numpy
        a_np = np.random.random((100, 784)).astype('float32')
        b_np = np.random.random((784, 100)).astype('float32')
        out_np = np.dot(a_np, b_np)

<<<<<<< HEAD
        place = paddle.CPUPlace()
=======
        place = core.CPUPlace()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        exe = fluid.Executor(place)

        def _train(use_program_cache, max_iters=1):
            import time

            run_time = 0.0
            for i in range(max_iters):
                begin = time.time()
<<<<<<< HEAD
                outs = exe.run(
                    program=main_program,
                    feed={'a': a_np, 'b': b_np},
                    fetch_list=[output.name],
                    use_program_cache=use_program_cache,
                )
=======
                outs = exe.run(program=main_program,
                               feed={
                                   'a': a_np,
                                   'b': b_np
                               },
                               fetch_list=[output.name],
                               use_program_cache=use_program_cache)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                end = time.time()
                run_time += end - begin
                out = outs[0]
                self.assertEqual((100, 100), out.shape)
                np.testing.assert_allclose(out, out_np, rtol=1e-05)
            return run_time

        max_iters = 3
<<<<<<< HEAD
        run_time_with_cache = _train(
            use_program_cache=True, max_iters=max_iters
        )
        print("run time with program cache: %f" % run_time_with_cache)

        run_time_without_cache = _train(
            use_program_cache=False, max_iters=max_iters
        )
        print("run time without program cache: %f" % run_time_without_cache)

        run_time_with_cache = _train(
            use_program_cache=True, max_iters=max_iters
        )
        print("run time with program cache: %f" % run_time_with_cache)

        run_time_with_cache = _train(
            use_program_cache=True, max_iters=max_iters
        )
=======
        run_time_with_cache = _train(use_program_cache=True,
                                     max_iters=max_iters)
        print("run time with program cache: %f" % run_time_with_cache)

        run_time_without_cache = _train(use_program_cache=False,
                                        max_iters=max_iters)
        print("run time without program cache: %f" % run_time_without_cache)

        run_time_with_cache = _train(use_program_cache=True,
                                     max_iters=max_iters)
        print("run time with program cache: %f" % run_time_with_cache)

        run_time_with_cache = _train(use_program_cache=True,
                                     max_iters=max_iters)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        print("run time with program cache: %f" % run_time_with_cache)


class ExecutorPaddingRNNTest(PaddingRNNTestBase):
<<<<<<< HEAD
    def train_and_save_inference_program(
        self, rnn_model="static", parallel=True, use_program_cache=True
    ):
=======

    def train_and_save_inference_program(self,
                                         rnn_model="static",
                                         parallel=True,
                                         use_program_cache=True):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        config = RNNConfig("test", rnn_model)
        with fluid.scope_guard(fluid.Scope()):
            self.train(config, parallel, use_program_cache)
            fluid.io.save_inference_model(
                main_program=self.main_program,
                feeded_var_names=self.feed_order,
                target_vars=[self.loss, self.last_hidden, self.last_cell],
                executor=self.exe,
                dirname="padding_rnn." + rnn_model + ".inference_model",
<<<<<<< HEAD
                params_filename="__params__",
            )
=======
                params_filename="__params__")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_inference_output(self):
        for rnn_model in ["static", "padding"]:
            # Set parallel to False to use the default executor.
<<<<<<< HEAD
            self.train_and_save_inference_program(
                rnn_model=rnn_model, parallel=True, use_program_cache=True
            )

            x_np = np.random.random(
                (self.config.batch_size, self.config.num_steps, 1)
            ).astype("int64")
            y_np = np.random.random(
                (self.config.batch_size * self.config.num_steps, 1)
            ).astype("int64")
            init_hidden_np = np.random.random(
                (
                    self.config.num_layers,
                    self.config.batch_size,
                    self.config.hidden_size,
                )
            ).astype("float32")
            init_cell_np = np.random.random(
                (
                    self.config.num_layers,
                    self.config.batch_size,
                    self.config.hidden_size,
                )
            ).astype("float32")

            for use_program_cache in [False, True]:
                with fluid.scope_guard(fluid.Scope()):
                    save_dirname = (
                        "padding_rnn." + rnn_model + ".inference_model"
                    )
                    [
                        inference_program,
                        feed_target_names,
                        fetch_targets,
                    ] = fluid.io.load_inference_model(
                        save_dirname, self.exe, params_filename="__params__"
                    )

                    results = self.exe.run(
                        program=inference_program,
                        feed={
                            "x": x_np,
                            "y": y_np,
                            "init_hidden": init_hidden_np,
                            "init_cell": init_cell_np,
                        },
                        fetch_list=fetch_targets,
                        use_program_cache=use_program_cache,
                    )
=======
            self.train_and_save_inference_program(rnn_model=rnn_model,
                                                  parallel=True,
                                                  use_program_cache=True)

            x_np = np.random.random((self.config.batch_size,
                                     self.config.num_steps, 1)).astype("int64")
            y_np = np.random.random(
                (self.config.batch_size * self.config.num_steps,
                 1)).astype("int64")
            init_hidden_np = np.random.random(
                (self.config.num_layers, self.config.batch_size,
                 self.config.hidden_size)).astype("float32")
            init_cell_np = np.random.random(
                (self.config.num_layers, self.config.batch_size,
                 self.config.hidden_size)).astype("float32")

            for use_program_cache in [False, True]:
                with fluid.scope_guard(fluid.Scope()):
                    save_dirname = "padding_rnn." + rnn_model + ".inference_model"
                    [inference_program, feed_target_names,
                     fetch_targets] = fluid.io.load_inference_model(
                         save_dirname, self.exe, params_filename="__params__")

                    results = self.exe.run(program=inference_program,
                                           feed={
                                               "x": x_np,
                                               "y": y_np,
                                               "init_hidden": init_hidden_np,
                                               "init_cell": init_cell_np
                                           },
                                           fetch_list=fetch_targets,
                                           use_program_cache=use_program_cache)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    if use_program_cache is True:
                        results_with_cache = results
                    else:
                        results_without_cache = results
<<<<<<< HEAD
            self.assertEqual(
                len(results_with_cache), len(results_without_cache)
            )
            for i in range(len(results_with_cache)):
                self.assertEqual(
                    results_with_cache[i].shape, results_without_cache[i].shape
                )
                np.testing.assert_allclose(
                    results_with_cache[i], results_without_cache[i], rtol=1e-05
                )
=======
            self.assertEqual(len(results_with_cache),
                             len(results_without_cache))
            for i in range(len(results_with_cache)):
                self.assertEqual(results_with_cache[i].shape,
                                 results_without_cache[i].shape)
                np.testing.assert_allclose(results_with_cache[i],
                                           results_without_cache[i],
                                           rtol=1e-05)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == '__main__':
    unittest.main()
