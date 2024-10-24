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

import sys
import unittest

import numpy as np

sys.path.append("../../legacy_test")
from test_eager_deletion_padding_rnn import PaddingRNNTestBase, RNNConfig

import paddle
from paddle import base


class ExecutorPaddingRNNTest(PaddingRNNTestBase):
    def train_and_save_inference_program(
        self, rnn_model="static", use_program_cache=True
    ):
        config = RNNConfig("test", rnn_model)
        with base.scope_guard(base.Scope()):
            self.train(config, use_program_cache)
            paddle.static.io.save_inference_model(
                path_prefix="padding_rnn." + rnn_model + ".inference_model",
                feed_vars=self.feed_list,
                fetch_vars=[self.loss, self.last_hidden, self.last_cell],
                executor=self.exe,
                program=self.main_program,
            )

    def test_inference_output(self):
        for rnn_model in ["static"]:
            # Set parallel to False to use the default executor.
            self.train_and_save_inference_program(
                rnn_model=rnn_model, use_program_cache=True
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
                with base.scope_guard(base.Scope()):
                    save_dirname = (
                        "padding_rnn." + rnn_model + ".inference_model"
                    )
                    [
                        inference_program,
                        feed_target_names,
                        fetch_targets,
                    ] = paddle.static.io.load_inference_model(
                        save_dirname, self.exe
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
                    if use_program_cache is True:
                        results_with_cache = results
                    else:
                        results_without_cache = results
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


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
