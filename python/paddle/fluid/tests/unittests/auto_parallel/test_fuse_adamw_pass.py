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
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.static as static
from paddle.distributed.passes import PassManager, new_pass


def apply_passes(main_prog, startup_prog):
    pass_manager = PassManager([new_pass("fuse_adamw")])
    pass_manager.apply([main_prog], [startup_prog])


class MLPLayer(nn.Layer):
    def __init__(self, input_size, hidden_size, output_size, n):
        super(MLPLayer, self).__init__()
        self.linear_first = nn.Linear(input_size, hidden_size)
        self.decoder_layers = nn.LayerList()
        for i in range(n):
            self.decoder_layers.append(nn.Linear(hidden_size, hidden_size))

        self.linear_last = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear_first(x)
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.linear_last(x)
        return x.mean()


class TestFuseAdamWPass(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.input_size = 30
        self.hidden_size = 50
        self.output_size = 20
        self.n = 2

    def get_loss_data(self, use_apply_passes):
        paddle.enable_static()
        np.random.seed(10)
        paddle.seed(10)
        paddle.set_device("gpu")
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.scope_guard(paddle.static.Scope()):
            with paddle.static.program_guard(main_prog, startup_prog):
                model = MLPLayer(
                    self.input_size, self.hidden_size, self.output_size, self.n
                )
                inp = static.data(
                    name='x', shape=[10, self.input_size], dtype='float32'
                )
                y = static.data(
                    name='y', shape=[10, self.output_size], dtype='float32'
                )
                inp_data = np.random.random([10, self.input_size]).astype(
                    "float32"
                )
                y_data = np.random.random([10, self.output_size]).astype(
                    "float32"
                )
                beta1 = 0.9
                beta2 = 0.999
                input_paramters = model.parameters()

                out = model(inp)
                cost = F.square_error_cost(input=out, label=y)
                loss = paddle.mean(cost)

                decay_params = [
                    p.name
                    for p in input_paramters
                    if not any(nd in p.name for nd in ["b_"])
                ]

                apply_decay_param_fun = lambda x: x in decay_params

                # AdamW
                opt = paddle.optimizer.AdamW(
                    learning_rate=0.001,
                    apply_decay_param_fun=apply_decay_param_fun,
                )
                opt.minimize(loss)

                if use_apply_passes:
                    apply_passes(
                        static.default_main_program(),
                        static.default_startup_program(),
                    )

                places = paddle.static.cuda_places()
                exe = static.Executor(paddle.CUDAPlace(0))
                exe.run(static.default_startup_program())

                compiled_program = static.CompiledProgram(
                    static.default_main_program()
                )

                # train
                for num_batch in range(5):
                    loss_data = exe.run(
                        compiled_program,
                        feed={'x': inp_data, 'y': y_data},
                        fetch_list=[loss],
                    )

        return loss_data

    def test_fuse_adamw_pass(self):
        loss_without_passes = self.get_loss_data(False)
        loss_with_passes = self.get_loss_data(True)
        np.testing.assert_allclose(
            np.array(loss_without_passes),
            np.array(loss_with_passes),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
