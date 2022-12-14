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

import unittest

import numpy as np
from test_imperative_base import new_program_scope
from test_imperative_ptb_rnn import PtbModel

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.framework as framework
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.optimizer import SGDOptimizer


class TestDygraphPtbRnnSortGradient(unittest.TestCase):
    def test_ptb_rnn_sort_gradient(self):
        for is_sparse in [True, False]:
            self.ptb_rnn_sort_gradient_cpu_float32(is_sparse)

    def ptb_rnn_sort_gradient_cpu_float32(self, is_sparse):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 200

        with fluid.dygraph.guard():
            fluid.set_flags({'FLAGS_sort_sum_gradient': True})
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)

            # TODO: marsyang1993 Change seed to
            ptb_model = PtbModel(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale,
                is_sparse=is_sparse,
            )

            sgd = SGDOptimizer(
                learning_rate=1e-3, parameter_list=ptb_model.parameters()
            )
            dy_param_updated = dict()
            dy_param_init = dict()
            dy_loss = None
            last_hidden = None
            last_cell = None

            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                x_data = x_data.reshape((-1, num_steps, 1))
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                x = to_variable(x_data)
                y = to_variable(y_data)
                init_hidden = to_variable(init_hidden_data)
                init_cell = to_variable(init_cell_data)
                dy_loss, last_hidden, last_cell = ptb_model(
                    x, y, init_hidden, init_cell
                )
                if i == 0:
                    for param in ptb_model.parameters():
                        dy_param_init[param.name] = param.numpy()
                dy_loss.backward()
                sgd.minimize(dy_loss)
                ptb_model.clear_gradients()
                if i == batch_num - 1:
                    for param in ptb_model.parameters():
                        dy_param_updated[param.name] = param.numpy()

            dy_loss_value = dy_loss.numpy()
            dy_last_cell_value = last_cell.numpy()
            dy_last_hidden_value = last_hidden.numpy()

        with new_program_scope():
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)

            ptb_model = PtbModel(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale,
                is_sparse=is_sparse,
            )

            exe = fluid.Executor(
                fluid.CPUPlace()
                if not core.is_compiled_with_cuda()
                else fluid.CUDAPlace(0)
            )
            sgd = SGDOptimizer(learning_rate=1e-3)
            x = fluid.layers.data(
                name="x", shape=[-1, num_steps, 1], dtype='int64'
            )
            y = fluid.layers.data(name="y", shape=[-1, 1], dtype='float32')
            init_hidden = fluid.layers.data(
                name="init_hidden", shape=[1], dtype='float32'
            )
            init_cell = fluid.layers.data(
                name="init_cell", shape=[1], dtype='float32'
            )

            static_loss, static_last_hidden, static_last_cell = ptb_model(
                x, y, init_hidden, init_cell
            )
            sgd.minimize(static_loss)
            static_param_updated = dict()
            static_param_init = dict()
            static_param_name_list = list()
            for param in ptb_model.parameters():
                static_param_name_list.append(param.name)

            out = exe.run(
                framework.default_startup_program(),
                fetch_list=static_param_name_list,
            )
            for i in range(len(static_param_name_list)):
                static_param_init[static_param_name_list[i]] = out[i]
            static_loss_value = None
            static_last_cell_value = None
            static_last_hidden_value = None
            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                x_data = x_data.reshape((-1, num_steps, 1))
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32'
                )
                fetch_list = [static_loss, static_last_hidden, static_last_cell]
                fetch_list.extend(static_param_name_list)
                out = exe.run(
                    fluid.default_main_program(),
                    feed={
                        "x": x_data,
                        "y": y_data,
                        "init_hidden": init_hidden_data,
                        "init_cell": init_cell_data,
                    },
                    fetch_list=fetch_list,
                )
                static_loss_value = out[0]
                static_last_hidden_value = out[1]
                static_last_cell_value = out[2]

                if i == batch_num - 1:
                    for k in range(3, len(out)):
                        static_param_updated[
                            static_param_name_list[k - 3]
                        ] = out[k]

        np.testing.assert_array_equal(static_loss_value, dy_loss_value)
        np.testing.assert_array_equal(
            static_last_cell_value, dy_last_cell_value
        )
        np.testing.assert_array_equal(
            static_last_hidden_value, dy_last_hidden_value
        )
        for key, value in static_param_init.items():
            np.testing.assert_array_equal(value, dy_param_init[key])
        for key, value in static_param_updated.items():
            np.testing.assert_array_equal(value, dy_param_updated[key])


if __name__ == '__main__':
    unittest.main()
