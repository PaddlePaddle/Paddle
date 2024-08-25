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
from paddle import base
from paddle.autograd.backward_utils import ValueDict
from paddle.base import core


def create_parameter_mapping(startup_program, main_program):
    startup_params = {}
    main_params = {}
    parameter_mapping = ValueDict()
    for op in startup_program.global_block().ops:
        if op.name() == "builtin.set_parameter":
            name = op.attrs()["parameter_name"]
            param = op.operand(0).source()
            startup_params[name] = param

    for op in main_program.global_block().ops:
        if op.name() == "builtin.parameter":
            name = op.attrs()["parameter_name"]
            param = op.result(0)
            main_params[name] = param

    assert len(startup_params) == len(main_params)
    for name, startup_param in startup_params.items():
        assert name in main_params
        main_param = main_params[name]
        parameter_mapping[main_param] = startup_param
    return parameter_mapping


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

        with base.dygraph.guard():
            base.set_flags({'FLAGS_sort_sum_gradient': True})
            paddle.seed(seed)
            if paddle.framework.use_pir_api():
                with paddle.pir_utils.OldIrGuard():
                    # Note: dygraph use self.main_program.global_block().create_parameter(), it's need manual seed to old Program
                    paddle.framework.random._manual_program_seed(seed)
                paddle.framework.random._manual_program_seed(seed)
            else:
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

            sgd = paddle.optimizer.SGD(
                learning_rate=1e-3, parameters=ptb_model.parameters()
            )
            dy_param_updated = {}
            dy_param_init = {}
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
                x = paddle.to_tensor(x_data)
                y = paddle.to_tensor(y_data)
                init_hidden = paddle.to_tensor(init_hidden_data)
                init_cell = paddle.to_tensor(init_cell_data)
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
            if paddle.framework.use_pir_api():
                with paddle.pir_utils.OldIrGuard():
                    # Note: dygraph use self.main_program.global_block().create_parameter(), it's need manual seed to old Program
                    paddle.framework.random._manual_program_seed(seed)
                paddle.framework.random._manual_program_seed(seed)
            else:
                paddle.framework.random._manual_program_seed(seed)

            ptb_model = PtbModel(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale,
                is_sparse=is_sparse,
            )

            exe = base.Executor(
                base.CPUPlace()
                if not core.is_compiled_with_cuda()
                else base.CUDAPlace(0)
            )
            sgd = paddle.optimizer.SGD(learning_rate=1e-3)
            x = paddle.static.data(
                name="x", shape=[-1, num_steps, 1], dtype='int64'
            )
            y = paddle.static.data(name="y", shape=[-1, 1], dtype='float32')
            init_hidden = paddle.static.data(
                name="init_hidden", shape=[-1, 1], dtype='float32'
            )
            init_cell = paddle.static.data(
                name="init_cell", shape=[-1, 1], dtype='float32'
            )
            if not paddle.framework.use_pir_api():
                x.desc.set_need_check_feed(False)
                y.desc.set_need_check_feed(False)
                init_hidden.desc.set_need_check_feed(False)
                init_cell.desc.set_need_check_feed(False)

            static_loss, static_last_hidden, static_last_cell = ptb_model(
                x, y, init_hidden, init_cell
            )
            sgd.minimize(static_loss)
            static_param_updated = {}
            static_param_init = {}
            static_param_name_list = []
            static_params = []
            for param in ptb_model.parameters():
                static_param_name_list.append(param.name)
                static_params.append(param)
            if paddle.framework.use_pir_api():
                parameter_mapping = create_parameter_mapping(
                    paddle.static.default_startup_program(),
                    paddle.static.default_main_program(),
                )
                startup_params = [
                    parameter_mapping[param] for param in static_params
                ]
            else:
                startup_params = static_params

            out = exe.run(
                paddle.static.default_startup_program(),
                fetch_list=startup_params,
            )
            for i in range(len(static_params)):
                param_name = static_param_name_list[i]
                static_param_init[param_name] = out[i]
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
                fetch_list.extend(static_params)
                out = exe.run(
                    base.default_main_program(),
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
                        static_param_updated[static_param_name_list[k - 3]] = (
                            out[k]
                        )

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
            np.testing.assert_allclose(
                value, dy_param_updated[key], atol=1e-10, rtol=1e-6
            )


if __name__ == '__main__':
    unittest.main()
