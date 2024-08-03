# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn.functional as F
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


class Policy(paddle.nn.Layer):
    def __init__(self, input_size):
        super().__init__()

        self.affine1 = paddle.nn.Linear(input_size, 128)
        self.affine2 = paddle.nn.Linear(128, 2)
        self.dropout_ratio = 0.6

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, inputs):
        x = paddle.reshape(inputs, shape=[-1, 4])
        x = self.affine1(x)
        x = paddle.nn.functional.dropout(x, self.dropout_ratio)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return paddle.nn.functional.softmax(action_scores, axis=1)


class TestImperativeMnist(unittest.TestCase):
    def test_mnist_float32(self):
        seed = 90
        epoch_num = 1

        state = np.random.normal(size=4).astype("float32")
        state_list = state.tolist()
        reward = np.random.random(size=[1, 1]).astype("float32")
        reward_list = reward.tolist()
        action_list = [1]
        action = np.array(action_list).astype("float32")
        mask_list = [[0, 1]]
        mask = np.array(mask_list).astype("float32")

        def run_dygraph():
            paddle.seed(seed)
            if paddle.framework.use_pir_api():
                with paddle.pir_utils.OldIrGuard():
                    # Note: dygraph use self.main_program.global_block().create_parameter(), it's need manual seed to old Program
                    paddle.framework.random._manual_program_seed(seed)
                paddle.framework.random._manual_program_seed(seed)
            else:
                paddle.framework.random._manual_program_seed(seed)

            policy = Policy(input_size=4)

            dy_state = paddle.to_tensor(state)
            dy_state.stop_gradient = True
            loss_probs = policy(dy_state)

            dy_mask = paddle.to_tensor(mask)
            dy_mask.stop_gradient = True

            loss_probs = paddle.log(loss_probs)
            loss_probs = paddle.multiply(loss_probs, dy_mask)
            loss_probs = paddle.sum(loss_probs, axis=-1)

            dy_reward = paddle.to_tensor(reward)
            dy_reward.stop_gradient = True

            loss_probs = paddle.multiply(dy_reward, loss_probs)
            loss = paddle.sum(loss_probs)

            sgd = paddle.optimizer.SGD(
                learning_rate=1e-3, parameters=policy.parameters()
            )

            dy_param_init_value = {}

            dy_out = loss.numpy()

            for param in policy.parameters():
                dy_param_init_value[param.name] = param.numpy()

            loss.backward()
            sgd.minimize(loss)
            policy.clear_gradients()

            dy_param_value = {}
            for param in policy.parameters():
                dy_param_value[param.name] = param.numpy()

            return dy_out, dy_param_init_value, dy_param_value

        with base.dygraph.guard():
            dy_out, dy_param_init_value, dy_param_value = run_dygraph()

        with base.dygraph.guard():
            (
                eager_out,
                eager_param_init_value,
                eager_param_value,
            ) = run_dygraph()

        with new_program_scope():
            paddle.seed(seed)
            if paddle.framework.use_pir_api():
                with paddle.pir_utils.OldIrGuard():
                    # Note: dygraph use self.main_program.global_block().create_parameter(), it's need manual seed to old Program
                    paddle.framework.random._manual_program_seed(seed)
                paddle.framework.random._manual_program_seed(seed)
            else:
                paddle.framework.random._manual_program_seed(seed)

            exe = base.Executor(
                base.CPUPlace()
                if not core.is_compiled_with_cuda()
                else base.CUDAPlace(0)
            )

            policy = Policy(input_size=4)

            st_sgd = paddle.optimizer.SGD(learning_rate=1e-3)

            st_state = paddle.static.data(
                name='st_state', shape=[-1, 4], dtype='float32'
            )
            st_reward = paddle.static.data(
                name='st_reward', shape=[-1, 1], dtype='float32'
            )
            st_mask = paddle.static.data(
                name='st_mask', shape=[-1, 2], dtype='float32'
            )

            st_loss_probs = policy(st_state)

            st_loss_probs = paddle.log(st_loss_probs)
            st_loss_probs = paddle.multiply(st_loss_probs, st_mask)
            st_loss_probs = paddle.sum(st_loss_probs, axis=-1)

            st_loss_probs = paddle.multiply(st_reward, st_loss_probs)
            st_loss = paddle.sum(st_loss_probs)

            st_sgd.minimize(st_loss)

            # initialize params and fetch them
            static_param_init_value = {}
            static_param_name_list = []
            static_params = []
            for param in policy.parameters():
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

            for i in range(len(static_param_name_list)):
                param_name = static_param_name_list[i]
                static_param_init_value[param_name] = out[i]

            fetch_list = [st_loss]
            fetch_list.extend(static_params)

            out = exe.run(
                base.default_main_program(),
                feed={"st_state": state, "st_reward": reward, "st_mask": mask},
                fetch_list=fetch_list,
            )

            static_param_value = {}
            static_out = out[0]
            for i in range(1, len(out)):
                static_param_value[static_param_name_list[i - 1]] = out[i]

        # np.testing.assert_allclose(dy_x_data.all(), static_x_data.all(), rtol=1e-5)

        for key, value in static_param_init_value.items():
            self.assertTrue(np.equal(value, dy_param_init_value[key]).all())

        self.assertTrue(np.equal(static_out, dy_out).all())

        for key, value in static_param_value.items():
            self.assertTrue(np.equal(value, dy_param_value[key]).all())

        # check eager
        for key, value in static_param_init_value.items():
            self.assertTrue(np.equal(value, eager_param_init_value[key]).all())

        self.assertTrue(np.equal(static_out, eager_out).all())

        for key, value in static_param_value.items():
            self.assertTrue(np.equal(value, eager_param_value[key]).all())


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
