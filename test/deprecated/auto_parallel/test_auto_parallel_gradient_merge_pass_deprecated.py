# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import random
import unittest

import numpy as np
from auto_parallel_pass_test_base_deprecated import AutoParallelPassTestBase

import paddle
import paddle.nn.functional as F
from paddle import nn, static, utils
from paddle.distributed import fleet
from paddle.distributed.fleet import auto

logging.getLogger().setLevel(logging.INFO)
paddle.enable_static()


class MLPLayer(nn.Layer):
    def __init__(
        self, hidden_size=128, intermediate_size=4 * 128, initializer_range=0.02
    ):
        super().__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        np.random.seed(2021)
        arr0 = np.random.normal(0, 0.02, size=(d_model, dim_feedforward))
        arr1 = np.random.normal(0, 0.02, size=(dim_feedforward, d_model))
        weight_attr0 = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Assign(arr0)
        )
        weight_attr1 = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Assign(arr1)
        )
        bias_attr = None
        self.linear0 = nn.Linear(
            d_model, dim_feedforward, weight_attr0, bias_attr=bias_attr
        )
        self.linear1 = nn.Linear(
            dim_feedforward, d_model, weight_attr1, bias_attr=bias_attr
        )
        self.linear2 = nn.Linear(
            d_model, dim_feedforward, weight_attr0, bias_attr=bias_attr
        )
        self.linear3 = nn.Linear(
            dim_feedforward, d_model, weight_attr1, bias_attr=bias_attr
        )
        self.linear4 = nn.Linear(
            d_model, dim_feedforward, weight_attr0, bias_attr=bias_attr
        )
        self.linear5 = nn.Linear(
            dim_feedforward, d_model, weight_attr1, bias_attr=bias_attr
        )
        self.norm0 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.norm1 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.norm2 = nn.LayerNorm(d_model, epsilon=1e-5)

    def forward(self, input):
        out = self.norm0(input)
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)

        out = self.norm1(out)
        out = self.linear2(out)
        out = F.gelu(out, approximate=True)
        out = self.linear3(out)

        out = self.norm2(out)
        out = self.linear4(out)
        out = F.gelu(out, approximate=True)
        out = self.linear5(out)
        return out


def mlp_forward(input, label, hidden_size):
    auto.shard_tensor(
        input, auto.ProcessMesh([0], dim_names=["x"]), [None, None]
    )
    mlp = MLPLayer(
        hidden_size=hidden_size,
        intermediate_size=4 * hidden_size,
        initializer_range=0.02,
    )
    predict = mlp(input)
    error_cost = paddle.nn.functional.square_error_cost(predict, label)
    loss = paddle.mean(error_cost)
    return loss


class TestGradientMergePass(AutoParallelPassTestBase):
    def init(self):
        paddle.seed(2022)
        random.seed(2022)
        np.random.seed(2022)

    def apply_passes(self):
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.semi_auto = True
        dist_strategy.gradient_merge = True
        dist_strategy.gradient_merge_configs = {"k_steps": 4, "avg": True}
        fleet.init(is_collective=True, strategy=dist_strategy)

    def test_result(self):
        no_pass_rets = self._distributed_launch(
            model=None,
            apply_pass=False,
            gpus=[0],
            batch_size=32,
            hidden_size=128,
            max_step=2,
        )
        pass_rets = self._distributed_launch(
            model=None,
            apply_pass=True,
            gpus=[0],
            batch_size=8,
            hidden_size=128,
            max_step=8,
        )
        # avg loss for gradient_merge pass
        avg_loss = 0
        pass_avg_ret_list = []
        for i, pass_ret in enumerate(pass_rets[0]):
            if (i + 1) % 4 == 0:
                avg_loss += pass_ret[0]
                pass_avg_ret_list.append([avg_loss / 4])
                avg_loss = 0
            else:
                avg_loss += pass_ret[0]

        for no_pass_ret, pass_ret in zip(no_pass_rets[0], pass_avg_ret_list):
            print(f"no_pass_ret={no_pass_ret}, pass_ret={pass_ret}")
            self.assertTrue(
                np.isclose(
                    no_pass_ret,
                    pass_ret,
                    rtol=self.rtol,
                    atol=self.atol,
                    equal_nan=self.equal_nan,
                )
            )

    def get_model(self, place, batch_size, hidden_size, max_step):
        def gen_data():
            for i in range(max_step):
                x_data = input_data[i * batch_size : (i + 1) * batch_size, :]
                y_data = label_data[i * batch_size : (i + 1) * batch_size, :]
                yield x_data, y_data

        train_program = static.Program()
        startup_program = static.Program()
        with static.program_guard(
            train_program, startup_program
        ), utils.unique_name.guard():
            input = static.data(
                name="input", shape=[batch_size, hidden_size], dtype='float32'
            )
            label = static.data(
                name="label", shape=[batch_size, 1], dtype='float32'
            )
            input.stop_gradient = False
            data_holder = [input, label]
            data_loader = paddle.base.io.DataLoader.from_generator(
                feed_list=data_holder, capacity=70, iterable=False
            )
            data_loader.set_batch_generator(
                gen_data, paddle.static.cuda_places()
            )

            loss = mlp_forward(input, label, hidden_size)

        optimizer = paddle.optimizer.Adam(learning_rate=0.01)
        optimizer = fleet.distributed_optimizer(optimizer)
        (
            _,
            self._params_grads,
            dist_startup_prog,
            dist_main_prog,
        ) = optimizer.minimize(loss, startup_program)

        input_data = np.random.random(size=(128, hidden_size)).astype('float32')
        label_data = np.random.random(size=(128, 1)).astype('float32')

        return (
            dist_main_prog,
            dist_startup_prog,
            [input, label],
            [loss],
            data_loader,
        )


if __name__ == "__main__":
    unittest.main()
