# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import random
import numpy as np
import paddle

import paddle.distributed.fleet as fleet
import paddle.distributed.auto_parallel as auto

from paddle.distributed.auto_parallel.engine import Engine
from get_gpt_model import generate_model, create_data_holder, FakeDataset

paddle.enable_static()


def apply_pass(use_sharding=False):
    strategy = fleet.DistributedStrategy()
    strategy.semi_auto = True
    if use_sharding:
        strategy.sharding = True
        strategy.sharding_configs = {
            "sharding_degree": 2,
            "stage": 2,
        }
    return strategy


def get_parameter_value(program):
    from paddle.fluid.framework import Parameter

    def is_parameter(var):
        return isinstance(var, Parameter)

    def get_tensor(var):
        t = paddle.fluid.global_scope().find_var(var.name).get_tensor()
        return np.array(t)

    def get_name(var):
        return len(var.name)

    parameters_list = list(filter(is_parameter, program.list_vars()))
    parameters_value = []
    for p in sorted(parameters_list, key=get_name):
        parameters_value.append(get_tensor(p))
    return parameters_value


def reset_prog():
    paddle.fluid.framework.switch_main_program(paddle.static.Program())
    paddle.fluid.framework.switch_startup_program(paddle.static.Program())


class TestGradientClipByGlobalNorm(unittest.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.batch_num = 1
        self.clip_norm = 0.2
        self.dataset = FakeDataset(self.batch_size * self.batch_num)

    def init(self, engine):
        paddle.seed(2022)
        np.random.seed(2022)
        random.seed(2022)
        engine.mode = "train"
        engine._executor.run(engine.startup_program)

    def get_dp2_engine(self):
        reset_prog()

        strategy = apply_pass()
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=0.00001, grad_clip=clip)
        model, loss = generate_model("dp")
        inputs_spec, labels_spec = create_data_holder(self.batch_size)

        engine = Engine(model, inputs_spec, labels_spec, strategy=strategy)
        engine.prepare(optimizer=opt, loss=loss)
        self.init(engine)
        return engine

    def get_dp2sharding2_engine(self):
        reset_prog()

        strategy = apply_pass(True)
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=0.00001, grad_clip=clip)
        model, loss = generate_model("dp")
        inputs_spec, labels_spec = create_data_holder(self.batch_size)

        engine = Engine(model, inputs_spec, labels_spec, strategy=strategy)
        engine.prepare(optimizer=opt, loss=loss)
        self.init(engine)
        return engine

    def check_result(self, dp_params, sharding_params):
        assert len(dp_params) == len(sharding_params)
        for dp_p, sharding_p in zip(dp_params, sharding_params):
            np.testing.assert_allclose(
                dp_p,
                sharding_p,
                rtol=1e-05,
                atol=1e-08,
                err_msg=
                'gradient clip by global norm has wrong results!, \nu={}\nv={}\ndiff={}'
                .format(dp_p, sharding_p, dp_p - sharding_p))

    def test_grad_clip(self):
        # dp2 training
        dp_engine = self.get_dp2_engine()
        dp_engine.fit(self.dataset, batch_size=self.batch_size, use_cache=True)
        dp_param_values = get_parameter_value(dp_engine.main_program)

        # dp2sharding2 training
        sharding_engine = self.get_dp2sharding2_engine()
        sharding_engine.fit(self.dataset,
                            batch_size=self.batch_size,
                            use_cache=True)
        sharding_param_values = get_parameter_value(
            sharding_engine.main_program)

        self.check_result(dp_param_values, sharding_param_values)


if __name__ == "__main__":
    unittest.main()
