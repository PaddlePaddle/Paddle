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

import random
import unittest

import numpy as np
from get_gpt_model import FakeDataset, generate_model

import paddle
from paddle.distributed.fleet import auto

paddle.enable_static()


def apply_pass(use_sharding=False, stage=None):
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"
    # strategy.reinit = True
    if use_sharding:
        sharding = strategy.sharding
        sharding.enable = True
        sharding.degree = 2
        sharding.stage = stage

    amp = strategy.amp
    amp.enable = True
    amp.dtype = "float16"
    amp.level = "o1"
    amp.custom_black_list = [
        'c_softmax_with_cross_entropy',
        'elementwise_div',
        'reduce_sum',
    ]

    return strategy


def reset_prog():
    paddle.base.framework.switch_main_program(paddle.static.Program())
    paddle.base.framework.switch_startup_program(paddle.static.Program())
    paddle.utils.unique_name.switch()


class TestShardingPass(unittest.TestCase):
    def setUp(self):
        self.rtol = 1e-6
        self.atol = 1e-8
        self.batch_size = 2
        self.batch_num = 10
        self.clip_norm = 0.2
        self.dataset = FakeDataset(self.batch_size * self.batch_num)

    def init(self, engine):
        paddle.seed(2022)
        np.random.seed(2022)
        random.seed(2022)

    def get_engine(self, use_sharding=False, stage=None):
        reset_prog()

        strategy = apply_pass(use_sharding, stage)
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        # NOTE: seting opt = paddle.optimizer.AdamW(learning_rate=0.00001, grad_clip=clip) will cause precision problem
        opt = paddle.optimizer.AdamW(learning_rate=0.00001)
        model, loss = generate_model("dp")

        engine = auto.Engine(model, loss, opt, strategy=strategy)
        self.init(engine)
        return engine

    def check_results(self, ref_losses, check_losses):
        np.testing.assert_equal(
            ref_losses,
            check_losses,
            err_msg='pass {} has wrong results!, \nu={}\nv={}\ndiff={}'.format(
                __class__, ref_losses, check_losses, ref_losses - check_losses
            ),
        )

    def test_sharding_pass(self):
        # dp2 training
        dp_engine = self.get_engine()
        input_spec = [
            paddle.static.InputSpec([self.batch_size, 512], 'int64', 'tokens'),
            paddle.static.InputSpec(
                [self.batch_size, 512], 'int64', 'position_ids'
            ),
            paddle.static.InputSpec(
                [self.batch_size, 1, 512, 512], 'float32', 'attention_mask'
            ),
        ]
        label_spec = [
            paddle.static.InputSpec([self.batch_size, 512], 'int64', 'label'),
            paddle.static.InputSpec(
                [self.batch_size, 512], 'float32', 'loss_mask'
            ),
        ]
        dp_engine.prepare(
            inputs_spec=input_spec, labels_spec=label_spec, mode='train'
        )
        dp_engine.save("./dp_engine", training=True)
        history = dp_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        dp_losses = np.array(history.history["loss"])

        # dp2 training after load dp_engine
        dp_load_engine = self.get_engine()
        dp_load_engine.load("./dp_engine")
        history = dp_load_engine.fit(
            self.dataset, 3, batch_size=self.batch_size
        )
        dp_load_losses2 = np.array(history.history["loss"])
        self.check_results(dp_losses, dp_load_losses2)

        # sharding2 stage1 training
        sharding1_engine = self.get_engine(True, 1)
        sharding1_engine.load("./dp_engine")
        history = sharding1_engine.fit(
            self.dataset, 3, batch_size=self.batch_size
        )
        sharding1_losses = np.array(history.history["loss"])
        self.check_results(dp_losses, sharding1_losses)

        # sharding2 stage2 training
        sharding2_engine = self.get_engine(True, 2)
        sharding2_engine.load("./dp_engine")
        history = sharding2_engine.fit(
            self.dataset, 3, batch_size=self.batch_size
        )
        sharding2_losses = np.array(history.history["loss"])
        self.check_results(dp_losses, sharding2_losses)

        # sharding2 stage3 training
        sharding3_engine = self.get_engine(True, 3)
        sharding3_engine.load("./dp_engine")
        history = sharding3_engine.fit(
            self.dataset, 3, batch_size=self.batch_size
        )
        sharding3_losses = np.array(history.history["loss"])
        self.check_results(dp_losses, sharding3_losses)


if __name__ == "__main__":
    unittest.main()
