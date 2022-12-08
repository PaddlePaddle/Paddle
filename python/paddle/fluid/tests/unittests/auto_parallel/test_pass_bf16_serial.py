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
from paddle.fluid.dygraph.parallel import ParallelEnv

paddle.enable_static()


def apply_pass(use_bf16=False):
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True
    if use_bf16:
        print("use_bf16")
        amp = strategy.amp
        amp.enable = True
        amp.enable_bf16 = True
        amp.custom_bf16_list = ['softmax', 'layer_norm', 'gelu']
        amp.custom_fp32_list = [
            'c_softmax_with_cross_entropy',
            'elementwise_div',
            'reduce_sum',
            'reshape2',
        ]
    else:
        print("don not use_bf16")
        amp = strategy.amp
        amp.enable = False
        amp.enable_bf16 = False
    return strategy


def reset_prog():
    paddle.fluid.framework.switch_main_program(paddle.static.Program())
    paddle.fluid.framework.switch_startup_program(paddle.static.Program())


class TestAMPPass(unittest.TestCase):
    def setUp(self):
        self.rtol = 1e-5
        self.atol = 1e-8
        self.batch_size = 1
        self.batch_num = 10
        self.clip_norm = 0.2
        self.dataset = FakeDataset(self.batch_size * self.batch_num)

    def init(self, engine):
        paddle.seed(2021)
        np.random.seed(2021)
        random.seed(2021)
        place = paddle.fluid.CUDAPlace(ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(self, use_bf16=False):
        reset_prog()

        strategy = apply_pass(use_bf16)
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=0.00001, grad_clip=clip)
        model, loss = generate_model("serial")

        engine = auto.Engine(model, loss, opt, strategy=strategy)
        self.init(engine)
        return engine

    def check_results(self, ref_losses, check_losses, rtol=None, atol=None):
        np.testing.assert_allclose(
            ref_losses,
            check_losses,
            rtol=rtol or self.rtol,
            atol=atol or self.atol,
            err_msg='pass {} has wrong results!, \nu={}\nv={}\ndiff={}'.format(
                __class__, ref_losses, check_losses, ref_losses - check_losses
            ),
        )

    def test_bf16_pass(self):
        # mp2 training
        serial_engine = self.get_engine()
        history = serial_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        serial_losses = np.array(history.history["loss"])

        # mp2 amp-o1 training
        bf16_o1_engine = self.get_engine(True)
        history = bf16_o1_engine.fit(
            self.dataset, 3, batch_size=self.batch_size
        )
        # file = open(
        #     "/workspace/Paddle/gpt_serial_main.log",
        #     "w",
        # )
        # print(serial_engine._dist_main_progs["train"][0], file=file)
        # file.close()
        bf16_o1_losses = np.array(history.history["loss"])
        bf16_o1_engine.evaluate(self.dataset, 3, batch_size=self.batch_size)
        self.check_results(serial_losses, bf16_o1_losses)


if __name__ == "__main__":
    unittest.main()
