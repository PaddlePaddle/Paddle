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


def apply_pass(use_amp=False, level=None):
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True
    if use_amp:
        amp = strategy.amp
        amp.enable = True
        amp.dtype = "float16"
        amp.level = level
        amp.custom_white_list = ['softmax', 'layer_norm', 'gelu']
        amp.custom_black_list = [
            'c_softmax_with_cross_entropy',
            'elementwise_div',
            'reduce_sum',
        ]
        amp.init_loss_scaling = 32768
        amp.use_fp16_guard = False
        print("amp level: ", level)
    return strategy


def reset_prog():
    paddle.base.framework.switch_main_program(paddle.static.Program())
    paddle.base.framework.switch_startup_program(paddle.static.Program())


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
        place = paddle.base.CUDAPlace(paddle.distributed.ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(self, use_amp=False, level=None):
        reset_prog()

        strategy = apply_pass(use_amp, level)
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=0.00001, grad_clip=clip)
        model, loss = generate_model("mp")

        engine = auto.Engine(model, loss, opt, strategy=strategy)
        self.init(engine)
        return engine

    def check_results(self, ref_losses, check_losses, rtol=None, atol=None):
        np.testing.assert_allclose(
            ref_losses,
            check_losses,
            rtol=rtol or self.rtol,
            atol=atol or self.atol,
            err_msg=f'pass {__class__} has wrong results!, \nu={ref_losses}\nv={check_losses}\ndiff={ref_losses - check_losses}',
        )

    def test_amp_pass(self):
        # mp2 training
        mp_engine = self.get_engine()
        history = mp_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        mp_losses = np.array(history.history["loss"])

        # mp2 amp-o1 training
        amp_o1_engine = self.get_engine(True, "o1")
        history = amp_o1_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        amp_o1_losses = np.array(history.history["loss"])
        amp_o1_engine.evaluate(self.dataset, 3, batch_size=self.batch_size)
        # self.check_results(mp_losses, amp_o1_losses)

        # mp2 amp-o2 training
        amp_o2_engine = self.get_engine(True, "o2")
        history = amp_o2_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        amp_o2_losses = np.array(history.history["loss"])
        amp_o2_engine.evaluate(self.dataset, 3, batch_size=self.batch_size)
        # self.check_results(mp_losses, amp_o2_losses)

        # mp2 amp-o3 training
        amp_o3_engine = self.get_engine(True, "o3")
        history = amp_o3_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        amp_o3_losses = np.array(history.history["loss"])
        amp_o3_engine.evaluate(self.dataset, 3, batch_size=self.batch_size)
        # self.check_results(mp_losses, amp_o3_losses)


if __name__ == "__main__":
    unittest.main()
