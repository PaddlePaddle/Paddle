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

import os
import random
import re
import unittest

import numpy as np
from get_gpt_model import FakeDataset, generate_model

import paddle
from paddle.distributed.fleet import auto

paddle.enable_static()


def get_cuda_version():
    result = os.popen("nvcc --version").read()
    regex = r'release (\S+),'
    match = re.search(regex, result)
    if match:
        num = str(match.group(1))
        integer, decimal = num.split('.')
        return int(integer) * 1000 + int(float(decimal) * 10)
    else:
        return -1


def apply_pass(use_amp=False, use_master_grad=False, amp_dtype="bfloat16"):
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True

    if use_amp:
        amp = strategy.amp
        amp.enable = True
        amp.dtype = amp_dtype
        amp.level = "o2"
        amp.custom_black_list = [
            'c_softmax_with_cross_entropy',
            'elementwise_div',
            'reduce_sum',
        ]
        if use_master_grad:
            amp.use_master_grad = True

    return strategy


def reset_prog():
    paddle.base.framework.switch_main_program(paddle.static.Program())
    paddle.base.framework.switch_startup_program(paddle.static.Program())


class TestShardingStage2WithNewEXE(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.batch_num = 10
        self.clip_norm = 0.2
        self.dataset = FakeDataset(self.batch_size * self.batch_num)

    def init(self, engine):
        paddle.seed(2022)
        np.random.seed(2022)
        random.seed(2022)
        place = paddle.base.CUDAPlace(paddle.distributed.ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(
        self, use_amp=False, use_master_grad=False, amp_dtype="bfloat16"
    ):
        reset_prog()

        strategy = apply_pass(use_amp, use_master_grad, amp_dtype)
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=0.00001, grad_clip=clip)
        model, loss = generate_model("mp")
        engine = auto.Engine(model, loss, opt, strategy=strategy)
        self.init(engine)
        return engine

    def check_bf16(self, program):
        num_bf16 = 0
        num_fp16 = 0
        num_fp32 = 0

        for p in program.all_parameters():
            if p.dtype == paddle.float32:
                num_fp32 += 1
            if p.dtype == paddle.float16:
                num_fp16 += 1
            if p.dtype == paddle.bfloat16:
                num_bf16 += 1

        self.assertEqual(num_bf16, 26)
        self.assertEqual(num_fp16, 0)
        self.assertEqual(num_fp32, 10)

    def check_fp16(self, program):
        num_bf16 = 0
        num_fp16 = 0
        num_fp32 = 0

        for p in program.all_parameters():
            if p.dtype == paddle.float32:
                num_fp32 += 1
            if p.dtype == paddle.float16:
                num_fp16 += 1
            if p.dtype == paddle.bfloat16:
                num_bf16 += 1

        self.assertEqual(num_bf16, 0)
        self.assertEqual(num_fp16, 26)
        self.assertEqual(num_fp32, 10)

    def test_param_grad_fuse_overlap(self):
        # std
        mp_engine = self.get_engine(use_amp=False)
        mp_history = mp_engine.fit(
            self.dataset,
            3,
            epochs=1,
            steps_per_epoch=self.batch_num,
            log_freq=1,
            batch_size=self.batch_size,
        )
        loss0 = mp_history.history['loss'][0]

        # bf16
        mp_bf16_engine = self.get_engine(use_amp=True)
        if not (
            paddle.amp.is_bfloat16_supported()
            and paddle.device.cuda.get_device_capability()[0] >= 8
        ):
            return

        mp_bf16_history = mp_bf16_engine.fit(
            self.dataset,
            3,
            epochs=1,
            steps_per_epoch=self.batch_num,
            log_freq=1,
            batch_size=self.batch_size,
        )
        loss1 = mp_bf16_history.history['loss'][0]
        np.testing.assert_allclose(loss0, loss1, atol=1e-3, rtol=1e-2)

        self.check_bf16(mp_bf16_engine.main_program)

    def test_master_grad(self):
        # fp16
        mp_fp16_engine = self.get_engine(use_amp=True, amp_dtype="float16")
        if not (paddle.amp.is_float16_supported()):
            return

        mp_fp16_history = mp_fp16_engine.fit(
            self.dataset,
            3,
            epochs=1,
            steps_per_epoch=self.batch_num,
            log_freq=1,
            batch_size=self.batch_size,
        )
        loss1 = mp_fp16_history.history['loss'][0]
        self.check_fp16(mp_fp16_engine.main_program)
        # fp16 + mater_grad
        mp_fp16_mater_grad_engine = self.get_engine(
            use_amp=True, use_master_grad=True, amp_dtype="float16"
        )
        mp_fp16_master_grad_history = mp_fp16_mater_grad_engine.fit(
            self.dataset,
            3,
            epochs=1,
            steps_per_epoch=self.batch_num,
            log_freq=1,
            batch_size=self.batch_size,
        )
        loss2 = mp_fp16_master_grad_history.history['loss'][0]
        np.testing.assert_allclose(loss1, loss2, atol=1e-3, rtol=1e-2)

        self.check_fp16(mp_fp16_mater_grad_engine.main_program)


if __name__ == "__main__":
    unittest.main()
