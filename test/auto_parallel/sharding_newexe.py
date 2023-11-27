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


def apply_pass(use_sharding=False, use_amp=False, use_recompute=False):
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True

    if use_sharding:
        sharding = strategy.sharding
        sharding.enable = True
        sharding.degree = 2
        sharding.stage = 2
        sharding.enable_overlap = True
        sharding.param_comm_stream_num = 2
        sharding.grad_comm_stream_num = 2
        sharding.param_bucket_size_numel = 512 * 512
        sharding.grad_bucket_size_numel = 128 * 128
        sharding.partition_algor = 'use_order'
    if use_recompute:
        recompute = strategy.recompute
        recompute.enable = True
    if use_amp:
        amp = strategy.amp
        amp.enable = True
        amp.custom_white_list = [
            'lookup_table_v2',
            'lookup_table',
            'softmax',
            'layer_norm',
            'gelu',
        ]
        amp.custom_black_list = [
            'c_softmax_with_cross_entropy',
            'elementwise_div',
            'reduce_sum',
        ]
        amp.init_loss_scaling = 32768
        amp.use_fp16_guard = False
        amp.use_pure_fp16 = True
        amp.use_optimizer_fp16 = False

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
        self, use_sharding=False, use_amp=False, use_recompute=False
    ):
        reset_prog()

        strategy = apply_pass(use_sharding, use_amp, use_recompute)
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=0.00001, grad_clip=clip)
        model, loss = generate_model("dp")
        engine = auto.Engine(model, loss, opt, strategy=strategy)
        self.init(engine)
        return engine

    def check_param_grad_fuse_overlap(self, program):
        num_op = 0
        num_coalesce = 0
        num_reduce = 0
        num_broadcast = 0
        for op in program.global_block().ops:
            if op.type == "nop" or op.type == "depend":
                num_op += 1
            elif op.type == "coalesce_tensor":
                num_coalesce += 1
            elif op.type == "c_reduce_sum":
                num_reduce += 1
            elif op.type == "c_broadcast":
                num_broadcast += 1

        if paddle.distributed.get_rank() == 0:
            self.assertEqual(num_op, 22)
        else:
            self.assertEqual(num_op, 54)

        self.assertEqual(num_coalesce, 5)
        self.assertEqual(num_reduce, 14)
        self.assertEqual(num_broadcast, 2)

    def test_param_grad_fuse_overlap(self):
        # dp2
        dp_engine = self.get_engine()
        dp_history = dp_engine.fit(
            self.dataset,
            3,
            epochs=1,
            steps_per_epoch=self.batch_num,
            log_freq=1,
            batch_size=self.batch_size,
        )
        dp_loss = dp_history.history['loss'][0]

        # sharding2
        sharding_engine = self.get_engine(use_sharding=True)
        sharding_history = sharding_engine.fit(
            self.dataset,
            3,
            epochs=1,
            steps_per_epoch=self.batch_num,
            log_freq=1,
            batch_size=self.batch_size,
        )
        sharding_loss = sharding_history.history['loss'][0]

        # amp, recompute
        amp_recompute_engine = self.get_engine(
            use_sharding=False, use_amp=True, use_recompute=True
        )
        amp_recompute_history = amp_recompute_engine.fit(
            self.dataset,
            3,
            epochs=1,
            steps_per_epoch=self.batch_num,
            log_freq=1,
            batch_size=self.batch_size,
        )
        amp_recompute_loss = amp_recompute_history.history['loss'][0]

        # sharding2, amp, recompute
        all_engine = self.get_engine(
            use_sharding=True, use_amp=True, use_recompute=True
        )
        all_history = all_engine.fit(
            self.dataset,
            3,
            epochs=1,
            steps_per_epoch=self.batch_num,
            log_freq=1,
            batch_size=self.batch_size,
        )
        all_loss = all_history.history['loss'][0]

        self.check_param_grad_fuse_overlap(sharding_engine.main_program)
        np.testing.assert_allclose(
            dp_loss, sharding_loss, rtol=1e-05, atol=1e-08
        )
        np.testing.assert_allclose(
            amp_recompute_loss, all_loss, rtol=1e-05, atol=1e-08
        )


if __name__ == "__main__":
    unittest.main()
