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


def apply_pass(use_recompute=False, no_recompute_segments=[]):
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True
    if use_recompute:
        recompute = strategy.recompute
        recompute.enable = True
        recompute.no_recompute_segments = no_recompute_segments
    return strategy


def reset_prog():
    paddle.fluid.framework.switch_main_program(paddle.static.Program())
    paddle.fluid.framework.switch_startup_program(paddle.static.Program())


class TestRandomControl(unittest.TestCase):
    def setUp(self):
        self.rtol = 1e-6
        self.atol = 1e-8
        self.batch_size = 1
        self.batch_num = 10
        self.clip_norm = 0.2
        self.dataset = FakeDataset(self.batch_size * self.batch_num)
        paddle.distributed.auto_parallel.parallel_manual_seed(100)

    def init(self, engine):
        paddle.seed(2022)
        np.random.seed(2022)
        random.seed(2022)
        place = paddle.fluid.CUDAPlace(paddle.distributed.ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(self, use_recompute=False, no_recompute_segments=[]):
        reset_prog()

        strategy = apply_pass(use_recompute, no_recompute_segments)
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=0.00001, grad_clip=clip)
        model, loss = generate_model("mp", dropout_prob=0.1)

        engine = auto.Engine(model, loss, opt, strategy=strategy)
        self.init(engine)
        return engine

    def check_results(self, ref_losses, check_losses):
        np.testing.assert_allclose(
            ref_losses,
            check_losses,
            rtol=self.rtol,
            atol=self.atol,
            err_msg='pass {} has wrong results!, \nu={}\nv={}\ndiff={}'.format(
                __class__, ref_losses, check_losses, ref_losses - check_losses
            ),
        )

    def test_random_ctrl_vanilla(self):
        # mp2 recompute training
        rc_engine = self.get_engine(False)
        history = rc_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        rc_losses = np.array(history.history["loss"])

        # check program
        ops = rc_engine.main_program.global_block().ops
        rng_names = []
        seed_var_names = []
        for op in ops:
            if op.type == "seed":
                rng_names.append(op.attr('rng_name'))
            if op.type == "dropout":
                seed_var_names.append(op.input("Seed")[0])

        self.assertEqual(
            rng_names,
            [
                'mesh:1_dim0:-1',
                'mesh:1_dim0:0',
                'mesh:1_dim0:-1',
                'mesh:1_dim0:-1',
                'mesh:1_dim0:0',
                'mesh:1_dim0:-1',
                'mesh:1_dim0:-1',
            ],
        )
        self.assertEqual(
            seed_var_names,
            [
                'rc_seed_0.tmp_0',
                'rc_seed_1.tmp_0',
                'rc_seed_2.tmp_0',
                'rc_seed_3.tmp_0',
                'rc_seed_4.tmp_0',
                'rc_seed_5.tmp_0',
                'rc_seed_6.tmp_0',
            ],
        )

    def test_random_ctrl_with_recompute(self):
        # mp2 recompute training
        rc_engine = self.get_engine(True)
        history = rc_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        rc_losses = np.array(history.history["loss"])

        # check program
        ops = rc_engine.main_program.global_block().ops
        rng_names = []
        seed_var_names = []
        for op in ops:
            if op.type == "seed":
                rng_names.append(op.attr('rng_name'))
            if op.type == "dropout":
                seed_var_names.append(op.input("Seed")[0])

        self.assertEqual(
            rng_names,
            [
                'mesh:1_dim0:-1',
                'mesh:1_dim0:0',
                'mesh:1_dim0:-1',
                'mesh:1_dim0:-1',
                'mesh:1_dim0:0',
                'mesh:1_dim0:-1',
                'mesh:1_dim0:-1',
            ],
        )
        self.assertEqual(
            seed_var_names,
            [
                'rc_seed_0.tmp_0',
                'rc_seed_1.tmp_0',
                'rc_seed_2.tmp_0',
                'rc_seed_3.tmp_0',
                'rc_seed_4.tmp_0',
                'rc_seed_5.tmp_0',
                'rc_seed_6.tmp_0',
                'rc_seed_4.tmp_0',
                'rc_seed_5.tmp_0',
                'rc_seed_6.tmp_0',
                'rc_seed_0.tmp_0',
                'rc_seed_1.tmp_0',
                'rc_seed_2.tmp_0',
                'rc_seed_3.tmp_0',
            ],
        )


if __name__ == "__main__":
    unittest.main()
