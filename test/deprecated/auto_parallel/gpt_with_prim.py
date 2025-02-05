# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np
from get_gpt_model import FakeDataset, generate_model

import paddle
from paddle.distributed import ParallelEnv
from paddle.distributed.fleet import auto

paddle.enable_static()


def apply_pass(
    use_recompute=False,
    use_amp=False,
    use_sharding=False,
    pipeline_mode=None,
    fuse_passes_list=None,
):
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True

    recompute = strategy.recompute
    if use_recompute:
        recompute.enable = True
    else:
        recompute.enable = False

    amp = strategy.amp
    if use_amp:
        amp.enable = True
        amp.dtype = "float16"
        amp.level = "o2"
        amp.custom_white_list = ['softmax', 'layer_norm', 'gelu']
        amp.custom_black_list = [
            'c_softmax_with_cross_entropy',
            'elementwise_div',
            'reduce_sum',
        ]
    else:
        amp.enable = False

    if use_sharding:
        sharding = strategy.sharding
        sharding.enable = True
        sharding.degree = 2
        sharding.stage = 2

    if pipeline_mode:
        pipeline = strategy.pipeline
        pipeline.enable = True
        pipeline.schedule_mode = pipeline_mode
        pipeline.accumulate_steps = 2

    if fuse_passes_list:
        fused_passes = strategy.fused_passes
        fused_passes.enable = True
        fused_passes.fused_passes_list = fuse_passes_list

    return strategy


def reset_prog():
    paddle.framework.switch_main_program(paddle.static.Program())
    paddle.framework.switch_startup_program(paddle.static.Program())
    paddle.utils.unique_name.switch()


class TestPrim(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.batch_num = 5
        self.clip_norm = 0.2
        self.dataset = FakeDataset(self.batch_size * self.batch_num)
        paddle.set_flags({'FLAGS_embedding_deterministic': 1})
        paddle.set_flags({'FLAGS_cudnn_deterministic': 1})

    def init(self, engine, name):
        paddle.seed(2021)
        np.random.seed(2021)
        random.seed(2021)
        paddle.distributed.fleet.init(is_collective=True)
        paddle.distributed.auto_parallel.random._rng_name_to_seed.clear()
        paddle.distributed.auto_parallel.random._inited_rng_name_to_seed.clear()
        paddle.distributed.auto_parallel.parallel_manual_seed(2021, name)
        place = paddle.CUDAPlace(ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(
        self,
        mode,
        name,
        use_recompute=False,
        use_amp=False,
        use_sharding=False,
        pipeline_mode=None,
        fuse_passes_list=None,
    ):
        reset_prog()

        paddle.set_default_dtype('float32')

        strategy = apply_pass(
            use_recompute,
            use_amp,
            use_sharding,
            pipeline_mode,
            fuse_passes_list,
        )
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=0.00001, grad_clip=clip)
        model, loss = generate_model(mode, dropout_prob=0.1)

        engine = auto.Engine(model, loss, opt, strategy=strategy)
        self.init(engine, name)
        return engine

    def check_results(self, ref_losses, check_losses):
        np.testing.assert_equal(
            ref_losses,
            check_losses,
            err_msg=f'pass {__class__} has wrong results!, \nu={ref_losses}\nv={check_losses}\ndiff={ref_losses - check_losses}',
        )

    def check_results_prim(self, ref_losses, check_losses):
        np.testing.assert_allclose(
            ref_losses,
            check_losses,
            rtol=2e-2,
            atol=2e-2,
            err_msg=f'pass {__class__} has wrong results!, \nu={ref_losses}\nv={check_losses}\ndiff={ref_losses - check_losses}',
        )

    def enable_pir(self, flag):
        paddle.set_flags({'FLAGS_enable_pir_in_executor': flag})  # for c++
        os.environ['FLAGS_enable_pir_in_executor'] = str(flag)  # for python

    def enable_prim_in_dist(self, flag):
        os.environ['FLAGS_enable_prim_after_distribute'] = str(
            flag
        )  # for python

    def test_dp(self):
        self.enable_pir(True)
        engine_dp_pir = self.get_engine("dp", name="dp_pir", use_sharding=True)
        out_dp_pir = engine_dp_pir.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )

        # test prim enabled distributed engine
        self.enable_prim_in_dist(True)
        engine_dp_pir_prim = self.get_engine(
            "dp", name="dp_pir_prim", use_sharding=True
        )
        dataloader_dp_pir_prim = engine_dp_pir_prim.dataloader(
            self.dataset,
            batch_size=self.batch_size,
            sample_split=3,
            mode="train",
        )
        engine_dp_pir_prim.prepare(mode="train")
        for data in dataloader_dp_pir_prim:
            out_dp_pir_prim = engine_dp_pir_prim.run(data, mode="train")

        if paddle.distributed.get_rank() == 1:
            self.check_results_prim(
                out_dp_pir_prim["loss"], out_dp_pir.history["loss"][0]
            )
        self.enable_prim_in_dist(False)

    def test_mp(self):
        self.enable_pir(True)
        engine_mp_pir = self.get_engine("mp", name="mp_pir")
        out_mp_pir = engine_mp_pir.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )

        # test prim enabled distributed engine
        self.enable_prim_in_dist(True)
        engine_mp_pir_prim = self.get_engine("mp", name="mp_pir_prim")
        dataloader_mp_pir_prim = engine_mp_pir_prim.dataloader(
            self.dataset,
            batch_size=self.batch_size,
            sample_split=3,
            mode="train",
        )
        engine_mp_pir_prim.prepare(mode="train")
        for data in dataloader_mp_pir_prim:
            out_mp_pir_prim = engine_mp_pir_prim.run(data, mode="train")

        if paddle.distributed.get_rank() == 1:
            self.check_results_prim(
                out_mp_pir_prim["loss"], out_mp_pir.history["loss"][0]
            )
        self.enable_prim_in_dist(False)

    def test_amp(self):
        self.enable_pir(True)
        engine_amp_pir = self.get_engine(
            "dp", name="amp_pir", use_amp=True, use_sharding=True
        )
        out_amp_pir = engine_amp_pir.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )

        # test prim enabled distributed engine
        self.enable_prim_in_dist(True)
        engine_amp_pir_prim = self.get_engine(
            "dp", name="amp_pir_prim", use_amp=True, use_sharding=True
        )
        dataloader_amp_pir_prim = engine_amp_pir_prim.dataloader(
            self.dataset,
            batch_size=self.batch_size,
            sample_split=3,
            mode="train",
        )
        engine_amp_pir_prim.prepare(mode="train")
        for data in dataloader_amp_pir_prim:
            out_amp_pir_prim = engine_amp_pir_prim.run(data, mode="train")

        if paddle.distributed.get_rank() == 1:
            self.check_results_prim(
                out_amp_pir_prim["loss"], out_amp_pir.history["loss"][0]
            )
        self.enable_prim_in_dist(False)


if __name__ == "__main__":
    unittest.main()
