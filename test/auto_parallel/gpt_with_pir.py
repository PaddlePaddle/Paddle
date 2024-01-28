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
from test_sparse_addmm_op import get_cuda_version

import paddle
from paddle.distributed import ParallelEnv
from paddle.distributed.fleet import auto

paddle.enable_static()


def apply_pass(use_sharding=False, pipeline_mode=None, fuse_passes_list=None):
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True

    amp = strategy.amp
    amp.enable = True
    amp.dtype = "float16"
    amp.level = "o2"
    amp.custom_white_list = ['softmax', 'layer_norm', 'gelu']
    amp.custom_black_list = [
        'c_softmax_with_cross_entropy',
        'elementwise_div',
        'reduce_sum',
    ]

    recompute = strategy.recompute
    recompute.enable = True

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


class TestPir(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.batch_num = 5
        self.clip_norm = 0.2
        self.dataset = FakeDataset(self.batch_size * self.batch_num)
        os.environ['FLAGS_new_executor_micro_batching'] = 'True'
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
        use_sharding=False,
        pipeline_mode=None,
        fuse_passes_list=None,
    ):
        reset_prog()

        paddle.set_default_dtype('float32')

        strategy = apply_pass(use_sharding, pipeline_mode, fuse_passes_list)
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
            err_msg='pass {} has wrong results!, \nu={}\nv={}\ndiff={}'.format(
                __class__, ref_losses, check_losses, ref_losses - check_losses
            ),
        )

    def enable_pir(self, flag):
        paddle.set_flags({'FLAGS_enable_pir_in_executor': flag})  # for c++
        os.environ['FLAGS_enable_pir_in_executor'] = str(flag)  # for python

    def test_dp(self):
        self.enable_pir(False)
        engine_dp_prog = self.get_engine(
            "dp", name="dp_prog", use_sharding=True
        )
        out_dp_prog = engine_dp_prog.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )

        self.enable_pir(True)
        engine_dp_ir = self.get_engine("dp", name="dp_pir", use_sharding=True)
        out_dp_ir = engine_dp_ir.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )

        self.check_results(
            out_dp_prog.history["loss"][0], out_dp_ir.history["loss"][0]
        )

    def test_dp_with_fused_linear(self):
        if not get_cuda_version() >= 11060:
            return

        self.enable_pir(False)
        engine_dp_prog = self.get_engine(
            "dp",
            name="dp_prog_fuse_linear",
            fuse_passes_list=['fuse_gemm_epilogue'],
        )
        out_dp_prog = engine_dp_prog.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )

        self.enable_pir(True)
        engine_dp_ir = self.get_engine(
            "dp",
            name="dp_pir_fuse_linear",
            use_sharding=True,
            fuse_passes_list=['fused_gemm_epilogue_pass'],
        )
        out_dp_ir = engine_dp_ir.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )

        self.check_results(
            out_dp_prog.history["loss"][0], out_dp_ir.history["loss"][0]
        )

    def test_mp(self):
        self.enable_pir(False)
        engine_mp_prog = self.get_engine("mp", name="mp_prog")
        out_mp_prog = engine_mp_prog.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )

        self.enable_pir(True)
        engine_mp_ir = self.get_engine("mp", name="mp_pir")
        out_mp_ir = engine_mp_ir.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )

        self.check_results(
            out_mp_prog.history["loss"][0], out_mp_ir.history["loss"][0]
        )

    def test_pp(self):
        # naive pipeline parallel without schedule
        self.enable_pir(False)
        engine_pp_prog = self.get_engine("pp", name="pp_prog0")
        out_pp_prog = engine_pp_prog.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )

        self.enable_pir(True)
        # send_v2/recv_v2 dynamic_shape is True
        engine_pp_ir = self.get_engine("pp", name="pp_pir")
        out_pp_ir = engine_pp_ir.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )

        if paddle.distributed.get_rank() == 1:
            self.check_results(
                out_pp_prog.history["loss"][0], out_pp_ir.history["loss"][0]
            )

        # send_v2/recv_v2 dynamic_shape is False
        engine_pp_prog1 = self.get_engine("pp", name="pp_prog1")
        dataloader_pp_prog = engine_pp_prog1.dataloader(
            self.dataset,
            batch_size=self.batch_size,
            sample_split=3,
            mode="train",
        )
        engine_pp_prog1.prepare(mode="train")
        for op in engine_pp_prog1.main_program.global_block().ops:
            if op.type in ["send_v2", "recv_v2"]:
                op.desc._set_attr("dynamic_shape", False)
        for data in dataloader_pp_prog:
            out_pp_prog1 = engine_pp_prog1.run(data, mode="train")

        if paddle.distributed.get_rank() == 1:
            self.check_results(
                out_pp_prog1["loss"], out_pp_ir.history["loss"][0]
            )

    def test_pp_1f1b(self):
        self.enable_pir(False)
        engine_1f1b_prog = self.get_engine(
            "pp", name="1f1b_prog", use_sharding=False, pipeline_mode="1F1B"
        )
        out_1f1b_prog = engine_1f1b_prog.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )

        self.enable_pir(True)
        engine_1f1b_ir = self.get_engine(
            "pp", name="1f1b_pir", use_sharding=False, pipeline_mode="1F1B"
        )
        out_1f1b_ir = engine_1f1b_ir.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )

        if paddle.distributed.get_rank() == 1:
            self.check_results(
                out_1f1b_prog.history["loss"][0],
                out_1f1b_ir.history["loss"][0],
            )

    def test_pp_fthenb(self):
        self.enable_pir(False)
        engine_fthenb_prog = self.get_engine(
            "pp", name="fthenb_prog", use_sharding=False, pipeline_mode="FThenB"
        )
        out_fthenb_prog = engine_fthenb_prog.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )

        self.enable_pir(True)
        engine_fthenb_ir = self.get_engine(
            "pp",
            name="fthenb_pir",
            use_sharding=False,
            pipeline_mode="FThenB",
        )
        out_fthenb_ir = engine_fthenb_ir.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )
        if paddle.distributed.get_rank() == 1:
            self.check_results(
                out_fthenb_prog.history["loss"][0],
                out_fthenb_ir.history["loss"][0],
            )


if __name__ == "__main__":
    unittest.main()
