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


def apply_pass():
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True
    return strategy


def reset_prog():
    paddle.framework.switch_main_program(paddle.static.Program())
    paddle.framework.switch_startup_program(paddle.static.Program())
    paddle.utils.unique_name.switch()


class TestNewIR(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.batch_num = 5
        self.clip_norm = 0.2
        self.dataset = FakeDataset(self.batch_size * self.batch_num)
        os.environ['FLAGS_new_executor_micro_batching'] = 'True'
        paddle.set_flags({'FLAGS_embedding_deterministic': 1})
        paddle.set_flags({'FLAGS_cudnn_deterministic': 1})

    def init(self, engine):
        paddle.seed(2021)
        np.random.seed(2021)
        random.seed(2021)
        paddle.distributed.fleet.init(is_collective=True)
        place = paddle.CUDAPlace(ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(self, mode):
        reset_prog()

        strategy = apply_pass()
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=0.00001, grad_clip=None)
        model, loss = generate_model(mode)

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

    def enable_new_ir(self, flag):
        paddle.set_flags({'FLAGS_enable_new_ir_in_executor': flag})  # for c++
        os.environ['FLAGS_enable_new_ir_in_executor'] = str(flag)  # for python

    def test_dp(self):
        self.enable_new_ir(False)
        engine_dp_prog = self.get_engine("dp")
        out_dp_prog = engine_dp_prog.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )

        self.enable_new_ir(True)
        engine_dp_ir = self.get_engine("dp")
        out_dp_ir = engine_dp_ir.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )

        self.check_results(
            out_dp_prog.history["loss"][0], out_dp_ir.history["loss"][0]
        )

    def test_mp(self):
        self.enable_new_ir(False)
        engine_mp_prog = self.get_engine("mp")
        out_mp_prog = engine_mp_prog.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )

        self.enable_new_ir(True)
        engine_mp_ir = self.get_engine("mp")
        out_mp_ir = engine_mp_ir.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )

        self.check_results(
            out_mp_prog.history["loss"][0], out_mp_ir.history["loss"][0]
        )

    def test_pp(self):
        # navie pipeline parallel without schedule
        self.enable_new_ir(False)
        engine_pp_prog = self.get_engine("pp")
        out_pp_prog = engine_pp_prog.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )

        self.enable_new_ir(True)
        # send_v2/recv_v2 dynamic_shape is True
        engine_pp_ir = self.get_engine("pp")
        out_pp_ir = engine_pp_ir.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )

        if paddle.distributed.get_rank() == 1:
            self.check_results(
                out_pp_prog.history["loss"][0], out_pp_ir.history["loss"][0]
            )

        # send_v2/recv_v2 dynamic_shape is False
        engine_pp_prog1 = self.get_engine("pp")
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


if __name__ == "__main__":
    unittest.main()
