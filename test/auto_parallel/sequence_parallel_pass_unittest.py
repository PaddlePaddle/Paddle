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

import random
import sys
import unittest

import numpy as np
from get_gpt_model import FakeDataset

import paddle
from paddle.distributed.fleet import auto

sys.path.append("..")
import auto_parallel_gpt_model as modeling
from auto_parallel_gpt_model import (
    GPTForPretraining,
    GPTModel,
    GPTPretrainingCriterion,
)


def generate_model():
    modeling.init_global()
    modeling._global_parallel_strategy = "mp"
    ranks = list(range(paddle.distributed.get_world_size()))
    modeling._global_process_mesh = auto.ProcessMesh(
        mesh=ranks, dim_names=["x"]
    )

    gpt = GPTModel(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=8,
        intermediate_size=256,
        hidden_act="gelu",
        hidden_dropout_prob=1e-3,
        attention_probs_dropout_prob=1e-3,
        max_position_embeddings=1024,
        type_vocab_size=1,
        initializer_range=0.02,
        pad_token_id=0,
        eos_token_id=7,
        bos_token_id=0,
        eol_token_id=3,
        use_new_recompute=True,
    )
    model = GPTForPretraining(
        gpt, vocab_size=1000, hidden_size=64, initializer_range=0.02
    )
    criterion = GPTPretrainingCriterion()
    return model, criterion


def apply_pass(use_sp_pass=False):
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True

    # amp
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
    amp.init_loss_scaling = 32768
    amp.use_fp16_guard = False

    # grad merge
    gradient_merge = strategy.gradient_merge
    gradient_merge.enable = True
    gradient_merge.k_steps = 4
    gradient_merge.avg = True

    # recompute
    recompute = strategy.recompute
    recompute.enable = True

    if use_sp_pass:
        sp = strategy.sequence_parallel
        sp.enable = True

    return strategy


def reset_prog():
    paddle.fluid.framework.switch_main_program(paddle.static.Program())
    paddle.fluid.framework.switch_startup_program(paddle.static.Program())


class TestSequenceParallelPass(unittest.TestCase):
    def setUp(self):
        self.rtol = 1e-3
        self.atol = 1e-3
        self.batch_size = 8
        self.batch_num = 10
        self.clip_norm = 0.2
        self.dataset = FakeDataset(self.batch_size * self.batch_num)

    def init(self, engine):
        paddle.seed(2023)
        np.random.seed(2023)
        random.seed(2023)
        place = paddle.fluid.CUDAPlace(paddle.distributed.ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(self, use_sequnce_parallel=False):
        reset_prog()

        strategy = apply_pass(use_sequnce_parallel)
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=0.00001, grad_clip=clip)
        model, loss = generate_model()

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

    def check_program(self, main_prog):
        c_reduce_scatter_num = 0
        c_allgather_num = 0
        for idx, op in enumerate(list(main_prog.block(0).ops)):
            if op.type == 'c_reducescatter':
                c_reduce_scatter_num += 1
            elif op.type == 'c_allgather':
                c_allgather_num += 1
        # Each forward and recompute block has 4 allgather and 2 reducescatter.
        # Each backward block has 2 allgather and 4 reducescatter.
        # The forward final output and backward final grad has 1 allgather respectively.
        # For model with 2 layers and use recompute
        # num of allgather = (4 + 4 + 2) * 2 + 2 = 22
        # num of reducescatter = (2 + 2 + 4) * 2 = 16
        assert c_reduce_scatter_num == 16
        assert c_allgather_num == 22

    def test_sequence_parallel_pass(self):
        # mp2 training
        mp_engine = self.get_engine()
        history = mp_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        mp_losses = np.array(history.history["loss"])

        # mp2 with sequence parallel training
        mp_sp_engine = self.get_engine(True)
        history = mp_sp_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        sp_losses = np.array(history.history["loss"])
        self.check_results(mp_losses, sp_losses)
        self.check_program(mp_sp_engine.main_program)


if __name__ == "__main__":
    unittest.main()
