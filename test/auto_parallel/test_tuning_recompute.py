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

import sys
import unittest

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
    modeling._global_parallel_strategy = "serial"

    gpt = GPTModel(
        vocab_size=50304,
        hidden_size=1024,
        num_hidden_layers=14,
        num_attention_heads=16,
        intermediate_size=1024 * 4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1024,
        type_vocab_size=1,
        initializer_range=0.02,
        pad_token_id=0,
        eos_token_id=7,
        bos_token_id=0,
        eol_token_id=3,
        use_new_recompute=True,
        recompute_granularity="full",
    )
    model = GPTForPretraining(
        gpt, vocab_size=50304, hidden_size=1024, initializer_range=0.02
    )
    criterion = GPTPretrainingCriterion()
    return model, criterion


def apply_pass():
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"

    recompute = strategy.recompute
    recompute.enable = True
    recompute.enable_tuning = True

    tuning = strategy.tuning
    tuning.enable = True
    tuning.profile_start_step = 1
    tuning.profile_end_step = 2
    tuning.run_after_tuning = True
    tuning.verbose = True
    return strategy


class TestRecomputePassTuning(unittest.TestCase):
    def setUp(self):

        self.batch_size = 8
        self.batch_num = 200
        self.dataset = FakeDataset(
            self.batch_size * self.batch_num,
            vocab_size=50304,
            sequence_len=1024,
        )

    def test_recompute_pass(self):

        strategy = apply_pass()
        clip = paddle.nn.ClipGradByGlobalNorm(0.2)
        opt = paddle.optimizer.AdamW(learning_rate=0.00001, grad_clip=clip)
        model, loss = generate_model()

        engine = auto.Engine(model, loss, opt, strategy=strategy)
        engine._tune(self.dataset, 3, batch_size=self.batch_size)

        assert (
            len(
                engine._dist_contexts[
                    'train'
                ].strategy.recompute.no_recompute_segments
            )
            > 0
        )


if __name__ == "__main__":
    unittest.main()
