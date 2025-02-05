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

sys.path.append("../../legacy_test")
import auto_parallel_gpt_model as modeling
from auto_parallel_gpt_model import (
    GPTForPretraining,
    GPTModel,
    GPTPretrainingCriterion,
)

import paddle
from paddle import static
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.process_mesh import ProcessMesh
from paddle.distributed.auto_parallel.static.cluster import Cluster
from paddle.distributed.auto_parallel.static.dist_context import (
    DistributedContext,
    set_default_distributed_context,
)
from paddle.distributed.auto_parallel.static.tuner.parallel_tuner import (
    ParallelTuner,
)

paddle.enable_static()

batch_size = 4
epoch_num = 10
hidden_size = 1024
sequence_len = 512
_g_process_mesh = [
    ProcessMesh([0, 1], dim_names=["x"]),
    ProcessMesh([2, 3], dim_names=["x"]),
]


def get_program_v3():
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.semi_auto = True
    # fleet.init(is_collective=True, strategy=dist_strategy)
    place = paddle.set_device("gpu")
    gpus = [0, 1]
    batch_size = 8
    sequence_len = 512
    vocab_size = 1000

    train_program = static.Program()
    start_program = static.Program()
    modeling.init_global()
    modeling._global_parallel_strategy = None
    # modeling.DPMPPP_MESH_LIST = [
    #     ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"]),
    #     ProcessMesh([[4, 5], [6, 7]], dim_names=["x", "y"])
    # ]
    with static.program_guard(train_program, start_program):
        tokens = paddle.static.data(
            name="tokens", shape=[batch_size, sequence_len], dtype='int64'
        )
        position_ids = paddle.static.data(
            name="position_ids", shape=[batch_size, sequence_len], dtype='int64'
        )
        attention_mask = paddle.static.data(
            name="attention_mask",
            shape=[batch_size, 1, sequence_len, sequence_len],
            dtype='float32',
        )
        labels = paddle.static.data(
            name="labels", shape=[batch_size, sequence_len], dtype='int64'
        )
        loss_mask = paddle.static.data(
            name="loss_mask", shape=[batch_size, sequence_len], dtype='float32'
        )
        data_holder = [tokens, position_ids, attention_mask, labels, loss_mask]

        gpt = GPTModel(
            vocab_size=1000,
            hidden_size=1024,
            num_hidden_layers=2,
            num_attention_heads=16,
            intermediate_size=4 * 1024,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            max_position_embeddings=1024,
            type_vocab_size=1,
            initializer_range=0.02,
            pad_token_id=0,
            eos_token_id=7,
            bos_token_id=0,
            eol_token_id=3,
            pp_degree=1,
        )

        model = GPTForPretraining(
            gpt, vocab_size=1000, hidden_size=64, initializer_range=0.02
        )
        preds = model(tokens, position_ids, attention_mask)
        criterion = GPTPretrainingCriterion()
        loss = criterion(preds, labels, loss_mask)

        optimizer = paddle.optimizer.Adam(
            learning_rate=0.00001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08,
            grad_clip=None,
        )

        feed_vars = {
            "inputs": [tokens, position_ids, attention_mask, loss_mask],
            "labels": [labels],
        }
        fetch_vars = {"loss": [loss]}

    return (
        train_program,
        start_program,
        None,
        loss,
        optimizer,
        feed_vars,
        fetch_vars,
    )


class TestParallelTunerTrain(unittest.TestCase):
    def test_tune_with_train(self):
        flag = False
        set_default_distributed_context(DistributedContext())
        (
            train_program,
            start_program,
            dataloader,
            loss,
            optimizer,
            feed_vars,
            fetch_vars,
        ) = get_program_v3()
        cluster = Cluster()
        cluster.gen_default_config_cluster(node_count=1, device_count=8)
        dist_context = DistributedContext(
            train_program,
            start_program,
            optimizer,
            loss,
            feed_vars,
            fetch_vars,
            cluster,
        )
        dist_context.initialize()
        parallel_tuner = ParallelTuner(dist_context, max_trials=3, mode="train")
        parallel_tuner.tune()
        parallel_tuner._store_best_parallel_strategy()
        flag = True
        self.assertTrue(flag)


if __name__ == "__main__":
    unittest.main()
