# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet


def set_random_seed(seed, dp_id, rank_id):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed + dp_id)
    paddle.seed(seed + rank_id)


vocab_size = 20
hidden_size = 10
inner_size = 8
output_size = 10
seq_length = 2
batch_size = 4


def parallel_matmul(lm_output, logit_weights, parallel_output):
    hcg = fleet.get_hybrid_communicate_group()
    model_parallel_group = hcg.get_model_parallel_group()
    world_size = hcg.get_model_parallel_world_size()
    rank = hcg.get_model_parallel_rank()

    if world_size > 1:
        input_parallel = paddle.distributed.collective._c_identity(
            lm_output, group=model_parallel_group
        )

        logits = paddle.matmul(input_parallel, logit_weights, transpose_y=True)

        if parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(
            logits, group=model_parallel_group
        )
    else:
        logits = paddle.matmul(lm_output, logit_weights, transpose_y=True)
        return logits


class SimpleMPNet(paddle.nn.Layer):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        inner_size,
        output_size,
        np_fc1,
        np_fc2,
        mp_id,
    ):
        super().__init__()

        if mp_id == 0:
            init_fc1_data = np_fc1[:, : (inner_size // 2)]
            init_fc2_data = np_fc2[: (inner_size // 2), :]
        else:
            init_fc1_data = np_fc1[:, (inner_size // 2) :]
            init_fc2_data = np_fc2[(inner_size // 2) :, :]

        self.linear1 = fleet.meta_parallel.ColumnParallelLinear(
            hidden_size,
            inner_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(init_fc1_data)
            ),
            gather_output=False,
            has_bias=True,
        )

        self.linear2 = fleet.meta_parallel.RowParallelLinear(
            inner_size,
            hidden_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(init_fc2_data)
            ),
            input_is_parallel=True,
            has_bias=True,
        )

        self.linear3 = paddle.nn.Linear(
            hidden_size,
            output_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
        )

        self.embedding = fleet.meta_parallel.VocabParallelEmbedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.nn.initializer.Constant(value=0.5),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = parallel_matmul(x, self.embedding.weight, False)
        return x


class SimpleDPNet(paddle.nn.Layer):
    def __init__(
        self, vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
    ):
        super().__init__()
        self.linear1 = paddle.nn.Linear(
            hidden_size,
            inner_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_fc1)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
        )

        self.linear2 = paddle.nn.Linear(
            inner_size,
            hidden_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_fc2)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
        )

        self.linear3 = paddle.nn.Linear(
            hidden_size,
            output_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
            bias_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
        )

        self.embedding = paddle.nn.Embedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.nn.initializer.Constant(value=0.5),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = paddle.matmul(x, self.embedding.weight, transpose_y=True)
        return x


class TestDistMPSyncTraining(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 2
        self.data_parallel_size = 1
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": 1,
            "mp_configs": {
                "sync_param": False,
                "sync_grad": False,
                "sync_moment": False,
            },
        }
        fleet.init(is_collective=True, strategy=strategy)

    def build_model_optimizer_train(
        self,
        batchs,
        fp16=False,
        amp_level="O1",
        mp_sync_param=False,
        mp_sync_grad=False,
        mp_sync_moment=False,
    ):
        hcg = fleet.get_hybrid_communicate_group()
        word_size = hcg.get_model_parallel_world_size()
        mp_id = hcg.get_model_parallel_rank()
        dp_id = hcg.get_data_parallel_rank()
        rank_id = dist.get_rank()
        paddle.seed(2023)
        np.random.seed(2023)
        random.seed(2023)
        set_random_seed(1024, dp_id, rank_id)

        np_fc1 = np.random.random_sample((hidden_size, inner_size))
        np_fc2 = np.random.random_sample((inner_size, hidden_size))

        model = SimpleMPNet(
            vocab_size,
            hidden_size,
            inner_size,
            output_size,
            np_fc1,
            np_fc2,
            mp_id,
        )
        optimizer = paddle.optimizer.AdamW(
            learning_rate=0.1, parameters=model.parameters()
        )

        if fp16 and amp_level == "O2":
            model, optimizer = paddle.amp.decorate(
                models=model, optimizers=optimizer, level='O2'
            )

        strategy = fleet.fleet._user_defined_strategy
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": 1,
            "mp_configs": {
                "sync_param": mp_sync_param,
                "sync_grad": mp_sync_grad,
                "sync_moment": mp_sync_moment,
            },
        }

        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)
        return self.train_batch(batchs, model, optimizer, fp16, amp_level)

    def train_batch(self, batchs, model, optimizer, fp16=False, amp_level="O1"):
        losses = []
        if fp16:
            scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
            scaler = fleet.distributed_scaler(scaler)
        for batch in batchs:
            with paddle.amp.auto_cast(enable=fp16, level=amp_level):
                output = model(batch)
                loss = output.mean()
                losses.append(loss.numpy())
            if fp16:
                scaled = scaler.scale(loss)
                scaled.backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
        return losses

    def mp_sync_base(
        self, mp_sync_param=False, mp_sync_grad=False, mp_sync_moment=False
    ):
        batchs = []
        for _ in range(5):
            np_data = np.random.randint(
                0,
                vocab_size,
                (
                    batch_size,
                    seq_length,
                ),
            )
            batchs.append(paddle.to_tensor(np_data))

        losses = self.build_model_optimizer_train(batchs)
        losses_sync = self.build_model_optimizer_train(
            batchs,
            mp_sync_param=mp_sync_param,
            mp_sync_grad=mp_sync_grad,
            mp_sync_moment=mp_sync_moment,
        )

        for i in range(len(losses)):
            np.testing.assert_allclose(losses[i], losses_sync[i], rtol=1e-6)

        # test fp16 O1
        losses_fp16 = self.build_model_optimizer_train(batchs, fp16=True)
        losses_sync_fp16 = self.build_model_optimizer_train(
            batchs,
            fp16=True,
            mp_sync_param=mp_sync_param,
            mp_sync_grad=mp_sync_grad,
            mp_sync_moment=mp_sync_moment,
        )

        for i in range(len(losses_fp16)):
            np.testing.assert_allclose(
                losses_fp16[i], losses_sync_fp16[i], rtol=1e-6
            )

        # test fp16 O2
        losses_fp16_O2 = self.build_model_optimizer_train(
            batchs, fp16=True, amp_level="O2"
        )
        losses_sync_fp16_O2 = self.build_model_optimizer_train(
            batchs,
            fp16=True,
            amp_level="O2",
            mp_sync_param=mp_sync_param,
            mp_sync_grad=mp_sync_grad,
            mp_sync_moment=mp_sync_moment,
        )

        for i in range(len(losses_fp16_O2)):
            np.testing.assert_allclose(
                losses_fp16_O2[i], losses_sync_fp16_O2[i], rtol=1e-6
            )

    def test_mp_sync_param(self):
        self.mp_sync_base(mp_sync_param=True)

    def test_mp_sync_grad(self):
        self.mp_sync_base(mp_sync_grad=True)

    def test_mp_sync_moment(self):
        self.mp_sync_base(mp_sync_moment=True)

    def test_mp_sync_all(self):
        self.mp_sync_base(
            mp_sync_param=True, mp_sync_grad=True, mp_sync_moment=True
        )


class TestDistMPSyncModelTraining(TestDistMPSyncTraining):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 2
        self.data_parallel_size = 1
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": 1,
            "mp_configs": {
                "sync_param": False,
                "sync_grad": False,
                "sync_moment": False,
                "sync_mode": "average",
                "sync_param_name": ["embedding", "layer_norm", ".b_"],
            },
        }
        fleet.init(is_collective=True, strategy=strategy)


class TestDistMPTraining(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 2
        self.data_parallel_size = 1
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)

    def train_batch(self, batch, model, optimizer, is_mp):
        output = model(batch)
        loss = output.mean()
        loss.backward()  # do backward
        optimizer.step()  # update parameters
        optimizer.clear_grad()
        return loss

    def build_optimizer(self, model):
        optimizer = paddle.optimizer.SGD(
            learning_rate=0.001, parameters=model.parameters()
        )
        return optimizer

    def build_model_optimizer(self):
        hcg = fleet.get_hybrid_communicate_group()
        word_size = hcg.get_model_parallel_world_size()
        mp_id = hcg.get_model_parallel_rank()
        dp_id = hcg.get_data_parallel_rank()
        rank_id = dist.get_rank()
        set_random_seed(1024, dp_id, rank_id)

        np_fc1 = np.random.random_sample((hidden_size, inner_size))
        np_fc2 = np.random.random_sample((inner_size, hidden_size))

        model_a = SimpleMPNet(
            vocab_size,
            hidden_size,
            inner_size,
            output_size,
            np_fc1,
            np_fc2,
            mp_id,
        )
        optimizer_a = self.build_optimizer(model_a)
        model_a = fleet.distributed_model(model_a)
        optimizer_a = fleet.distributed_optimizer(optimizer_a)

        model_b = SimpleDPNet(
            vocab_size, hidden_size, inner_size, output_size, np_fc1, np_fc2
        )
        optimizer_b = self.build_optimizer(model_b)

        return model_a, optimizer_a, model_b, optimizer_b

    def test_mp_model(self):
        (
            model_a,
            optimizer_a,
            model_b,
            optimizer_b,
        ) = self.build_model_optimizer()

        for _ in range(5):
            np_data = np.random.randint(
                0,
                vocab_size,
                (
                    batch_size,
                    seq_length,
                ),
            )
            batch = paddle.to_tensor(np_data)
            loss_a = self.train_batch(batch, model_a, optimizer_a, True)
            loss_b = self.train_batch(batch, model_b, optimizer_b, False)

            np.testing.assert_allclose(
                loss_a.numpy(), loss_b.numpy(), rtol=1e-6
            )


if __name__ == "__main__":
    unittest.main()
