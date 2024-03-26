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
from paddle.distributed.fleet.utils import sequence_parallel_utils as spu
from paddle.distributed.sharding import group_sharded_parallel


def finanlize_fleet():
    # hack to finanlize fleet
    # if call multi-times fleet.init(), must call finanlize_fleet() before call fleet.init()
    import paddle.distributed.fleet.base.topology as tp
    from paddle.distributed import parallel_helper

    parallel_helper.__parallel_ctx__clz__ = None
    tp._HYBRID_PARALLEL_GROUP = None


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


class SimpleSPNet(paddle.nn.Layer):
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

        self.embedding = fleet.meta_parallel.VocabParallelEmbedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.nn.initializer.Constant(value=0.5),
        )

        self.linear1 = spu.ColumnSequenceParallelLinear(
            hidden_size,
            inner_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(init_fc1_data)
            ),
            gather_output=False,
            has_bias=True,
        )

        self.linear2 = spu.RowSequenceParallelLinear(
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

        self.norm = paddle.nn.LayerNorm(hidden_size, epsilon=1e-5)
        # if sequence parallel is true,
        # register hook to all_reduce gradient of weight, bias
        spu.mark_as_sequence_parallel_parameter(self.norm.weight)
        spu.mark_as_sequence_parallel_parameter(self.norm.bias)

        spu.register_sequence_parallel_allreduce_hooks(self, 1, False)

    def forward(self, x):
        x = self.embedding(x)

        x = paddle.transpose(x, perm=[1, 0, 2])
        x = spu.ScatterOp.apply(x)

        x = self.linear1(x)
        x = self.linear2(x)
        x = self.norm(x)
        x = self.linear3(x)

        x = paddle.transpose(x, perm=[1, 0, 2])

        x = parallel_matmul(x, self.embedding.weight, False)
        return x


class TestSpGroupShardedHybridTraining(unittest.TestCase):
    def build_model_optimizer(self, sharding_stage=2, test_dp=False):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 2
        self.data_parallel_size = 1
        self.sharding_parallel_size = 1

        assert sharding_stage in [2, 3]
        if sharding_stage == 2:
            level = 'os_g'
        elif sharding_stage == 3:
            level = 'p_g_os'

        if test_dp:
            self.data_parallel_size = 2
        else:
            self.sharding_parallel_size = 2

        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": 1,
            "sharding_degree": self.sharding_parallel_size,
        }
        fleet.init(is_collective=True, strategy=strategy)

        hcg = fleet.get_hybrid_communicate_group()
        mp_id = hcg.get_model_parallel_rank()
        dp_id = hcg.get_data_parallel_rank()
        rank_id = dist.get_rank()
        set_random_seed(1024, dp_id, rank_id)

        np_fc1 = np.random.random_sample((hidden_size, inner_size))
        np_fc2 = np.random.random_sample((inner_size, hidden_size))

        model = SimpleSPNet(
            vocab_size,
            hidden_size,
            inner_size,
            output_size,
            np_fc1,
            np_fc2,
            mp_id,
        )
        grad_clip = paddle.nn.ClipGradByGlobalNorm(1.0)
        scheduler = paddle.optimizer.lr.ExponentialDecay(
            learning_rate=0.001, gamma=0.999, verbose=True
        )
        optimizer = paddle.optimizer.SGD(
            scheduler, grad_clip=grad_clip, parameters=model.parameters()
        )

        model, optimizer = paddle.amp.decorate(
            models=model,
            optimizers=optimizer,
            level='O2',
            dtype='bfloat16',
            save_dtype='float32',
        )

        scaler = paddle.amp.GradScaler(init_loss_scaling=5160)
        if test_dp:
            scaler = fleet.distributed_scaler(scaler)

        if test_dp:
            model = fleet.distributed_model(model)
            optimizer = fleet.distributed_optimizer(optimizer)
        else:
            model, optimizer, scaler = group_sharded_parallel(
                model,
                optimizer,
                level,
                scaler,
                hcg.get_sharding_parallel_group(),
            )

        return model, optimizer, scaler

    def train_batches(
        self, batches, model, optimizer, scaler, accumulate_grad=False
    ):
        losses = []

        for iter in range(len(batches)):
            batch = batches[iter]
            with paddle.amp.auto_cast(
                enable=True, level="O2", dtype="bfloat16"
            ):
                output = model(batch)
                loss = output.mean()
                losses.append(loss)

                if not accumulate_grad:
                    scaled = scaler.scale(loss)
                    scaled.backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.clear_grad()

                if accumulate_grad:
                    if iter % 2 == 0:
                        scaled = scaler.scale(loss)
                        scaled.backward()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.clear_grad()

        return losses

    def build_model(self):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 2,
            "mp_degree": 2,
            "pp_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)

        hcg = fleet.get_hybrid_communicate_group()
        mp_id = hcg.get_model_parallel_rank()
        dp_id = hcg.get_data_parallel_rank()
        rank_id = dist.get_rank()
        set_random_seed(1024, dp_id, rank_id)

        np_fc1 = np.random.random_sample((hidden_size, inner_size))
        np_fc2 = np.random.random_sample((inner_size, hidden_size))

        model = SimpleSPNet(
            vocab_size,
            hidden_size,
            inner_size,
            output_size,
            np_fc1,
            np_fc2,
            mp_id,
        )
        finanlize_fleet()

        return model

    def test_sp_sharding_stage2_hybrid(self):
        state_dict = self.build_model().state_dict()

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

        # test mp + sharding vs mp + dp
        model_1, optimizer_1, scaler_1 = self.build_model_optimizer(
            sharding_stage=2
        )
        model_1.set_dict(state_dict)
        losses_1 = self.train_batches(batchs, model_1, optimizer_1, scaler_1)
        finanlize_fleet()

        model_2, optimizer_2, scaler_2 = self.build_model_optimizer(
            sharding_stage=2, test_dp=True
        )
        model_2.set_dict(state_dict)
        losses_2 = self.train_batches(batchs, model_2, optimizer_2, scaler_2)
        finanlize_fleet()

        for i in range(len(losses_1)):
            loss_1_fp32 = losses_1[i].cast("float32")
            loss_2_fp32 = losses_2[i].cast("float32")
            np.testing.assert_allclose(
                loss_1_fp32.numpy(), loss_2_fp32.numpy(), atol=1e-5, rtol=1e-5
            )

        # test acc
        model_3, optimizer_3, scaler_3 = self.build_model_optimizer(
            sharding_stage=2
        )
        model_3.set_dict(state_dict)
        losses_3 = self.train_batches(
            batchs, model_3, optimizer_3, scaler_3, accumulate_grad=True
        )
        finanlize_fleet()

        model_4, optimizer_4, scaler_4 = self.build_model_optimizer(
            sharding_stage=2, test_dp=True
        )
        model_4.set_dict(state_dict)
        losses_4 = self.train_batches(
            batchs, model_4, optimizer_4, scaler_4, accumulate_grad=True
        )
        finanlize_fleet()

        for i in range(len(losses_3)):
            loss_3_fp32 = losses_3[i].cast("float32")
            loss_4_fp32 = losses_4[i].cast("float32")
            np.testing.assert_allclose(
                loss_3_fp32.numpy(), loss_4_fp32.numpy(), atol=1e-5
            )

    def test_sp_sharding_stage3_hybrid(self):
        state_dict = self.build_model().state_dict()

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

        # test mp + sharding vs mp + dp
        model_1, optimizer_1, scaler_1 = self.build_model_optimizer(
            sharding_stage=3
        )
        model_1.set_dict(state_dict)
        losses_1 = self.train_batches(batchs, model_1, optimizer_1, scaler_1)
        finanlize_fleet()

        model_2, optimizer_2, scaler_2 = self.build_model_optimizer(
            sharding_stage=3, test_dp=True
        )
        model_2.set_dict(state_dict)
        losses_2 = self.train_batches(batchs, model_2, optimizer_2, scaler_2)
        finanlize_fleet()

        for i in range(len(losses_1)):
            loss_1_fp32 = losses_1[i].cast("float32")
            loss_2_fp32 = losses_2[i].cast("float32")
            np.testing.assert_allclose(
                loss_1_fp32.numpy(), loss_2_fp32.numpy(), atol=1e-5, rtol=1e-5
            )

        # test acc
        model_3, optimizer_3, scaler_3 = self.build_model_optimizer(
            sharding_stage=3
        )
        model_3.set_dict(state_dict)
        losses_3 = self.train_batches(
            batchs, model_3, optimizer_3, scaler_3, accumulate_grad=True
        )
        finanlize_fleet()

        model_4, optimizer_4, scaler_4 = self.build_model_optimizer(
            sharding_stage=3, test_dp=True
        )
        model_4.set_dict(state_dict)
        losses_4 = self.train_batches(
            batchs, model_4, optimizer_4, scaler_4, accumulate_grad=True
        )
        finanlize_fleet()

        for i in range(len(losses_3)):
            loss_3_fp32 = losses_3[i].cast("float32")
            loss_4_fp32 = losses_4[i].cast("float32")
            np.testing.assert_allclose(
                loss_3_fp32.numpy(), loss_4_fp32.numpy(), atol=1e-5, rtol=1e-5
            )


if __name__ == "__main__":
    unittest.main()
