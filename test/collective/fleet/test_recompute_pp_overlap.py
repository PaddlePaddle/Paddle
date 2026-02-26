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

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    PipelineLayer,
)
from paddle.distributed.fleet.recompute import recompute
from paddle.nn import Layer


def set_random_seed(seed, dp_id, rank_id):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed + dp_id)
    paddle.seed(seed + dp_id)


batch_size = 8
length = 8
micro_batch_size = 2
vocab_size = 128
hidden_size = 16
d_model = hidden_size
dim_feedforward = 4 * d_model


class EmbeddingNet(Layer):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(vocab_size, hidden_size)

    def forward(self, x):
        attention_mask = paddle.tensor.triu(
            (paddle.ones((length, length), dtype="float32") * -1e9), 1
        )

        no_used = paddle.ones((3, 3), dtype="int32")

        w_emb = self.word_embeddings(x)
        p_emb = self.position_embeddings(x)
        w_emb = w_emb + p_emb

        attention_mask.stop_gradient = True
        no_used.stop_gradient = True
        # need to fix bug of backward()
        return w_emb, attention_mask, no_used, p_emb


class TransformerNet(Layer):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model, epsilon=1e-5)

    def forward(self, x, mask):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        product = paddle.matmul(x=q, y=k, transpose_y=True)
        product = paddle.scale(product, scale=d_model**-0.5)

        weights = F.softmax(product + mask)
        # TODO(shenliang03) For save/load in PipeLineParallel, can’t support dropout temporarily.
        # weights = F.dropout(weights, 0.2)
        tgt = paddle.matmul(weights, v)
        residual = tgt
        tgt = self.norm1(tgt)
        tgt = residual + tgt

        out = self.linear2(F.gelu(self.linear1(tgt), approximate=True))
        return out


class EmbeddingPipe(EmbeddingNet):
    def forward(self, x):
        return super().forward(x)


class TransformerNetPipe(TransformerNet):
    def __init__(self, is_recompute):
        super().__init__()
        self._is_recompute = is_recompute

    def forward(self, args):
        x, mask, no_used, p_emb = args[0], args[1], args[2], args[3]
        print(
            f"lijinjin, before TransformerNetPipe forward, {self._is_recompute=}"
        )
        if self._is_recompute:
            output = recompute(
                super().forward,
                x.clone(),
                mask.clone(),
                recompute_overlap=self._is_recompute,
            )
        else:
            output = super().forward(x.clone(), mask.clone())
        output = output + p_emb
        mask.stop_gradient = True
        return output, mask, no_used, p_emb


class CriterionPipe(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, out, label):
        loss = out.mean()
        return loss


class ModelPipe(PipelineLayer):
    def __init__(
        self, topology, transformer_layer_num: int = 6, is_recompute=False
    ):
        self.descs = []
        self.descs.append(LayerDesc(EmbeddingPipe))

        for x in range(transformer_layer_num):
            self.descs.append(LayerDesc(TransformerNetPipe, is_recompute))

        self.descs.append(lambda x: x[0])

        super().__init__(
            layers=self.descs,
            loss_fn=CriterionPipe(),
            topology=topology,
            seg_method="layer:TransformerNetPipe",
        )


class TestDistPPTraining(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 1
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": self.pipeline_parallel_size,
        }
        strategy.pipeline_configs = {
            "accumulate_steps": batch_size // micro_batch_size,
            "micro_batch_size": micro_batch_size,
        }
        strategy.hybrid_configs['pp_configs'].recompute_overlap = True
        fleet.init(is_collective=True, strategy=strategy)

    def test_pp_model(self):
        hcg = fleet.get_hybrid_communicate_group()
        dp_id = hcg.get_data_parallel_rank()
        pp_id = hcg.get_stage_id()
        rank_id = dist.get_rank()
        topology = hcg.topology()
        set_random_seed(1024, dp_id, rank_id)

        # construct model_a
        model_a = ModelPipe(topology)
        scheduler_a = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2], values=[0.001, 0.002], verbose=True
        )
        optimizer_a = paddle.optimizer.SGD(
            learning_rate=scheduler_a, parameters=model_a.parameters()
        )
        model_a = fleet.distributed_model(model_a)
        optimizer_a = fleet.distributed_optimizer(optimizer_a)
        model_a._recompute_overlap = False

        # construct model_b
        model_b = ModelPipe(topology, is_recompute=True)
        scheduler_b = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2], values=[0.001, 0.002], verbose=True
        )
        optimizer_b = paddle.optimizer.SGD(
            learning_rate=scheduler_b, parameters=model_b.parameters()
        )
        model_b = fleet.distributed_model(model_b)
        optimizer_b = fleet.distributed_optimizer(optimizer_b)

        model_state = model_a.state_dict()
        model_b.set_state_dict(model_state)
        for _ in range(5):
            x_data = np.random.randint(0, vocab_size, size=[batch_size, length])
            x = paddle.to_tensor(x_data)
            x.stop_gradient = True

            loss_a = model_a.train_batch([x, x], optimizer_a, scheduler_a)
            print("lijinjin, begin model_b train_batch")
            loss_b = model_b.train_batch([x, x], optimizer_b, scheduler_b)

            if pp_id != 0:
                np.testing.assert_equal(loss_a.numpy(), loss_b.numpy())


if __name__ == "__main__":
    unittest.main()
