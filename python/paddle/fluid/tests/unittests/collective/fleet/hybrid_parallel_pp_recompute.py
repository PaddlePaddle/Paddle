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
import paddle.distributed.fleet as fleet
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import framework
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer
from paddle.fluid.dygraph.layers import Layer


def set_random_seed(seed, dp_id, rank_id):
    """Set random seed for reproducability."""
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
        w_emb = self.word_embeddings(x)
        p_emb = self.position_embeddings(x)
        w_emb = w_emb + p_emb
        return w_emb


class TransformerNet(Layer):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model, epsilon=1e-5)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        product = paddle.matmul(x=q, y=k, transpose_y=True)
        product = paddle.scale(product, scale=d_model**-0.5)
        weights = F.softmax(product)

        weights = F.dropout(weights, 0.2)
        tgt = paddle.matmul(weights, v)
        residual = tgt
        tgt = self.norm1(tgt)
        tgt = residual + tgt

        out = self.linear2(F.gelu(self.linear1(tgt), approximate=True))
        return out


class EmbeddingPipe(EmbeddingNet):
    def forward(self, tensors):
        if framework.in_dygraph_mode():
            stable, x = tensors
            return stable, super().forward(x)
        else:
            return super().forward(tensors)


class TransformerNetPipe(TransformerNet):
    def forward(self, tensors):
        if framework.in_dygraph_mode():
            stable, x = tensors
            output = super().forward(x)
            return stable, output
        else:
            return super().forward(tensors)


class CriterionPipe(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, out, label):
        if framework.in_dygraph_mode():
            out = out[-1]
        loss = out.mean()
        return loss


class ModelPipe(PipelineLayer):
    def __init__(self, hcg):
        self.descs = []
        self.descs.append(LayerDesc(EmbeddingPipe))
        self.hcg = hcg

        for x in range(2):
            self.descs.append(LayerDesc(TransformerNetPipe))

        super().__init__(
            layers=self.descs,
            loss_fn=CriterionPipe(),
            topology=self.hcg.topology(),
            seg_method="layer:TransformerNetPipe",
            recompute_interval=1,
            recompute_ctx={
                "mp_group": self.hcg.get_model_parallel_group(),
                "offload": False,
                "partition": False,
            },
        )


class TestDistPPTraning(unittest.TestCase):
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
        fleet.init(is_collective=True, strategy=strategy)

    def test_pp_model(self):
        hcg = fleet.get_hybrid_communicate_group()
        word_size = hcg.get_model_parallel_world_size()
        dp_id = hcg.get_data_parallel_rank()
        pp_id = hcg.get_stage_id()
        rank_id = dist.get_rank()
        topology = hcg.topology()
        set_random_seed(1024, dp_id, rank_id)

        model = ModelPipe(hcg)
        scheduler = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2], values=[0.001, 0.002], verbose=True
        )
        optimizer = paddle.optimizer.SGD(
            learning_rate=scheduler, parameters=model.parameters()
        )

        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)

        for step_id in range(5):
            x_data = np.random.randint(0, vocab_size, size=[batch_size, length])
            x = paddle.to_tensor(x_data)
            x.stop_gradient = True
            input_ = (x, x) if framework.in_dygraph_mode() else x
            loss = model.train_batch([input_, x], optimizer, scheduler)
            # TODO(shenliang03) add utest for loss
            print("loss: ", loss)


if __name__ == "__main__":
    unittest.main()
