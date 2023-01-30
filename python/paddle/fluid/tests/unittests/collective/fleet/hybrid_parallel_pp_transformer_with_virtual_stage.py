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

<<<<<<< HEAD
import random
import unittest

import numpy as np

import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer
from paddle.fluid.dygraph.layers import Layer
=======
from __future__ import division
from __future__ import print_function

import unittest
import paddle
import numpy as np
import random
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from paddle.fluid import layers
import paddle.nn.functional as F
from paddle.distributed.fleet.meta_parallel import PipelineLayer, LayerDesc
from paddle.fluid.dygraph.layers import Layer
import paddle.nn as nn
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def set_random_seed(seed, dp_id, rank_id):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed + dp_id)
    paddle.seed(seed + dp_id)


batch_size = 8
length = 8
micro_batch_size = 2
num_virtual_pipeline_stages = 2
vocab_size = 128
hidden_size = 16
d_model = hidden_size
dim_feedforward = 4 * d_model


class EmbeddingNet(Layer):
<<<<<<< HEAD
    def __init__(self):
        super().__init__()
=======

    def __init__(self):
        super(EmbeddingNet, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(vocab_size, hidden_size)

    def forward(self, x):
<<<<<<< HEAD
        attention_mask = paddle.tensor.triu(
            (paddle.ones((length, length), dtype="float32") * -1e9), 1
        )
=======
        attention_mask = paddle.tensor.triu((paddle.ones(
            (length, length), dtype="float32") * -1e9), 1)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        no_used = paddle.ones((3, 3), dtype="int32")

        w_emb = self.word_embeddings(x)
        p_emb = self.position_embeddings(x)
        w_emb = w_emb + p_emb

        attention_mask.stop_gradient = True
        no_used.stop_gradient = True
        # need to fix bug of backward()
        return w_emb, attention_mask, no_used, p_emb


class TransformerNet(Layer):
<<<<<<< HEAD
    def __init__(self):
        super().__init__()
=======

    def __init__(self):
        super(TransformerNet, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
        product = paddle.matmul(x=q, y=k, transpose_y=True)
        product = paddle.scale(product, scale=d_model**-0.5)

        weights = F.softmax(product + mask)
        tgt = paddle.matmul(weights, v)
=======
        product = layers.matmul(x=q, y=k, transpose_y=True, alpha=d_model**-0.5)

        weights = F.softmax(product + mask)
        tgt = layers.matmul(weights, v)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        residual = tgt
        tgt = self.norm1(tgt)
        tgt = residual + tgt

        out = self.linear2(F.gelu(self.linear1(tgt), approximate=True))
        return out


class EmbeddingPipe(EmbeddingNet):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def forward(self, x):
        return super().forward(x)


class TransformerNetPipe(TransformerNet):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def forward(self, args):
        x, mask, no_used, p_emb = args[0], args[1], args[2], args[3]

        output = super().forward(x, mask)
        output = output + p_emb
        mask.stop_gradient = True
        return output, mask, no_used, p_emb


class CriterionPipe(Layer):
<<<<<<< HEAD
    def __init__(self):
        super().__init__()
=======

    def __init__(self):
        super(CriterionPipe, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def forward(self, out, label):
        loss = out.mean()
        return loss


class ModelPipe(PipelineLayer):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self, topology):
        self.descs = []
        self.descs.append(LayerDesc(EmbeddingPipe))

        for x in range(8):
            self.descs.append(LayerDesc(TransformerNetPipe))

        self.descs.append(lambda x: x[0])

        super().__init__(
            layers=self.descs,
            loss_fn=CriterionPipe(),
            topology=topology,
            num_virtual_pipeline_stages=num_virtual_pipeline_stages,
<<<<<<< HEAD
            seg_method="layer:TransformerNetPipe",
        )


class TestDistPPTraning(unittest.TestCase):
=======
            seg_method="layer:TransformerNetPipe")


class TestDistPPTraning(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
            "micro_batch_size": micro_batch_size,
=======
            "micro_batch_size": micro_batch_size
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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

        model = ModelPipe(topology)
<<<<<<< HEAD
        scheduler = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2], values=[0.001, 0.002], verbose=True
        )
        optimizer = paddle.optimizer.SGD(
            learning_rate=scheduler, parameters=model.parameters()
        )
=======
        scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=[2],
                                                       values=[0.001, 0.002],
                                                       verbose=True)
        optimizer = paddle.optimizer.SGD(learning_rate=scheduler,
                                         parameters=model.parameters())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)

        for step_id in range(5):
            x_data = np.random.randint(0, vocab_size, size=[batch_size, length])
            x = paddle.to_tensor(x_data)
            x.stop_gradient = True

            e_loss = model.eval_batch([x, x], True)
            loss = model.train_batch([x, x], optimizer, scheduler)

            np.testing.assert_allclose(loss.numpy(), e_loss.numpy())


if __name__ == "__main__":
    unittest.main()
