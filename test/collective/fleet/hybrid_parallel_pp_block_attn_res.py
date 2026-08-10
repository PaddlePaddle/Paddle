# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

# NOTE: enable the BlockAttnRes communication optimization BEFORE any pipeline
# model is constructed. PipelineParallel reads this env var once in __init__
# (pipeline_parallel.py: `self._block_atten_res_opt = ...`), so it must be set
# before fleet.distributed_model() is called.
import os

os.environ["BLOCK_ATTEN_RES_COMM_OPT"] = "1"

import random
import unittest

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer
from paddle.nn import Layer


def set_random_seed(seed, dp_id, rank_id):
    random.seed(seed)
    np.random.seed(seed + dp_id)
    paddle.seed(seed + dp_id)


batch_size = 8
length = 8
micro_batch_size = 2
num_virtual_pipeline_stages = 2
vocab_size = 128
hidden_size = 16


# ---------------------------------------------------------------------------
# Minimal model reproducing the BlockAttnRes "blocks" dict contract used by
# pipeline_parallel.py. Each stage/chunk passes a dict:
#     {"hidden": Tensor[B, S, H], "blocks": [Tensor[B, S, H], ...]}
# appending one new block per BlockLayerPipe. The pipeline engine's
# _merge_block_cache / _update_block_cache handle caching, meta and trimming;
# the model just keeps the "blocks" list flowing.
# ---------------------------------------------------------------------------
class EmbeddingPipe(Layer):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)

    def forward(self, x):
        h = self.word_embeddings(x)
        return {"hidden": h, "blocks": []}


class BlockLayerPipe(Layer):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size, epsilon=1e-5)

    def forward(self, state):
        if isinstance(state, dict):
            hidden = state["hidden"]
            blocks = list(state["blocks"])
        else:  # only reachable if fed a raw tensor (not expected here)
            hidden = state
            blocks = []
        partial = self.norm(self.linear(hidden)) + hidden
        blocks = blocks + [partial]
        return {"hidden": partial, "blocks": blocks}


class FinalPipe(Layer):
    def forward(self, state):
        if isinstance(state, dict):
            return state["hidden"]
        return state


class CriterionPipe(Layer):
    def forward(self, out, label):
        return out.mean()


class ModelPipe(PipelineLayer):
    def __init__(self, topology, block_layer_num: int = 4):
        self.descs = []
        self.descs.append(LayerDesc(EmbeddingPipe))
        for _ in range(block_layer_num):
            self.descs.append(LayerDesc(BlockLayerPipe))
        self.descs.append(LayerDesc(FinalPipe))

        super().__init__(
            layers=self.descs,
            loss_fn=CriterionPipe(),
            topology=topology,
            num_virtual_pipeline_stages=num_virtual_pipeline_stages,
            seg_method="layer:BlockLayerPipe",
        )


class TestDistPPBlockAttnRes(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.pipeline_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": self.pipeline_parallel_size,
            "pp_configs": {
                "enable_timer": True,
            },
        }
        strategy.pipeline_configs = {
            "accumulate_steps": batch_size // micro_batch_size,
            "micro_batch_size": micro_batch_size,
        }
        fleet.init(is_collective=True, strategy=strategy)

    def test_pp_block_attn_res(self):
        hcg = fleet.get_hybrid_communicate_group()
        dp_id = hcg.get_data_parallel_rank()
        rank_id = dist.get_rank()
        topology = hcg.topology()
        set_random_seed(1024, dp_id, rank_id)

        model = ModelPipe(topology)
        scheduler = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2], values=[0.001, 0.002], verbose=True
        )
        optimizer = paddle.optimizer.SGD(
            learning_rate=scheduler, parameters=model.parameters()
        )

        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)

        # Guarantee the optimization path is active (interleave 1F1B).
        assert model._block_atten_res_opt is True, (
            "BLOCK_ATTEN_RES_COMM_OPT must be enabled for this test"
        )

        for _ in range(3):
            x_data = np.random.randint(0, vocab_size, size=[batch_size, length])
            x = paddle.to_tensor(x_data)
            x.stop_gradient = True

            loss = model.train_batch([x, x], optimizer, scheduler)
            # loss is only meaningful on the last pipeline stage; on other
            # stages train_batch may return None.
            if loss is not None:
                assert paddle.isfinite(loss).item(), (
                    f"loss is not finite: {loss.item()}"
                )


if __name__ == "__main__":
    unittest.main()
