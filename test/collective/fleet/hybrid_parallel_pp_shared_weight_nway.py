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

"""Worker for the shared-weight N-way sync test.

One shared embedding is placed on every one of 4 PP stages; each stage is given a
distinct local gradient, then the ENGINE's own ``allreduce_shared_weight_gradients``
must leave every replica with the full N-way sum.  The >=3-stage path is exactly
what the old pairwise-star engine handled incorrectly (non-pivot stages got
incomplete gradients).  Launched by test_pp_with_shared_weight_nway.py through
``get_cluster_from_args``/``start_local_trainers``; do not run directly.
"""

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    PipelineLayer,
    SharedLayerDesc,
)

PP_SIZE = 4
VOCAB_SIZE = 64
HIDDEN_SIZE = 16


class EmbeddingPipe(nn.Layer):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)

    @property
    def embedding_weight(self):
        return self.embed_tokens.weight

    def forward(self, args):
        return args


class NWayEmbedNet(PipelineLayer):
    """One shared embedding placed on every one of 4 PP stages."""

    def __init__(self, **kwargs):
        layers = [
            SharedLayerDesc(
                "embed",
                EmbeddingPipe,
                shared_weight_attr="embedding_weight",
            )
            for _ in range(PP_SIZE)
        ]
        super().__init__(layers=layers, **kwargs)


def _local_embed_weight(model):
    for sub in model.sublayers():
        if isinstance(sub, EmbeddingPipe):
            return sub.embedding_weight
    return None


def _grad_of(param):
    if hasattr(param, "main_grad") and param.main_grad is not None:
        return param.main_grad
    return param.grad


def main():
    paddle.seed(2026)
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": PP_SIZE,
    }
    fleet.init(is_collective=True, strategy=strategy)
    assert dist.get_world_size() == PP_SIZE

    hcg = fleet.get_hybrid_communicate_group()
    model = NWayEmbedNet(topology=hcg.topology())

    weight = _local_embed_weight(model)
    assert weight is not None, "each PP stage must hold an embed copy"

    pp_rank = hcg.get_stage_id()
    pipe_group = hcg.get_pipe_parallel_group()

    # Distinct local gradient per stage: 10, 20, 30, 40 -> full sum 100.
    weight.grad = paddle.full_like(weight, float((pp_rank + 1) * 10))

    model.allreduce_shared_weight_gradients()

    grad = _grad_of(weight)
    assert grad is not None
    copies = []
    dist.all_gather(copies, grad.detach(), group=pipe_group)
    assert len(copies) == PP_SIZE

    expected = float(sum((r + 1) * 10 for r in range(PP_SIZE)))  # 100
    for r, c in enumerate(copies):
        np.testing.assert_allclose(
            c.astype("float32").numpy(),
            np.full_like(c.astype("float32").numpy(), expected),
            err_msg=(
                f"stage {r} gradient != full N-way sum {expected}; "
                "pairwise star leaves non-pivot stages incomplete"
            ),
        )
    print("NWAY SHARED-WEIGHT SYNC OK", flush=True)


if __name__ == "__main__":
    main()
