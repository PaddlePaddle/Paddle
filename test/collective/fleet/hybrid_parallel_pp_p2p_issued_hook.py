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

"""Cover the ``P2P_ISSUED`` micro-step hook of ``VPPFhenBInBalancedMemory``.

The hook is raised after the forward P2P ops of a micro-step have been issued
but before their wait handles are consumed, so that a user hook can enqueue
kernels that overlap the NCCL send/recv. This test checks that

* the location fires exactly once per micro-step of the schedule,
* the ``output_tensor`` handed to the hook is the tensor being sent,
* enqueuing real compute in the hook does not corrupt the pipeline, i.e. the
  training loss still matches the ``forward_only`` schedule's loss,

for both ``use_batch_p2p_comm=False`` (the overlapping path) and
``use_batch_p2p_comm=True`` (no overlap possible, hook still raised).
"""

import os
import random
import unittest

import numpy as np

import paddle
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer
from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
    PipelineParallelMicroStepLocations,
    pipeline_parallel_callbacks_,
    register_global_pipeline_parallel_hook,
)
from paddle.nn import Layer

batch_size = 8
micro_batch_size = 2
accumulate_steps = batch_size // micro_batch_size
length = 8
# `VPPFhenBInBalancedMemory` is picked when the model is interleaved and
# `pp_degree <= accumulate_steps < 2 * pp_degree`, and its `_check_sanity`
# additionally requires `pp_degree > 2`.
pipeline_parallel_size = 4
num_virtual_pipeline_stages = 2
transformer_layer_num = pipeline_parallel_size * num_virtual_pipeline_stages
vocab_size = 128
hidden_size = 16


def set_random_seed(seed, dp_id):
    random.seed(seed)
    np.random.seed(seed + dp_id)
    paddle.seed(seed + dp_id)


class EmbeddingPipe(Layer):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)

    def forward(self, x):
        return self.word_embeddings(x)


class MlpPipe(Layer):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.linear(x)


class CriterionPipe(Layer):
    def forward(self, out, label):
        return out.mean()


class ModelPipe(PipelineLayer):
    def __init__(self, topology):
        descs = [LayerDesc(EmbeddingPipe)]
        descs += [LayerDesc(MlpPipe) for _ in range(transformer_layer_num)]
        super().__init__(
            layers=descs,
            loss_fn=CriterionPipe(),
            topology=topology,
            num_virtual_pipeline_stages=num_virtual_pipeline_stages,
            seg_method="layer:MlpPipe",
        )


class P2pIssuedHookRecorder:
    """Counts hook firings and enqueues real GPU work in the overlap window."""

    def __init__(self):
        self.calls = []
        self.probe = paddle.ones([8, 8], dtype="float32")

    def __call__(self, output_tensor=None, step_id=None):
        # Real kernels, enqueued on the calculation stream after the p2p ops
        # have been issued -- exactly what the location exists for.
        self.probe = paddle.matmul(self.probe, self.probe) / 8.0
        self.calls.append((step_id, output_tensor))

    def reset(self):
        self.calls = []


class TestP2pIssuedHook(unittest.TestCase):
    def setUp(self):
        self.use_batch_p2p_comm = (
            os.getenv("USE_BATCH_P2P_COMM", "False").lower() == "true"
        )
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": pipeline_parallel_size,
            "pp_configs": {
                "best_unbalanced_scheduler": True,
                "use_batch_p2p_comm": self.use_batch_p2p_comm,
            },
        }
        strategy.pipeline_configs = {
            "accumulate_steps": accumulate_steps,
            "micro_batch_size": micro_batch_size,
        }
        fleet.init(is_collective=True, strategy=strategy)

    def test_p2p_issued_hook(self):
        hcg = fleet.get_hybrid_communicate_group()
        set_random_seed(1024, hcg.get_data_parallel_rank())

        model = ModelPipe(hcg.topology())
        optimizer = paddle.optimizer.SGD(
            learning_rate=0.001, parameters=model.parameters()
        )
        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)

        self.assertEqual(
            model._get_scheduler_name(), "VPPFhenBInBalancedMemory"
        )
        self.assertEqual(model._use_batch_p2p_comm, self.use_batch_p2p_comm)

        location = PipelineParallelMicroStepLocations.P2P_ISSUED
        recorder = P2pIssuedHookRecorder()
        registered = len(pipeline_parallel_callbacks_.hooks[location])
        register_global_pipeline_parallel_hook(location, recorder)
        try:
            self.assertEqual(
                len(pipeline_parallel_callbacks_.hooks[location]),
                registered + 1,
            )
            # every micro-step of every model chunk issues one forward send
            expected_calls = accumulate_steps * num_virtual_pipeline_stages
            for _ in range(3):
                x_data = np.random.randint(
                    0, vocab_size, size=[batch_size, length]
                )
                x = paddle.to_tensor(x_data)
                x.stop_gradient = True

                # forward_only falls back to the FthenB schedule, which does
                # not raise the location: it doubles as the reference loss.
                recorder.reset()
                e_loss = model.eval_batch([x, x], True)
                self.assertEqual(recorder.calls, [])

                loss = model.train_batch([x, x], optimizer)
                self.assertEqual(len(recorder.calls), expected_calls)
                self.assertEqual(
                    sorted(step_id for step_id, _ in recorder.calls),
                    list(range(expected_calls)),
                )
                # the hook always sees the tensor of the micro-step that was
                # just issued (an activation, or the loss on the last stage,
                # which raises the location without sending anything)
                for _, output_tensor in recorder.calls:
                    self.assertIsNotNone(output_tensor)

                # the deferred wait must not corrupt the tensors in flight
                np.testing.assert_allclose(
                    loss.numpy(), e_loss.numpy(), rtol=1e-5
                )
        finally:
            del pipeline_parallel_callbacks_.hooks[location][registered:]
        self.assertEqual(
            len(pipeline_parallel_callbacks_.hooks[location]), registered
        )


if __name__ == "__main__":
    unittest.main()
