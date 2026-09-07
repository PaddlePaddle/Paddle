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

"""Multi-process VPP regression for the two p2p-window fillers.

What a single-process queue test cannot reach, and this one does:

1. **The sync / async p2p branch is chosen per rank, per micro step.** The
   scheduler opens an *asynchronous* window only when it has something to run
   inside it -- a flushed dW batch (``have_dw``) or a chunk with pending
   recompute (``rc_key``) -- and otherwise takes the synchronous path. That
   decision is rank-local, so getting it wrong hangs the job instead of failing
   an assert. Layer ``(vpp0, pp2)`` here is parameter free, so on that one rank
   the window has neither, while the other three have both: the asymmetric
   pending case.

2. **The received activation is appended to a per-virtual-chunk FIFO.** A wrong
   chunk key does not raise; it silently routes gradients into the wrong virtual
   chunk. Comparing every parameter after an SGD step catches exactly that
   (same starting weights and same lr, so equal parameters means equal grads).

3. **``WeightGradStore.flush()`` enqueues unconditionally**, so a blind call puts
   an *empty* batch on the queue and makes ``have_dw`` true with nothing to run.
   The parameter-free chunk is what exercises the ``if WeightGradStore.cache:``
   guard -- without it that rank takes the async branch with an empty window.

Layout: ``pp=4``, ``vpp=2`` -> 8 virtual chunks, one segmentable layer each.

    chunk (vpp, pp) | layer          | queues dW | registers recompute
    ----------------|----------------|-----------|--------------------
    (0, 0)          | LinearPipe     | yes       | no
    (0, 1)          | LinearPipe     | yes       | no
    (0, 2)          | NoParamPipe    | **no**    | **no**
    (0, 3)          | LinearPipe     | yes       | no
    (1, 0)          | RecomputePipe  | yes       | **yes**
    (1, 1)          | LinearPipe     | yes       | no
    (1, 2)          | RecomputePipe  | yes       | **yes**
    (1, 3)          | LinearPipe     | yes       | no

So ``rc_key`` is non-None on two of the four ranks only, and ``have_dw`` is false
on one -- both asymmetries hold simultaneously.

The reference is the same model with both fillers off. The fillers only move
*when* a computation runs, never what it computes, so loss and parameters must
match to the bit.

``RecomputeStore`` is a registry that Paddle only consumes; the span type that
fills it lives in the model library. ``_RecomputeSpan`` below is a minimal
stand-in with the same contract (discard in forward, ``run_recompute_now()``
rebuilds, run-on-demand if nobody ran it early).
"""

import os
import random
import unittest

import numpy as np

import paddle
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer
from paddle.distributed.fleet.meta_parallel.zero_bubble_utils import (
    RecomputeStore,
    SplitBWLinear,
    WeightGradStore,
)
from paddle.io import DataLoader, Dataset
from paddle.nn import Layer

HIDDEN = 8
MICRO_BATCH_SIZE = 2
VPP = 2
STEPS = 4

# Which scheduler to exercise; set by the launcher. See fleet/model.py:
#   acc >= 2 * pp                         -> PipelineParallelWithInterleave
#   pp <= acc < 2 * pp, best_unbalanced   -> VPPFhenBInBalancedMemory
ACC_STEPS = int(os.environ.get("PP_DW_ACC_STEPS", "8"))
BEST_UNBALANCED = os.environ.get("PP_DW_BEST_UNBALANCED", "0") == "1"
# _check_data_valid requires exactly micro_batch_size * accumulate_steps.
BATCH_SIZE = MICRO_BATCH_SIZE * ACC_STEPS


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        rng = np.random.RandomState(idx)
        image = rng.random([HIDDEN]).astype('float32')
        label = rng.randint(0, HIDDEN, (1,)).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples


class _RecomputeSpan:
    """Stand-in for the model library's selective-recompute span.

    Contract the scheduler relies on, and the only part Paddle owns:
      * registered on ``RecomputeStore`` during the forward of a known chunk;
      * ``run_recompute_now()`` rebuilds the discarded activation;
      * if nobody ran it early, the backward runs it itself and tells the store
        via ``drop()`` so ``pending()`` stays honest.
    """

    def __init__(self, x):
        self._x = x
        self.value = None

    def _rebuild(self):
        return paddle.tanh(self._x)

    def run_recompute_now(self):
        # Called by the scheduler inside a p2p window.
        if self.value is None:
            self.value = self._rebuild()

    def materialize(self):
        # Called by the backward. Either the scheduler already ran it early, or
        # it runs here; both must produce the same value.
        if self.value is None:
            self.value = self._rebuild()
            RecomputeStore.drop(self)
        return self.value

    def release(self):
        self._x = None
        self.value = None


class _DiscardTanh(PyLayer):
    """``tanh`` that keeps no output: the backward rebuilds it from the input.

    One tensor input only, so no parameter gradient crosses the PyLayer boundary
    -- the dW side is covered by ``SplitBWLinear`` instead. The backward really
    *consumes* the rebuilt value, so a span that ran early and produced something
    different would show up as a wrong gradient, not just as a missing call.
    """

    @staticmethod
    def forward(ctx, x):
        span = _RecomputeSpan(x)
        out = paddle.tanh(x)
        if RecomputeStore.enabled:
            # Discard: hand the rebuild recipe to the store so the scheduler can
            # run it early inside a p2p window.
            RecomputeStore.put(span)
        else:
            span.value = out
        ctx.save_for_backward(x)
        ctx.span = span
        return out

    @staticmethod
    def backward(ctx, out_grad):
        h = ctx.span.materialize()
        x_grad = out_grad * (1 - h * h)
        ctx.span.release()
        return x_grad


class LinearPipe(SplitBWLinear):
    """A linear whose dW goes through WeightGradStore when it is enabled.

    ``SplitBWLinear`` is Paddle's own split-backward linear: its PyLayer queues
    the weight-grad closure on ``WeightGradStore`` when ``enabled`` is set and
    computes it inline otherwise. That is what the scheduler pops inside a p2p
    window, so it is what the deferral path has to be tested against. Subclassed
    only to give ``seg_method`` a name to match on.
    """


class NoParamPipe(Layer):
    """A parameter-free chunk: queues no dW and registers no recompute.

    This is the asymmetric case. On the rank holding it, the p2p window has
    nothing to fill it with (``have_dw`` false, ``rc_key`` None) while the other
    three ranks have both, so the four ranks must not disagree about whether the
    transfer is asynchronous. It is also what exercises the
    ``if WeightGradStore.cache:`` guard -- ``flush()`` enqueues unconditionally,
    so a blind call would put an empty batch on the queue and make ``have_dw``
    true here with nothing to run.
    """

    def forward(self, input):
        return paddle.tanh(input)


class RecomputePipe(Layer):
    """A chunk that both queues dW and registers a recompute span."""

    def __init__(self, hidden):
        super().__init__()
        self.fc1 = SplitBWLinear(hidden, hidden, bias_attr=False)
        self.fc2 = SplitBWLinear(hidden, hidden, bias_attr=False)

    def forward(self, input):
        return self.fc2(_DiscardTanh.apply(self.fc1(input)))


class CriterionPipe(Layer):
    def forward(self, logits, label):
        return paddle.nn.functional.cross_entropy(
            logits, label.reshape([-1]), reduction='mean'
        )


class ModelPipe(PipelineLayer):
    """8 segmentable layers -> exactly one per (vpp, pp) chunk at pp=4, vpp=2."""

    def __init__(self, **kwargs):
        decs = [
            LayerDesc(LinearPipe, HIDDEN, HIDDEN, bias_attr=False),  # (0, 0)
            LayerDesc(LinearPipe, HIDDEN, HIDDEN, bias_attr=False),  # (0, 1)
            LayerDesc(NoParamPipe),  # (0, 2)  no dW, no recompute
            LayerDesc(LinearPipe, HIDDEN, HIDDEN, bias_attr=False),  # (0, 3)
            LayerDesc(RecomputePipe, HIDDEN),  # (1, 0)  registers a span
            LayerDesc(LinearPipe, HIDDEN, HIDDEN, bias_attr=False),  # (1, 1)
            LayerDesc(RecomputePipe, HIDDEN),  # (1, 2)  registers a span
            LayerDesc(LinearPipe, HIDDEN, HIDDEN, bias_attr=False),  # (1, 3)
        ]
        super().__init__(
            layers=decs,
            loss_fn=CriterionPipe(),
            seg_method="layer:LinearPipe|NoParamPipe|RecomputePipe",
            **kwargs,
        )


class TestDwRecomputeOverlap(unittest.TestCase):
    def setUp(self):
        self.pp_degree = 4
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": self.pp_degree,
            "pp_configs": {
                "best_unbalanced_scheduler": BEST_UNBALANCED,
            },
        }
        strategy.pipeline_configs = {
            "accumulate_steps": ACC_STEPS,
            "micro_batch_size": MICRO_BATCH_SIZE,
        }
        fleet.init(is_collective=True, strategy=strategy)

    @staticmethod
    def _reset_stores():
        WeightGradStore.enabled = False
        RecomputeStore.enabled = False
        WeightGradStore.clear()
        RecomputeStore.clear()

    def _build(self):
        set_random_seed(1024)
        model = ModelPipe(
            num_stages=self.pp_degree,
            num_virtual_pipeline_stages=VPP,
        )
        optimizer = paddle.optimizer.SGD(
            learning_rate=0.01, parameters=model.parameters()
        )
        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)
        return model, optimizer

    def _run(self, defer_dw, early_recompute):
        """Train STEPS steps; return (losses, per-step grads, final params).

        The gradients are snapshotted **after backward and before the optimizer
        step**, which is the direct check: deferring dW only changes *where* the
        weight-grad GEMM runs and running a recompute span early only changes
        *when* an activation is rebuilt, so every gradient must come out bit for
        bit identical, not merely close.

        ``train_batch`` always steps the optimizer (and ``_optimizer_step``
        rescales the grads in place by 1/accumulate_steps), so the two halves are
        called separately here to get at the grads in between. Final parameters
        are compared as well, which is what a gradient landing on the *wrong*
        parameter -- a wrong virtual-chunk key -- shows up as.
        """
        self._reset_stores()
        model, optimizer = self._build()

        dataset = RandomDataset(BATCH_SIZE * (STEPS + 1))
        reader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
        )

        WeightGradStore.enabled = defer_dw
        RecomputeStore.enabled = early_recompute
        losses = []
        grads = []
        try:
            for step, (img, label) in enumerate(reader()):
                if step >= STEPS:
                    break
                data = model._prepare_training([img, label], optimizer, None)
                loss = model.forward_backward_pipeline(data, None)
                losses.append(np.array(loss))
                grads.append(
                    [
                        (
                            name,
                            None if p.grad is None else p.grad.numpy().copy(),
                        )
                        for name, p in model.named_parameters()
                    ]
                )
                model._optimizer_step()
        finally:
            self._reset_stores()

        params = [
            (name, p.numpy().copy()) for name, p in model.named_parameters()
        ]
        return losses, grads, params

    def _assert_same(self, ref, got, label):
        ref_losses, ref_grads, ref_params = ref
        got_losses, got_grads, got_params = got

        self.assertEqual(len(ref_losses), STEPS)
        for step in range(STEPS):
            np.testing.assert_equal(
                ref_losses[step],
                got_losses[step],
                err_msg=f"{label} changed the loss at step {step}",
            )

            self.assertEqual(len(ref_grads[step]), len(got_grads[step]))
            for (name, ref_g), (got_name, got_g) in zip(
                ref_grads[step], got_grads[step]
            ):
                self.assertEqual(name, got_name)
                if ref_g is None or got_g is None:
                    self.assertIs(
                        ref_g,
                        got_g,
                        f"{label}: {name} has a gradient in one run only "
                        f"at step {step}",
                    )
                    continue
                np.testing.assert_equal(
                    ref_g,
                    got_g,
                    err_msg=(
                        f"{label} changed the gradient of {name} at step {step}"
                    ),
                )

        for (name, ref_p), (got_name, got_p) in zip(ref_params, got_params):
            self.assertEqual(name, got_name)
            np.testing.assert_equal(
                ref_p,
                got_p,
                err_msg=f"{label} moved parameter {name}",
            )

    def test_dw_and_recompute_match_baseline(self):
        baseline = self._run(defer_dw=False, early_recompute=False)
        self._assert_same(
            baseline,
            self._run(defer_dw=True, early_recompute=False),
            "dW deferral",
        )
        self._assert_same(
            baseline,
            self._run(defer_dw=True, early_recompute=True),
            "dW deferral + early recompute",
        )

    def test_stores_are_drained_every_step(self):
        """Both stores must be empty when a step ends, on every rank.

        A leftover dW batch means some window popped nothing and the grad is
        missing; a leftover recompute group means a chunk key was built for a
        chunk whose backward never came. Either one is silent, so assert it.
        """
        self._reset_stores()
        model, optimizer = self._build()
        dataset = RandomDataset(BATCH_SIZE * 3)
        reader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
        )

        WeightGradStore.enabled = True
        RecomputeStore.enabled = True
        try:
            for step, (img, label) in enumerate(reader()):
                if step >= 2:
                    break
                model.train_batch([img, label], optimizer)
                self.assertTrue(
                    WeightGradStore.funcs_queue.empty(),
                    "WeightGradStore.funcs_queue not drained at step end",
                )
                self.assertEqual(
                    WeightGradStore.cache,
                    [],
                    "WeightGradStore.cache not flushed at step end",
                )
                self.assertEqual(
                    RecomputeStore.groups,
                    {},
                    "RecomputeStore has spans left over at step end",
                )
        finally:
            self._reset_stores()


if __name__ == "__main__":
    unittest.main()
