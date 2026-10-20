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
#
# Validates MixedInnerOptimizer running inside MixedShardingOptimizer
# (Sharding Stage1 V3). The two sub-optimizers share one update rule but
# declare opposite class-level ``keep_whole``, so the same model exercises
# both partitions at once: V1 (whole tensor on one owner rank) and V2
# (element-wise reduce-scatter shards). Their state layout is deliberately
# NOT AdamW's (velocity / step_count), which is what the generic
# ``optimizer_state_layout()`` contract exists for.
# Topology: sharding_degree=2, dp/mp/pp=1 (2 ranks total).

import contextlib
import os
import random
import types
import unittest
import weakref

import numpy as np

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers import (
    mixed_sharding_optimizer as msopt,
)
from paddle.distributed.fleet.meta_optimizers.mixed_sharding_optimizer import (
    MixedShardingOptimizer,
)
from paddle.distributed.fleet.utils import mix_precision_utils
from paddle.optimizer.inner_optimizer import InnerOptimizer
from paddle.optimizer.mixed_inner_optimizer import MixedInnerOptimizer

SHARDING_DEGREE = 2
DIM = 64
BATCH_SIZE = 8
STEPS = 3
LR = 0.01
STATE_SUFFIXES = {"velocity_0", "step_count_0"}
LAYERS = ("whole_fc", "slice_fc", "whole2_fc")

g_enable_fuse_optimizer_states = int(
    os.environ.get("ENABLE_FUSE_OPTIMIZER_STATES", "0")
)
g_comm_overlap = int(os.environ.get("COMM_OVERLAP", "0"))


class _MomentumSGD(InnerOptimizer):
    """Element-wise momentum SGD with a vector and a scalar state.

    ``keep_whole`` is deliberately left undeclared: the two subclasses below
    declare it, and this class itself doubles as the "forgot to declare" case.
    """

    def __init__(self, parameters, momentum=0.9, **kwargs):
        super().__init__(parameters=parameters, **kwargs)
        self._momentum = momentum

    def _ensure_accumulators(self, param):
        if (
            "velocity" in self._accumulators
            and param.name in self._accumulators["velocity"]
        ):
            return
        self._ensure_master_weight(param)
        self._add_accumulator(
            "velocity", param, dtype=paddle.float32, shape=param.shape
        )
        self._add_accumulator(
            "step_count", param, dtype=paddle.float32, shape=[1]
        )

    def _update_parameters(self, params_grads, lr):
        with paddle.no_grad():
            for param, grad in params_grads:
                v = self._get_accumulator("velocity", param)
                s = self._get_accumulator("step_count", param)
                paddle.assign(self._momentum * v + grad.astype(v.dtype), v)
                paddle.assign(s + 1.0, s)
                step = self._effective_lr(param, lr) * v
                param.subtract_(step.astype(param.dtype))

    def optimizer_state_layout(self):
        return {
            "vector": ["velocity_0"],
            "scalar": ["step_count_0"],
            "master": "fp32_master_0",
        }


class _WholeSGD(_MomentumSGD):
    """Same math, declared V1 (whole tensor on one owner rank)."""

    keep_whole = True


class _SlicedSGD(_MomentumSGD):
    """Same math, declared V2 (element-wise shards)."""

    keep_whole = False


class _Net(paddle.nn.Layer):
    def __init__(self, weights):
        super().__init__()
        self.whole_fc = self._fc(weights[0])
        self.slice_fc = self._fc(weights[1])
        self.whole2_fc = self._fc(weights[2])

    @staticmethod
    def _fc(np_w):
        return paddle.nn.Linear(
            DIM,
            DIM,
            bias_attr=False,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_w)
            ),
        )

    def forward(self, x):
        return self.whole2_fc(self.slice_fc(self.whole_fc(x))).mean()


def _mixed_optimizer(net):
    """Two subs over the same model: V1 for whole_fc/whole2_fc, V2 for slice_fc.

    Different ``momentum`` per sub: hyperparameters are frozen per group at
    construction, which is the point of the mixed optimizer.
    """
    sub_whole = _WholeSGD(
        parameters=[net.whole_fc.weight, net.whole2_fc.weight],
        learning_rate=LR,
        weight_decay=0.0,
    )
    sub_sliced = _SlicedSGD(
        parameters=[net.slice_fc.weight],
        learning_rate=LR,
        weight_decay=0.0,
        momentum=0.5,
    )
    return MixedInnerOptimizer([sub_whole, sub_sliced], learning_rate=LR)


@contextlib.contextmanager
def _patched(obj, name, value):
    """Temporarily set ``obj.name``, restoring the previous state exactly.

    Attributes that did not exist before (monkeypatched onto the process-wide
    ``HybridCommunicateGroup`` singleton) are removed again rather than left
    behind as ``None``.
    """
    had_own = name in getattr(obj, "__dict__", {})
    old = getattr(obj, name, None)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if had_own:
            setattr(obj, name, old)
        else:
            try:
                delattr(obj, name)
            except (AttributeError, TypeError):
                setattr(obj, name, old)


@contextlib.contextmanager
def _env(name, value):
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


class TestDistShardingMixedTraining(unittest.TestCase):
    def setUp(self):
        random.seed(2026)
        np.random.seed(2026)
        paddle.seed(2026)

        self.strategy = fleet.DistributedStrategy()
        self.strategy.hybrid_configs = {
            "sharding_degree": SHARDING_DEGREE,
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
        }
        sharding_configs = self.strategy.hybrid_configs["sharding_configs"]
        sharding_configs.accumulate_steps = 1
        sharding_configs.comm_buffer_size_MB = 1
        sharding_configs.split_param = True
        if g_enable_fuse_optimizer_states:
            sharding_configs.enable_fuse_optimizer_states = True
        if g_comm_overlap:
            sharding_configs.comm_overlap = True
        fleet.init(is_collective=True, strategy=self.strategy)
        self.hcg = fleet.get_hybrid_communicate_group()
        self.weights = [
            np.random.random_sample((DIM, DIM)).astype("float32")
            for _ in range(len(LAYERS))
        ]
        self.data = [
            np.random.random_sample((BATCH_SIZE, DIM)).astype("float32")
            for _ in range(STEPS + 1)
        ]

    @staticmethod
    def _build_sharded(weights):
        """Sharded model plus the shell that wraps the mixed optimizer.

        ``MixPrecisionLayer`` gives every parameter a ``main_grad``. The
        element-wise (V2) partition needs it: those gradients live inside a
        ``FusedCommBuffer``, and only ``clear_grad``'s ``main_grad`` branch
        zeroes them in place -- ``clear_gradient()`` hands the parameter a
        fresh tensor and the reduce-scatter then sees zeros. The
        ``split_param`` cases of ``hybrid_parallel_sharding_model.py`` are
        main_grad-only for the same reason. Parameters stay fp32, so the
        comparison against the reference below is still exact.
        """
        net = _Net(weights)
        optimizer = _mixed_optimizer(net)
        model = fleet.distributed_model(
            mix_precision_utils.MixPrecisionLayer(net)
        )
        return net, model, fleet.distributed_optimizer(optimizer)

    @staticmethod
    def _build_reference(weights):
        """Single-card model with the same main_grad setup, no shell."""
        net = _Net(weights)
        optimizer = _mixed_optimizer(net)
        model = mix_precision_utils.MixPrecisionLayer(net)
        return net, model, mix_precision_utils.MixPrecisionOptimizer(optimizer)

    def _step_sharded(self, model, optimizer, idx):
        rank = self.hcg.get_sharding_parallel_rank()
        local_bs = BATCH_SIZE // SHARDING_DEGREE
        start = rank * local_bs
        batch = paddle.to_tensor(self.data[idx][start : start + local_bs])
        model(batch).backward()
        optimizer.step()
        optimizer.clear_grad()

    def _step_reference(self, model, optimizer, idx):
        """One step on the whole batch, single card, no sharding shell."""
        model(paddle.to_tensor(self.data[idx])).backward()
        optimizer.step()
        optimizer.clear_grad()

    @staticmethod
    def _accumulator_values(shell):
        return {
            var.name: var.numpy().copy()
            for accs in shell._inner_opt.merged_accumulators().values()
            for var in accs.values()
        }

    def test_mixed_sharding(self):
        net, model, optimizer = self._build_sharded(self.weights)
        shell = optimizer._inner_opt

        # 0. HybridParallelOptimizer selects the shell from the inner
        # optimizer's type alone -- no strategy flag is involved.
        self.assertIsInstance(shell, MixedShardingOptimizer)

        whole = net.whole_fc.weight.name
        sliced = net.slice_fc.weight.name
        whole2 = net.whole2_fc.weight.name

        # 1. V1 / V2 partition follows each sub's declared ``keep_whole``.
        v1 = {
            p.name
            for params in shell._params_2d_by_color.values()
            for p in params
        }
        v2 = set()
        for buffer in shell._comm_buffer_list:
            v2.update(buffer._sharding_param_grad_view.keys())
        self.assertEqual(v1, {whole, whole2})
        self.assertEqual(v2, {sliced})

        # 2. Every rank agrees on which rank owns which whole tensor.
        owner_map = {
            str(color): dict(name2rank)
            for color, name2rank in shell._param2rank_2d_by_color.items()
        }
        gathered = []
        paddle.distributed.all_gather_object(gathered, owner_map)
        for other in gathered:
            self.assertEqual(other, gathered[0])

        # 3. Sharded training reproduces the single-card reference step by
        # step, and every rank ends each step with the same parameters.
        ref_net, ref_model, ref_optimizer = self._build_reference(self.weights)
        for idx in range(STEPS):
            self._step_sharded(model, optimizer, idx)
            self._step_reference(ref_model, ref_optimizer, idx)
            for param, ref in zip(net.parameters(), ref_net.parameters()):
                np.testing.assert_allclose(
                    param.numpy(),
                    ref.numpy(),
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"{param.name} differs from the reference at "
                    f"step {idx}",
                )
            for name, param in net.named_parameters():
                values = []
                paddle.distributed.all_gather(values, param)
                for other in values:
                    np.testing.assert_array_equal(
                        other.numpy(), values[0].numpy(), err_msg=name
                    )

        # 4. state_dict / set_state_dict round-trip. The accumulator names come
        # from this optimizer's own layout, not from AdamW's.
        state = shell.state_dict()
        saved = self._accumulator_values(shell)
        self.assertTrue(saved)
        for key in saved:
            self.assertTrue(
                any(key.endswith("_" + s) for s in STATE_SUFFIXES), key
            )
        snapshot = {k: v.clone() for k, v in state.items()}
        self._step_sharded(model, optimizer, STEPS)
        shell.set_state_dict(snapshot)
        restored = self._accumulator_values(shell)
        self.assertEqual(set(restored), set(saved))
        for key, value in saved.items():
            np.testing.assert_array_equal(restored[key], value, err_msg=key)

        # 5. sharded_state_dict (flex checkpoint) uses the custom suffixes; the
        # V2 parameter's state is flattened, and the union over ranks covers
        # every (layer, state) pair exactly once.
        sharded = shell.sharded_state_dict(net.sharded_state_dict())
        for key, weight in sharded.items():
            self.assertIn(key.rsplit(".", 1)[1], STATE_SUFFIXES, key)
            if key == "slice_fc.weight.velocity_0":
                self.assertTrue(weight.is_flattened)
        all_keys = []
        paddle.distributed.all_gather_object(all_keys, sorted(sharded))
        self.assertEqual(
            {k for keys in all_keys for k in keys},
            {f"{layer}.weight.{s}" for layer in LAYERS for s in STATE_SUFFIXES},
        )

        # 6. The remaining sharded_state_dict branches: fp32 master weights for
        # both partitions (the V1 one stored flat, so it is reshaped back),
        # grouped-GEMM reshaping, and states whose owner is not in the model.
        sub_whole, sub_sliced = shell._inner_opt._sub_opts
        sub_whole._master_weights = {
            net.whole_fc.weight.name: paddle.zeros([DIM * DIM], "float32"),
            "not_owned_by_this_sub": paddle.zeros([1], "float32"),
        }
        sub_sliced._master_weights = {
            net.slice_fc.weight.name: paddle.zeros(
                [DIM * DIM // SHARDING_DEGREE], "float32"
            )
        }
        stray = paddle.zeros([1], "float32")
        stray.name = "no_such_param_step_count_0"
        sub_whole._accumulators["step_count"]["no_such_param"] = stray
        model_state = net.sharded_state_dict()
        model_state["whole_fc.weight"].grouped_gemm_param = True
        sharded = shell.sharded_state_dict(model_state)
        self.assertEqual(
            tuple(sharded["whole_fc.weight.w_0"].local_tensor.shape),
            (DIM, DIM),
        )
        self.assertTrue(sharded["slice_fc.weight.w_0"].is_flattened)

        # An accumulator name that matches no declared suffix is an error, not
        # a silently dropped tensor.
        bogus = paddle.zeros([1], "float32")
        bogus.name = "bogus"
        sub_whole._accumulators["step_count"]["bogus_param"] = bogus
        with self.assertRaisesRegex(ValueError, "Cannot parse"):
            shell.sharded_state_dict(model_state)

    def test_module_helpers(self):
        # Pure helpers, no collectives: node-local rank arithmetic and the
        # colour bookkeeping that the MoE group-call path relies on.
        with _env("PADDLE_LOCAL_SIZE", "4"):
            self.assertEqual(msopt.get_gpus_per_node(), 4)
        with _env("PADDLE_LOCAL_SIZE", None):
            self.assertGreaterEqual(msopt.get_gpus_per_node(), 1)

        self.assertEqual(msopt.get_same_card_rank({2, 3}, 3, 2), 2)
        self.assertEqual(msopt.get_same_card_rank({8}, 0, 2), -1)
        self.assertEqual(msopt.get_trainer_ranks(5, 4), (1, range(4, 8)))

        group = self.hcg.get_sharding_parallel_group()
        plain = types.SimpleNamespace()
        colored = types.SimpleNamespace(color={"color": "c1", "group": group})
        info = (
            msopt.MixedShardingOptimizer._build_color_to_group_info_from_params(
                [plain, colored], group
            )
        )
        self.assertEqual(set(info), {None, "c1"})

        self.assertIsNone(
            msopt.MixedShardingOptimizer._buffer_color(
                types.SimpleNamespace(params=[])
            )
        )
        self.assertEqual(
            msopt.MixedShardingOptimizer._buffer_color(
                types.SimpleNamespace(params=[colored])
            ),
            "c1",
        )
        self.assertEqual(
            msopt.MixedShardingOptimizer._buffer_color(
                types.SimpleNamespace(
                    params=[types.SimpleNamespace(color="c2")]
                )
            ),
            "c2",
        )
        self.assertEqual(
            msopt.MixedShardingOptimizer._compute_shared_colors(
                [types.SimpleNamespace(color={"color": "dense_weight"})]
            ),
            set(),
        )

    def test_init_contract(self):
        # Every user-facing configuration that the shell rejects, plus the two
        # optional construction paths (custom color / timers).
        hybrid = fleet.fleet._user_defined_strategy.strategy.hybrid_configs
        sharding_configs = hybrid.sharding_configs
        pp_configs = hybrid.pp_configs

        with self.assertRaisesRegex(TypeError, "param_groups"):
            MixedShardingOptimizer(
                types.SimpleNamespace(_parameter_list=[{}]), self.hcg
            )
        with self.assertRaisesRegex(ValueError, "_apply_optimize"):
            MixedShardingOptimizer(
                types.SimpleNamespace(
                    _parameter_list=[_Net(self.weights).whole_fc.weight]
                ),
                self.hcg,
            )

        fake_hcg = types.SimpleNamespace(
            get_sharding_parallel_world_size=lambda: SHARDING_DEGREE,
            get_sharding_parallel_rank=lambda: 0,
            get_model_parallel_world_size=lambda: 2,
        )
        with self.assertRaisesRegex(ValueError, "tensor parallelism"):
            MixedShardingOptimizer(
                _mixed_optimizer(_Net(self.weights)), fake_hcg
            )

        # A parameter that no sub-optimizer owns is reported by name.
        orphan = _mixed_optimizer(_Net(self.weights))
        orphan._parameter_list = [
            *orphan._parameter_list,
            _Net(self.weights).whole_fc.weight,
        ]
        with self.assertRaisesRegex(ValueError, "not owned by any mixed"):
            MixedShardingOptimizer(orphan, self.hcg)

        with (
            _patched(pp_configs, "sharding_comm_overlap", True),
            self.assertRaisesRegex(ValueError, "PP overlap"),
        ):
            MixedShardingOptimizer(
                _mixed_optimizer(_Net(self.weights)), self.hcg
            )
        with (
            _patched(hybrid, "pp_degree", 2),
            _patched(sharding_configs, "comm_overlap", True),
            self.assertRaisesRegex(ValueError, "pipeline parallel"),
        ):
            MixedShardingOptimizer(
                _mixed_optimizer(_Net(self.weights)), self.hcg
            )
        with (
            _patched(sharding_configs, "comm_overlap", True),
            _patched(sharding_configs, "accumulate_steps", 0),
            self.assertRaisesRegex(ValueError, "accumulate_steps"),
        ):
            MixedShardingOptimizer(
                _mixed_optimizer(_Net(self.weights)), self.hcg
            )

        # group_call_opt needs a MoE topology; on this plain one every guard
        # fires in turn.
        with (
            _patched(sharding_configs, "comm_group_call_opt", True),
            _env("PADDLE_LOCAL_SIZE", "2"),
        ):
            with self.assertRaisesRegex(ValueError, "comm_group_call_opt"):
                MixedShardingOptimizer(
                    _mixed_optimizer(_Net(self.weights)), self.hcg
                )
            with (
                _patched(self.hcg, "get_expert_parallel_world_size", lambda: 2),
                _patched(
                    self.hcg, "get_moe_sharding_parallel_world_size", lambda: 2
                ),
                _patched(
                    self.hcg,
                    "_moe_topo",
                    types.SimpleNamespace(
                        _parallel_names=["pipe", "moe_sharding", "expert"]
                    ),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "MoE topology order"):
                    MixedShardingOptimizer(
                        _mixed_optimizer(_Net(self.weights)), self.hcg
                    )
                with (
                    _patched(
                        self.hcg,
                        "_moe_topo",
                        types.SimpleNamespace(
                            _parallel_names=["moe_sharding", "pipe", "expert"]
                        ),
                    ),
                    _patched(
                        msopt, "is_avg_reduce_op_supported", lambda: False
                    ),
                    _patched(sharding_configs, "use_reduce_avg", True),
                    self.assertRaisesRegex(ValueError, "use_reduce_avg"),
                ):
                    MixedShardingOptimizer(
                        _mixed_optimizer(_Net(self.weights)), self.hcg
                    )

        # Custom per-parameter color plus a frozen parameter, and the optional
        # optimizer timers.
        net = _Net(self.weights)
        net.whole_fc.weight.trainable = False
        colored = {
            "color": "c1",
            "group": self.hcg.get_sharding_parallel_group(),
        }
        net.slice_fc.weight.color = colored
        net.whole2_fc.weight.color = colored
        with _patched(hybrid, "enable_optimizer_timer", True):
            shell = MixedShardingOptimizer(_mixed_optimizer(net), self.hcg)
        self.assertEqual(set(shell._params_2d_by_color), {"c1"})

        # A color whose group is None degenerates to "everything on rank 0".
        net = _Net(self.weights)
        solo = {"color": "solo", "group": None}
        net.whole_fc.weight.color = solo
        net.whole2_fc.weight.color = solo
        with _patched(sharding_configs, "comm_buffer_size_MB", 0):
            shell = MixedShardingOptimizer(_mixed_optimizer(net), self.hcg)
        self.assertEqual(
            shell._param2rank_2d_by_color["solo"],
            {net.whole_fc.weight.name: 0, net.whole2_fc.weight.name: 0},
        )

    def test_machine_balanced_partition(self):
        # End-to-end smoke test of the machine-balanced path: the two
        # all_gather_object calls (hostnames and per-color parameter info) run
        # on the global world, and every rank plans over the same gathered
        # information, so the owner map must come out identical everywhere.
        # The assignment itself is unit-tested without a launcher in
        # test/legacy_test/test_mixed_2d_partition.py; the two small whole
        # params here need one owner each, so the mapping matches what the
        # per-color partitioner produces.
        hybrid = fleet.fleet._user_defined_strategy.strategy.hybrid_configs
        sharding_configs = hybrid.sharding_configs
        net = _Net(self.weights)
        with (
            _patched(
                msopt.socket,
                "gethostname",
                lambda: f"host{self.hcg.get_sharding_parallel_rank()}",
            ),
            _patched(sharding_configs, "machine_balanced_2d_partition", True),
        ):
            shell = MixedShardingOptimizer(_mixed_optimizer(net), self.hcg)
        self.assertEqual(
            shell._param2rank_2d_by_color[None],
            {net.whole_fc.weight.name: 0, net.whole2_fc.weight.name: 0},
        )

    def test_shell_plumbing(self):
        # The shell entry points that a real training loop reaches only under
        # configurations the three ctest cases do not cover (non-fused
        # gradients, released gradients, adaptive-norm clipping, parameter
        # storage recycling). They are driven directly here.
        if g_comm_overlap:
            # These entry points are driven by hand, without the backward
            # hooks that feed the overlapped communication.
            self.skipTest("plumbing is driven without the backward hooks")
        net, model, optimizer = self._build_sharded(self.weights)
        shell = optimizer._inner_opt
        sharding_group = self.hcg.get_sharding_parallel_group()

        # Without gradient fusion the 2D params are reduced one by one, and a
        # parameter whose gradient was never produced gets a zero one.
        shell._use_fuse_gradients = False
        shell.reduce_gradients(shell._parameter_list, self.hcg)
        shell._use_fuse_gradients = True

        # The same zero-filling on a shell whose params have no main_grad, and
        # the frozen-parameter short circuit.
        plain_net = _Net(self.weights)
        plain = MixedShardingOptimizer(_mixed_optimizer(plain_net), self.hcg)
        plain._reduce_2d_grads(
            plain._params_2d_by_color[None],
            plain._param2rank_2d_by_color[None],
            sharding_group,
        )
        plain_net.whole_fc.weight.trainable = False
        self.assertIsNone(plain._get_param_grad(plain_net.whole_fc.weight))

        # In align mode the gradients are pre-scaled before the collective, so
        # the reduce must use SUM even though use_reduce_avg asks for AVG --
        # the same contract as MuonShardingOptimizer and as the 1D path inside
        # FusedCommBuffer.
        ops = []
        with (
            _patched(
                paddle.distributed,
                "in_auto_parallel_align_mode",
                lambda: True,
            ),
            _patched(
                paddle.distributed,
                "reduce",
                lambda tensor, dst, op, group, sync_op: ops.append(op),
            ),
        ):
            shell._reduce_2d_grads(
                shell._params_2d_by_color[None],
                shell._param2rank_2d_by_color[None],
                sharding_group,
            )
        self.assertEqual(set(ops), {paddle.distributed.ReduceOp.SUM})

        # Buffers are rediscovered from the parameters' weak references when
        # the list was dropped (the PP-overlap handover).
        buffers = list(shell._comm_buffer_list)
        shell._comm_buffer_list = []
        shell.param2bucket = {}
        shell._collect_comm_buffers()
        self.assertEqual(shell._comm_buffer_list, buffers)
        # The references are consumed, so a second round finds nothing.
        shell._comm_buffer_list = []
        shell._collect_comm_buffers()
        self.assertEqual(shell._comm_buffer_list, [])
        for buffer in buffers:
            for param in buffer.params:
                param.comm_buffer_ref = weakref.ref(buffer)
        shell._collect_comm_buffers()
        self.assertEqual(shell._comm_buffer_list, buffers)

        # Adaptive-norm clipping needs the sharding layout; the second call is
        # a no-op because the layout is already attached.
        from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.hybrid_parallel_optimizer import (
            ClipGradByAdaptiveNorm,
        )

        clip = ClipGradByAdaptiveNorm()
        with _patched(shell._inner_opt, "_grad_clip", clip):
            shell.set_grad_clip_info()
            shell.set_grad_clip_info()
        self.assertTrue(clip.sharding_stage1_v2)

        # set_state_dict also routes the scheduler and the master weights.
        with _patched(shell._inner_opt, "set_state_dict", lambda sd: None):
            shell.set_state_dict(
                {
                    "LR_Scheduler": {"last_epoch": 0},
                    "master_weights": {
                        p.name: paddle.zeros_like(p)
                        for p in shell._parameter_list
                    },
                }
            )
        with self.assertRaisesRegex(TypeError, "attr_name"):
            shell._set_inner_opt_attr(1, 2)

        # Flattening several same-dtype params of one owner into one buffer,
        # and the empty-owner short circuit.
        spare = _Net(self.weights)
        flat = shell.reorder_params(
            [
                (
                    {0: [spare.whole_fc.weight, spare.whole2_fc.weight]},
                    sharding_group,
                )
            ]
        )
        self.assertEqual(len(flat), 1)
        self.assertEqual(
            shell.reorder_params(
                [({0: [], 1: [spare.slice_fc.weight]}, sharding_group)]
            )[0]["src_rank"],
            sharding_group.ranks[1],
        )

        # clear_grad: the release-the-tensor variant for main_grad, both
        # tensor-fusion variants for plain gradients, and the comm buffers'
        # own gradient storage.
        shell.clear_grad(set_to_zero=False)
        plain_net.whole_fc.weight.trainable = True
        plain_net(paddle.to_tensor(self.data[0])).backward()
        with _patched(plain, "tensor_fusion", True):
            plain.clear_grad()
            plain.clear_grad(set_to_zero=False)
        plain_net(paddle.to_tensor(self.data[0])).backward()
        with _patched(plain, "tensor_fusion", False):
            plain.clear_grad()
        with _patched(shell, "sd_release_grads", True):
            shell.clear_grad()

        # Parameter storage recycling (ZCC): drop the storage of one color and
        # bring it back. Needs every parameter to carry that color.
        colored_net = _Net(self.weights)
        tag = {"color": "c1", "group": sharding_group}
        for p in colored_net.parameters():
            p.color = tag
        colored = MixedShardingOptimizer(
            _mixed_optimizer(colored_net), self.hcg
        )
        colored.clear_param_storage("c1")
        colored.clear_param_storage(None)
        colored.reset_param_storage()
        self.assertEqual(colored.clear_color, {"c1", None})
        for p in colored_net.parameters():
            self.assertTrue(p._is_initialized())

    def test_group_call_opt(self):
        # The MoE group-call communication path: a two-level reduce /
        # broadcast over per-machine and per-card-index groups. This topology
        # has neither, so the groups are faked and the collectives themselves
        # are replaced by no-ops -- the point is that the index arithmetic and
        # the buffer classification run.
        if not msopt.is_avg_reduce_op_supported():
            self.skipTest("group_call_opt requires nccl reduce_avg support")
        if g_comm_overlap:
            self.skipTest("the reduce is driven without the backward hooks")
        hybrid = fleet.fleet._user_defined_strategy.strategy.hybrid_configs
        sharding_configs = hybrid.sharding_configs
        moe_group = paddle.distributed.new_group([0, 1])
        pair_group = paddle.distributed.new_group([0, 1])

        net = _Net(self.weights)
        net.whole2_fc.weight.color = {"color": "moe", "group": moe_group}
        with (
            _patched(sharding_configs, "comm_group_call_opt", True),
            _patched(sharding_configs, "use_reduce_avg", True),
            _env("PADDLE_LOCAL_SIZE", "2"),
            _patched(self.hcg, "get_expert_parallel_world_size", lambda: 2),
            _patched(
                self.hcg, "get_moe_sharding_parallel_world_size", lambda: 2
            ),
            _patched(
                self.hcg, "get_moe_sharding_parallel_group", lambda: moe_group
            ),
            _patched(
                self.hcg,
                "_moe_topo",
                types.SimpleNamespace(
                    _parallel_names=["moe_sharding", "pipe", "expert"]
                ),
            ),
        ):
            with _patched(
                paddle.distributed, "new_group", lambda ranks, **kw: pair_group
            ):
                shell = MixedShardingOptimizer(_mixed_optimizer(net), self.hcg)
            self.assertEqual(shell.trainer_comms, {0: pair_group})
            with (
                _patched(
                    paddle.distributed.stream, "reduce", lambda *a, **kw: None
                ),
                _patched(
                    paddle.distributed.stream,
                    "broadcast",
                    lambda *a, **kw: None,
                ),
            ):
                shell.reduce_gradients(shell._parameter_list, self.hcg)
                shell._sharding_sync_parameters()

    def test_requires_declared_keep_whole(self):
        # A sub that never declares keep_whole fails as soon as it is built,
        # not deep inside the shell's classification loop.
        net = _Net(self.weights)
        with self.assertRaisesRegex(TypeError, "keep_whole"):
            _MomentumSGD(
                parameters=[net.slice_fc.weight],
                learning_rate=LR,
                weight_decay=0.0,
            )

    def test_requires_weight_decay(self):
        # weight_decay has no safe default, so it must be passed explicitly.
        with self.assertRaisesRegex(ValueError, "weight_decay"):
            _WholeSGD(
                parameters=_Net(self.weights).parameters(), learning_rate=LR
            )

    def test_requires_mixed_inner_optimizer(self):
        # The shell is only valid around a MixedInnerOptimizer.
        net = _Net(self.weights)
        plain = _WholeSGD(
            parameters=net.parameters(),
            learning_rate=LR,
            weight_decay=0.0,
        )
        with self.assertRaisesRegex(TypeError, "MixedInnerOptimizer"):
            MixedShardingOptimizer(plain, self.hcg)


if __name__ == "__main__":
    unittest.main()
