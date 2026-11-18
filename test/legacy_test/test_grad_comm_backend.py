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

"""Single-process tests for the gradient communication backend extension point.

``tensor_fusion_helper.register_grad_comm_backend`` registers a factory that the
sharding optimizers query once, at the end of their ``__init__``. The backend it
returns may take over the gradient communication of the optimizer's
``FusedCommBuffer`` buckets, the gradient reduction and the parameter sync.

Both directions are checked: with no factory registered nothing changes (the
bucket keeps its defaults and the original collective is still issued), and with
one the backend is installed and the built-in code path is skipped exactly when
the backend reports that it handled the call.

Comm groups, collectives and the fleet strategy are stubbed with local
stand-ins, so neither a launcher nor an accelerator is needed (same style as
test_sharding_2d_bucket_key.py).
"""

import contextlib
import types
import unittest
from unittest import mock

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers import muon_sharding_optimizer
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer import (
    dygraph_sharding_optimizer,
)
from paddle.distributed.fleet.utils import tensor_fusion_helper
from paddle.distributed.fleet.utils.tensor_fusion_helper import (
    HOOK_ACTION,
    FusedCommBuffer,
    get_grad_comm_backend_factory,
    register_grad_comm_backend,
)

MODULES = (dygraph_sharding_optimizer, muon_sharding_optimizer)


def _optimizer_cls(module):
    if module is muon_sharding_optimizer:
        return module.MuonShardingOptimizer
    return module.DygraphShardingOptimizerV2


class _Backend:
    """A fake gradient communication backend that records its calls.

    ``handled`` is what ``reduce_gradients`` / ``sync_parameters`` return, i.e.
    whether the backend claims to have fully handled the call; ``setup_result``
    is whether it takes the optimizer over at all.
    """

    def __init__(self, handled=True, setup_result=True):
        self.handled = handled
        self.setup_result = setup_result
        self.task = types.SimpleNamespace(name="backend_task")
        self.calls = []

    def setup(self, optimizer):
        self.calls.append(("setup", optimizer))
        return self.setup_result

    def comm_grads(self, bucket, reduce_op):
        self.calls.append(("comm_grads", bucket, reduce_op))
        return self.task

    def reduce_gradients(self, optimizer, parameter_list, hcg):
        self.calls.append(("reduce_gradients", optimizer, parameter_list, hcg))
        return self.handled

    def sync_parameters(self, optimizer):
        self.calls.append(("sync_parameters", optimizer))
        return self.handled

    def rebuild(self, optimizer):
        self.calls.append(("rebuild", optimizer))

    def names(self):
        return [call[0] for call in self.calls]


class _Factory:
    """Records the optimizers it is called with, returns ``backend``."""

    def __init__(self, backend):
        self.backend = backend
        self.optimizers = []

    def __call__(self, optimizer):
        self.optimizers.append(optimizer)
        return self.backend


class _CommBufferSpy:
    """Stands in for a FusedCommBuffer in the optimizer level tests."""

    def __init__(self):
        self.calls = []
        self.grad_storage = None
        self._dst = 0

    def need_reduce_scale_sync(self):
        return True

    def _comm_grads(self):
        self.calls.append("_comm_grads")

    def scale_grads(self):
        self.calls.append("scale_grads")

    def sync_params(self):
        self.calls.append("sync_params")


def _group(nranks=2, rank=0, group_id=0):
    return types.SimpleNamespace(
        nranks=nranks, rank=rank, ranks=list(range(nranks)), _id=group_id
    )


def _hcg(world_size=2, rank=0):
    """A hybrid communicate group with the getters the optimizers call."""
    group = _group(nranks=world_size, rank=rank)
    return types.SimpleNamespace(
        get_sharding_parallel_world_size=lambda: world_size,
        get_sharding_parallel_rank=lambda: rank,
        get_sharding_parallel_group=lambda: group,
        get_model_parallel_world_size=lambda: 1,
        get_expert_parallel_world_size=lambda: 1,
        get_moe_sharding_parallel_world_size=lambda: 1,
        get_pipe_parallel_world_size=lambda: 1,
        get_stage_id=lambda: 0,
    )


def _inner_optimizer(param):
    return types.SimpleNamespace(
        _parameter_list=[param],
        _param_groups=[param],
        _apply_optimize=lambda *args, **kwargs: None,
    )


def _frozen_param():
    """A parameter no optimizer owns.

    ``stop_gradient`` keeps it out of the 2D / 1D buckets, so the optimizers
    can be built without any real communication or fused storage.
    """
    param = paddle.create_parameter([4, 4], dtype="float32")
    param.stop_gradient = True
    return param


def _strategy(sharding_comm_overlap):
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "sharding_degree": 2,
        "pp_configs": {"sharding_comm_overlap": sharding_comm_overlap},
    }
    # reduce_avg falls back to SUM when it is unsupported, keep it off so the
    # construction does not depend on the nccl version. The setter does not
    # take sharding_configs, so write the protobuf field directly.
    strategy.hybrid_configs["sharding_configs"].use_reduce_avg = False
    return strategy


@contextlib.contextmanager
def _sharding_env(module, hcg):
    """Make a sharding optimizer see a stub topology and a default strategy.

    ``DygraphShardingOptimizerV2`` is built with ``sharding_comm_overlap``, so
    that the comm buffers (and the storage they allocate) are left to the
    pipeline; ``MuonShardingOptimizer`` forbids that combination and is built
    with no bucketed parameter instead. Both reach the end of ``__init__``,
    which is where the backend factory is queried.
    """
    is_muon = module is muon_sharding_optimizer
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                module.fleet.fleet,
                "_user_defined_strategy",
                _strategy(sharding_comm_overlap=not is_muon),
                create=True,
            )
        )
        stack.enter_context(
            mock.patch.object(
                module.fleet, "get_hybrid_communicate_group", lambda: hcg
            )
        )
        stack.enter_context(
            mock.patch.object(
                paddle.distributed,
                "all_gather",
                lambda out, tensor, group=None, sync_op=True: out.append(
                    tensor
                ),
            )
        )
        yield


def _bare(cls, **attrs):
    """An instance without __init__, carrying only ``attrs``."""
    opt = object.__new__(cls)
    # __getattr__ forwards to _inner_opt; set it so that a missing attribute
    # raises AttributeError instead of recursing.
    opt._inner_opt = None
    for name, value in attrs.items():
        setattr(opt, name, value)
    return opt


def _dispatch_optimizer(module, backend, buffer):
    """A shell optimizer carrying only what the dispatched methods read."""
    attrs = {
        "_grad_comm_backend": backend,
        "_comm_buffer_list": [buffer],
        "sd_release_grads": False,
        "comm_overlap": False,
        "_enable_timer": False,
        "_hcg": _hcg(),
    }
    if module is muon_sharding_optimizer:
        attrs.update(
            _use_fuse_gradients=False,
            _params_2d_by_color={},
            _rank2params_2d_by_color={},
            _color_to_group_info={},
            _sync_only_buffer_ids=set(),
            use_group_call_opt=False,
        )
    else:
        attrs.update(_all_gather_overlap_forward=False)
    return _bare(_optimizer_cls(module), **attrs)


class TestGradCommBackendRegistry(unittest.TestCase):
    def setUp(self):
        self.addCleanup(register_grad_comm_backend, None)

    def test_no_backend_is_registered_by_default(self):
        self.assertIsNone(get_grad_comm_backend_factory())

    def test_register_and_unregister(self):
        factory = _Factory(_Backend())
        register_grad_comm_backend(factory)
        self.assertIs(get_grad_comm_backend_factory(), factory)
        register_grad_comm_backend(None)
        self.assertIsNone(get_grad_comm_backend_factory())


class TestFusedCommBufferBackend(unittest.TestCase):
    def setUp(self):
        # Tensor fusion is implemented for the accelerators only, and the
        # buckets below are built without grad storage, so the device type is
        # read for the buffer alignment and nothing else. Pin it, and the test
        # runs the same way on a CPU-only build.
        patcher = mock.patch.object(
            tensor_fusion_helper,
            "get_current_device_type",
            return_value="gpu",
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _bucket(self, use_reduce_avg=False):
        """A real bucket, built without grad storage (release_grads)."""
        linear = paddle.nn.Linear(4, 4)
        return FusedCommBuffer(
            id=0,
            params=[linear.weight, linear.bias],
            comm_group=_group(nranks=2),
            acc_steps=1,
            act=HOOK_ACTION.ALL_REDUCE,
            release_grads=True,
            use_reduce_avg=use_reduce_avg,
        )

    def test_bucket_defaults_to_no_backend(self):
        bucket = self._bucket()
        self.assertIsNone(bucket._grad_comm_backend)
        self.assertEqual(bucket._post_reduce_scale, 1.0)

    def test_comm_grads_without_backend_still_all_reduces(self):
        bucket = self._bucket()
        task = types.SimpleNamespace(name="all_reduce_task")
        with (
            mock.patch.object(
                paddle.distributed, "all_reduce", return_value=task
            ) as all_reduce,
            mock.patch.object(
                paddle.distributed, "reduce_scatter"
            ) as reduce_scatter,
        ):
            bucket._comm_grads()

        self.assertEqual(all_reduce.call_count, 1)
        self.assertEqual(
            all_reduce.call_args.kwargs["op"],
            paddle.distributed.ReduceOp.SUM,
        )
        reduce_scatter.assert_not_called()
        self.assertIs(bucket._task, task)

    def test_comm_grads_without_backend_still_reduce_scatters(self):
        """The sharding buckets use REDUCE_SCATTER, which slices the storage,
        so give this bucket a real one."""
        bucket = self._bucket()
        bucket._act = HOOK_ACTION.REDUCE_SCATTER
        bucket.grad_storage = paddle.zeros([8], dtype="float32")
        task = types.SimpleNamespace(name="reduce_scatter_task")
        with (
            mock.patch.object(
                paddle.distributed, "reduce_scatter", return_value=task
            ) as reduce_scatter,
            mock.patch.object(paddle.distributed, "all_reduce") as all_reduce,
        ):
            bucket._comm_grads()

        self.assertEqual(reduce_scatter.call_count, 1)
        all_reduce.assert_not_called()
        self.assertIs(bucket._task, task)

    def test_backend_replaces_the_collective(self):
        cases = (
            (False, paddle.distributed.ReduceOp.SUM),
            (True, paddle.distributed.ReduceOp.AVG),
        )
        for use_reduce_avg, expected_op in cases:
            with self.subTest(use_reduce_avg=use_reduce_avg):
                backend = _Backend()
                bucket = self._bucket(use_reduce_avg=use_reduce_avg)
                bucket._grad_comm_backend = backend
                with (
                    mock.patch.object(
                        paddle.distributed, "all_reduce"
                    ) as all_reduce,
                    mock.patch.object(
                        paddle.distributed, "reduce_scatter"
                    ) as reduce_scatter,
                    mock.patch.object(paddle.distributed, "reduce") as reduce,
                ):
                    bucket._comm_grads()

                all_reduce.assert_not_called()
                reduce_scatter.assert_not_called()
                reduce.assert_not_called()
                # The backend gets the bucket and the reduce op the collective
                # would have used, and its task is what scale_grads waits on.
                self.assertEqual(backend.names(), ["comm_grads"])
                self.assertIs(backend.calls[0][1], bucket)
                self.assertEqual(backend.calls[0][2], expected_op)
                self.assertIs(bucket._task, backend.task)


class TestOptimizerBackendInstall(unittest.TestCase):
    """The factory is queried once per optimizer, at the end of __init__."""

    def setUp(self):
        self.addCleanup(register_grad_comm_backend, None)

    def _build(self, module):
        hcg = _hcg()
        with _sharding_env(module, hcg):
            return _optimizer_cls(module)(
                _inner_optimizer(_frozen_param()), hcg
            )

    def test_no_factory_leaves_the_optimizer_unchanged(self):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                # Both classes forward unknown attributes to the inner
                # optimizer, so this also checks __init__ sets the attribute.
                self.assertIsNone(self._build(module)._grad_comm_backend)

    def test_factory_is_queried_once_and_the_backend_installed(self):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                backend = _Backend()
                factory = _Factory(backend)
                register_grad_comm_backend(factory)
                opt = self._build(module)

                self.assertEqual(len(factory.optimizers), 1)
                self.assertIs(factory.optimizers[0], opt)
                self.assertEqual(backend.names(), ["setup"])
                self.assertIs(backend.calls[0][1], opt)
                self.assertIs(opt._grad_comm_backend, backend)

    def test_backend_declining_setup_is_not_installed(self):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                backend = _Backend(setup_result=False)
                register_grad_comm_backend(_Factory(backend))
                opt = self._build(module)

                self.assertEqual(backend.names(), ["setup"])
                self.assertIsNone(opt._grad_comm_backend)

    def test_factory_returning_none_installs_nothing(self):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                factory = _Factory(None)
                register_grad_comm_backend(factory)
                opt = self._build(module)

                self.assertEqual(len(factory.optimizers), 1)
                self.assertIsNone(opt._grad_comm_backend)


class TestReduceGradientsDispatch(unittest.TestCase):
    def setUp(self):
        # Checking the bucket padding needs the reduce-scatter views of real
        # buckets; it is off by default.
        patcher = mock.patch.object(
            dygraph_sharding_optimizer, "g_sharding_v2_check_zero_padding", 0
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_skipped_when_the_backend_handles_it(self):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                backend = _Backend(handled=True)
                buffer = _CommBufferSpy()
                opt = _dispatch_optimizer(module, backend, buffer)
                parameter_list = [_frozen_param()]
                hcg = _hcg()

                opt.reduce_gradients(parameter_list, hcg)

                self.assertEqual(buffer.calls, [])
                self.assertEqual(backend.names(), ["reduce_gradients"])
                _, opt_arg, params_arg, hcg_arg = backend.calls[0]
                self.assertIs(opt_arg, opt)
                self.assertIs(params_arg, parameter_list)
                self.assertIs(hcg_arg, hcg)

    def test_runs_without_a_backend_takeover(self):
        for module in MODULES:
            for backend in (None, _Backend(handled=False)):
                with self.subTest(module=module.__name__, backend=backend):
                    buffer = _CommBufferSpy()
                    opt = _dispatch_optimizer(module, backend, buffer)

                    opt.reduce_gradients([], _hcg())

                    self.assertEqual(
                        buffer.calls, ["_comm_grads", "scale_grads"]
                    )
                    if backend is not None:
                        self.assertEqual(backend.names(), ["reduce_gradients"])


class TestSyncParametersDispatch(unittest.TestCase):
    def test_skipped_when_the_backend_handles_it(self):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                backend = _Backend(handled=True)
                buffer = _CommBufferSpy()
                opt = _dispatch_optimizer(module, backend, buffer)

                opt._sharding_sync_parameters()

                self.assertEqual(buffer.calls, [])
                self.assertEqual(backend.names(), ["sync_parameters"])
                self.assertIs(backend.calls[0][1], opt)

    def test_runs_without_a_backend_takeover(self):
        for module in MODULES:
            for backend in (None, _Backend(handled=False)):
                with self.subTest(module=module.__name__, backend=backend):
                    buffer = _CommBufferSpy()
                    opt = _dispatch_optimizer(module, backend, buffer)

                    opt._sharding_sync_parameters()

                    self.assertEqual(buffer.calls, ["sync_params"])
                    if backend is not None:
                        self.assertEqual(backend.names(), ["sync_parameters"])


class TestRebuildDispatch(unittest.TestCase):
    """Only DygraphShardingOptimizerV2 has a rebuild()."""

    def setUp(self):
        self.addCleanup(register_grad_comm_backend, None)

    def _rebuild(self, factory):
        if factory is not None:
            register_grad_comm_backend(factory)
        hcg = _hcg()
        with _sharding_env(dygraph_sharding_optimizer, hcg):
            opt = dygraph_sharding_optimizer.DygraphShardingOptimizerV2(
                _inner_optimizer(_frozen_param()), hcg
            )
            opt.rebuild()
        return opt

    def test_the_backend_is_notified(self):
        backend = _Backend()
        opt = self._rebuild(_Factory(backend))

        self.assertEqual(backend.names(), ["setup", "rebuild"])
        self.assertIs(backend.calls[1][1], opt)

    def test_rebuild_without_a_backend(self):
        opt = self._rebuild(None)

        self.assertIsNone(opt._grad_comm_backend)


if __name__ == "__main__":
    unittest.main()
