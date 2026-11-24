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
sharding optimizers query once through ``install_grad_comm_backend``, at the end
of their ``__init__``. The backend it returns may replace single steps of the
gradient communication of the optimizer's ``FusedCommBuffer`` buckets: the
collective of a bucket, and the batched parameter sync. It never takes a whole
method over, so the rest of ``reduce_gradients`` and
``_sharding_sync_parameters`` runs either way.

Both directions are checked: with no factory registered nothing changes (the
bucket keeps its defaults, the original collective is still issued and the
buckets are reduced one after another), and with one the backend is installed,
the two reduction phases are separated and announced, the per-bucket parameter
sync is skipped exactly when the backend reports it sent the batch, and the
steps that are not offered to a backend (the overlapped all_gather, Muon's 2D
broadcast) run as they do without one. A backend that does take a bucket over
is given the reduce op the collective would have used (align mode included),
its task is waited on, and the one bucket kind it may not take over is
rejected.

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
    install_grad_comm_backend,
    register_grad_comm_backend,
)

MODULES = (dygraph_sharding_optimizer, muon_sharding_optimizer)


def _optimizer_cls(module):
    if module is muon_sharding_optimizer:
        return module.MuonShardingOptimizer
    return module.DygraphShardingOptimizerV2


class _Task:
    """What a backend returns from comm_grads, like a collective task."""

    def __init__(self):
        self.waited = 0

    def wait(self):
        self.waited += 1


class _Backend:
    """A fake gradient communication backend that records its calls.

    ``handled`` is what ``sync_params_fused`` returns, i.e. whether the backend
    claims to have sent the batch of buckets it was given; ``setup_result`` is
    whether it takes the optimizer over at all. ``log`` is shared with the
    bucket spies of a test, so that the order of the calls across both can be
    asserted.
    """

    def __init__(self, handled=True, setup_result=True, log=None):
        self.handled = handled
        self.setup_result = setup_result
        self.task = _Task()
        self.calls = []
        self.log = [] if log is None else log
        self.rebuild_buffers = None

    def setup(self, optimizer):
        self.calls.append(("setup", optimizer))
        return self.setup_result

    def comm_grads(self, bucket, reduce_op):
        self.calls.append(("comm_grads", bucket, reduce_op))
        return self.task

    def on_all_grads_posted(self, optimizer):
        self.calls.append(("on_all_grads_posted", optimizer))
        self.log.append(("on_all_grads_posted", None))

    def on_grads_reduced(self, optimizer):
        self.calls.append(("on_grads_reduced", optimizer))
        self.log.append(("on_grads_reduced", None))

    def sync_params_fused(self, buckets):
        self.calls.append(("sync_params_fused", buckets))
        self.log.append(("sync_params_fused", None))
        return self.handled

    def rebuild(self, optimizer):
        self.calls.append(("rebuild", optimizer))
        # rebuild() replaces the bucket list, so this records which generation
        # of buckets the backend is given a chance to take over.
        self.rebuild_buffers = optimizer._comm_buffer_list

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
    """Stands in for a FusedCommBuffer in the optimizer level tests.

    ``log`` is shared with the other spies and the backend of a test, so that
    the order of the calls across all of them can be asserted; ``calls`` is
    this bucket's own order.
    """

    def __init__(self, name="bucket", log=None):
        self.name = name
        self.log = [] if log is None else log
        self.calls = []
        self.sync_params_kwargs = []
        self.grad_storage = None
        self._dst = 0

    def _record(self, what):
        self.calls.append(what)
        self.log.append((what, self.name))

    def need_reduce_scale_sync(self):
        return True

    def _comm_grads(self):
        self._record("_comm_grads")

    def scale_grads(self):
        self._record("scale_grads")

    def sync_params(self, sync=True, param2task=None):
        self.sync_params_kwargs.append((sync, param2task))
        self._record("sync_params")


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


def _dispatch_optimizer(module, backend, buffers, **extra):
    """A shell optimizer carrying only what the two methods read."""
    attrs = {
        "_grad_comm_backend": backend,
        "_comm_buffer_list": list(buffers),
        "sd_release_grads": False,
        "comm_overlap": False,
        "_enable_timer": False,
        "_hcg": _hcg(),
    }
    if module is muon_sharding_optimizer:
        attrs.update(
            _use_fuse_gradients=False,
            comm_buffer_2d=[],
            _params_2d_by_color={},
            _rank2params_2d_by_color={},
            _color_to_group_info={},
            _sync_only_buffer_ids=set(),
            use_group_call_opt=False,
        )
    else:
        attrs.update(
            _all_gather_overlap_forward=False,
            _layers=types.SimpleNamespace(sublayers=lambda: []),
            _forward_pre_hook_remove_helper=[],
        )
    attrs.update(extra)
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


class TestInstallGradCommBackend(unittest.TestCase):
    """install_grad_comm_backend: query the factory, honour both refusals."""

    def setUp(self):
        self.addCleanup(register_grad_comm_backend, None)
        # The helper only passes the optimizer through, so a stand-in is enough.
        self.optimizer = types.SimpleNamespace(name="optimizer")

    def test_nothing_is_installed_without_a_factory(self):
        self.assertIsNone(install_grad_comm_backend(self.optimizer))

    def test_the_backend_is_installed_when_setup_accepts(self):
        backend = _Backend()
        factory = _Factory(backend)
        register_grad_comm_backend(factory)

        self.assertIs(install_grad_comm_backend(self.optimizer), backend)
        self.assertEqual(factory.optimizers, [self.optimizer])
        self.assertEqual(backend.names(), ["setup"])
        self.assertIs(backend.calls[0][1], self.optimizer)

    def test_a_factory_returning_none_installs_nothing(self):
        factory = _Factory(None)
        register_grad_comm_backend(factory)

        self.assertIsNone(install_grad_comm_backend(self.optimizer))
        self.assertEqual(factory.optimizers, [self.optimizer])

    def test_a_backend_declining_setup_installs_nothing(self):
        backend = _Backend(setup_result=False)
        register_grad_comm_backend(_Factory(backend))

        self.assertIsNone(install_grad_comm_backend(self.optimizer))
        self.assertEqual(backend.names(), ["setup"])


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

    def _reduce_scatter_bucket(self, use_reduce_avg=False):
        """A bucket on the sharding code path.

        A REDUCE_SCATTER FusedCommBuffer builds its views from the real
        parameter storage, so flip the action of an ALL_REDUCE bucket and give
        it a storage to slice instead.
        """
        bucket = self._bucket(use_reduce_avg=use_reduce_avg)
        bucket._act = HOOK_ACTION.REDUCE_SCATTER
        bucket.grad_storage = paddle.zeros([8], dtype="float32")
        return bucket

    def test_bucket_defaults_to_no_backend(self):
        self.assertIsNone(self._bucket()._grad_comm_backend)

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
        bucket = self._reduce_scatter_bucket()
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

    def test_align_mode_hands_the_backend_a_sum_op(self):
        """In align mode the reduce_scatter path switches from AVG to SUM.

        The gradients are already scaled when it does, so the backend has to be
        given the op the collective would really have used, not the AVG the
        bucket was built with.
        """
        for align_mode, expected_op in (
            (True, paddle.distributed.ReduceOp.SUM),
            (False, paddle.distributed.ReduceOp.AVG),
        ):
            with self.subTest(align_mode=align_mode):
                backend = _Backend()
                bucket = self._reduce_scatter_bucket(use_reduce_avg=True)
                bucket._grad_comm_backend = backend
                with mock.patch.object(
                    paddle.distributed,
                    "in_auto_parallel_align_mode",
                    return_value=align_mode,
                ):
                    bucket._comm_grads()

                self.assertEqual(backend.names(), ["comm_grads"])
                self.assertEqual(backend.calls[0][2], expected_op)

    def test_align_mode_still_sums_without_a_backend(self):
        bucket = self._reduce_scatter_bucket(use_reduce_avg=True)
        with (
            mock.patch.object(
                paddle.distributed, "reduce_scatter"
            ) as reduce_scatter,
            mock.patch.object(
                paddle.distributed,
                "in_auto_parallel_align_mode",
                return_value=True,
            ),
        ):
            bucket._comm_grads()

        self.assertEqual(
            reduce_scatter.call_args.kwargs["op"],
            paddle.distributed.ReduceOp.SUM,
        )

    def test_align_mode_leaves_a_non_sharding_bucket_alone(self):
        """Only the reduce_scatter path scales the gradients in advance, so an
        all_reduce bucket keeps its AVG op in align mode."""
        backend = _Backend()
        bucket = self._bucket(use_reduce_avg=True)
        bucket._grad_comm_backend = backend
        with mock.patch.object(
            paddle.distributed,
            "in_auto_parallel_align_mode",
            return_value=True,
        ):
            bucket._comm_grads()

        self.assertEqual(backend.calls[0][2], paddle.distributed.ReduceOp.AVG)

    def test_a_free_grads_in_comm_bucket_is_not_handed_over(self):
        """Such a bucket needs its reduced result in a fresh shard buffer that
        replaces the grad storage, which the backend interface cannot say, so
        taking it over is rejected instead of silently dropping that step."""
        backend = _Backend()
        bucket = self._reduce_scatter_bucket()
        # __init__ asserts free_grads_in_comm against the sharding views this
        # stand-in bucket does not build, so set the flag afterwards.
        bucket._free_grads_in_comm = True
        bucket._grad_comm_backend = backend

        with self.assertRaisesRegex(ValueError, "free_grads_in_comm"):
            bucket._comm_grads()
        self.assertEqual(backend.names(), [])

    def test_comm_grads_hands_the_bucket_over_on_every_call(self):
        """reduce_gradients calls _comm_grads for every bucket, including the
        ones a backend already took in a backward hook of its own, so a backend
        is handed the same bucket more than once within a step and has to be
        idempotent."""
        backend = _Backend()
        bucket = self._bucket()
        bucket._grad_comm_backend = backend

        bucket._comm_grads()
        bucket._comm_grads()

        self.assertEqual(backend.names(), ["comm_grads", "comm_grads"])
        self.assertIs(bucket._task, backend.task)

    def test_scale_grads_waits_for_the_backend_task(self):
        backend = _Backend()
        bucket = self._bucket()
        bucket.grad_storage = paddle.full([8], 4.0, dtype="float32")
        bucket._grad_comm_backend = backend

        bucket._comm_grads()
        self.assertEqual(backend.task.waited, 0)
        bucket.scale_grads()

        # The task the backend returned is waited on exactly like a collective
        # task, and the post-communication scaling still applies.
        self.assertEqual(backend.task.waited, 1)
        self.assertEqual(bucket.grad_storage.tolist(), [2.0] * 8)


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


class TestReduceGradients(unittest.TestCase):
    """reduce_gradients: with a backend the loop is split in two phases and
    each is announced; without one it is exactly what it was."""

    def setUp(self):
        # Checking the bucket padding needs the reduce-scatter views of real
        # buckets; it is off by default.
        patcher = mock.patch.object(
            dygraph_sharding_optimizer, "g_sharding_v2_check_zero_padding", 0
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_every_bucket_is_posted_before_the_first_wait(self):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                log = []
                backend = _Backend(log=log)
                buffers = [_CommBufferSpy(name, log) for name in ("a", "b")]
                opt = _dispatch_optimizer(module, backend, buffers)

                opt.reduce_gradients([], _hcg())

                # on_all_grads_posted after the last _comm_grads and before
                # the first scale_grads (the wait), on_grads_reduced last.
                self.assertEqual(
                    log,
                    [
                        ("_comm_grads", "a"),
                        ("_comm_grads", "b"),
                        ("on_all_grads_posted", None),
                        ("scale_grads", "a"),
                        ("scale_grads", "b"),
                        ("on_grads_reduced", None),
                    ],
                )
                # Both callbacks are given the optimizer, and the backend is
                # told nothing else: no method of it owns this call.
                self.assertEqual(
                    backend.names(),
                    ["on_all_grads_posted", "on_grads_reduced"],
                )
                for call in backend.calls:
                    self.assertIs(call[1], opt)

    def test_the_loop_is_unchanged_without_a_backend(self):
        # DygraphShardingOptimizerV2 posts a bucket and waits on it before it
        # moves to the next one; MuonShardingOptimizer has always posted every
        # bucket before it waits on any.
        expected = {
            dygraph_sharding_optimizer: [
                ("_comm_grads", "a"),
                ("scale_grads", "a"),
                ("_comm_grads", "b"),
                ("scale_grads", "b"),
            ],
            muon_sharding_optimizer: [
                ("_comm_grads", "a"),
                ("_comm_grads", "b"),
                ("scale_grads", "a"),
                ("scale_grads", "b"),
            ],
        }
        for module in MODULES:
            with self.subTest(module=module.__name__):
                log = []
                buffers = [_CommBufferSpy(name, log) for name in ("a", "b")]
                opt = _dispatch_optimizer(module, None, buffers)

                opt.reduce_gradients([], _hcg())

                self.assertEqual(log, expected[module])

    def test_muon_still_reduces_its_2d_buckets_with_a_backend(self):
        log = []
        backend = _Backend(log=log)
        bucket_1d = _CommBufferSpy("1d", log)
        bucket_2d = _CommBufferSpy("2d", log)
        opt = _dispatch_optimizer(
            muon_sharding_optimizer,
            backend,
            [bucket_1d],
            _use_fuse_gradients=True,
            comm_buffer_2d=[bucket_2d],
        )

        opt.reduce_gradients([], _hcg())

        # The 2D reduce is not offered to a backend, and it still runs: posted
        # before on_all_grads_posted, waited on before on_grads_reduced.
        self.assertEqual(
            log,
            [
                ("_comm_grads", "2d"),
                ("_comm_grads", "1d"),
                ("on_all_grads_posted", None),
                ("scale_grads", "2d"),
                ("scale_grads", "1d"),
                ("on_grads_reduced", None),
            ],
        )


class TestSyncParameters(unittest.TestCase):
    """_sharding_sync_parameters: a backend replaces the per-bucket
    sync_params of the batch it is offered, and nothing else."""

    def test_the_per_bucket_sync_is_skipped_when_the_backend_sent_the_batch(
        self,
    ):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                backend = _Backend(handled=True)
                buffers = [_CommBufferSpy("a"), _CommBufferSpy("b")]
                opt = _dispatch_optimizer(module, backend, buffers)

                opt._sharding_sync_parameters()

                self.assertEqual([b.calls for b in buffers], [[], []])
                # One call for the whole batch, not one per bucket.
                self.assertEqual(backend.names(), ["sync_params_fused"])
                self.assertIs(backend.calls[0][1], opt._comm_buffer_list)

    def test_the_buckets_are_synced_one_by_one_otherwise(self):
        for module in MODULES:
            for backend in (None, _Backend(handled=False)):
                with self.subTest(module=module.__name__, backend=backend):
                    buffers = [_CommBufferSpy("a"), _CommBufferSpy("b")]
                    opt = _dispatch_optimizer(module, backend, buffers)

                    opt._sharding_sync_parameters()

                    self.assertEqual(
                        [b.calls for b in buffers],
                        [["sync_params"], ["sync_params"]],
                    )
                    if backend is not None:
                        self.assertEqual(backend.names(), ["sync_params_fused"])

    def test_the_overlapped_all_gather_is_not_offered_to_a_backend(self):
        backend = _Backend(handled=True)
        buffer = _CommBufferSpy("a")
        opt = _dispatch_optimizer(
            dygraph_sharding_optimizer,
            backend,
            [buffer],
            _all_gather_overlap_forward=True,
        )

        opt._sharding_sync_parameters()

        # Overlapping the all_gathers with the forward pass and fusing them are
        # two optimizations of the same step, so this branch is not offered;
        # the buckets are still synced, asynchronously, as they were.
        self.assertEqual(backend.names(), [])
        self.assertEqual(buffer.calls, ["sync_params"])
        self.assertEqual(buffer.sync_params_kwargs, [(False, {})])

    def test_muon_still_broadcasts_its_2d_params_with_a_backend(self):
        backend = _Backend(handled=True)
        buffer = _CommBufferSpy("1d")
        group = _group(nranks=2)
        rank2params = {0: ["param"]}
        opt = _dispatch_optimizer(
            muon_sharding_optimizer,
            backend,
            [buffer],
            _rank2params_2d_by_color={"color": rank2params},
            _color_to_group_info={"color": {"group": group, "world_size": 2}},
        )
        task = _Task()
        broadcasts = []

        def broadcast_2d(params, comm_group):
            broadcasts.append((params, comm_group))
            return [task]

        opt._broadcast_2d_params = broadcast_2d

        opt._sharding_sync_parameters()

        # The 2D broadcast is not offered to a backend either: it is issued and
        # waited on with one installed, and only the 1D batch goes to it.
        self.assertEqual(broadcasts, [(rank2params, group)])
        self.assertEqual(task.waited, 1)
        self.assertEqual(backend.names(), ["sync_params_fused"])
        self.assertEqual(buffer.calls, [])


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
            self.old_buffers = opt._comm_buffer_list
            opt.rebuild()
        return opt

    def test_the_backend_is_notified(self):
        backend = _Backend()
        opt = self._rebuild(_Factory(backend))

        self.assertEqual(backend.names(), ["setup", "rebuild"])
        self.assertIs(backend.calls[1][1], opt)
        # Notified at the end of rebuild(): the buckets it is asked to take
        # over are the new ones the optimizer will dispatch to, not the
        # replaced ones (whose _grad_comm_backend nothing would ever set).
        self.assertIs(backend.rebuild_buffers, opt._comm_buffer_list)
        self.assertIsNot(backend.rebuild_buffers, self.old_buffers)

    def test_rebuild_without_a_backend(self):
        opt = self._rebuild(None)

        self.assertIsNone(opt._grad_comm_backend)


if __name__ == "__main__":
    unittest.main()
