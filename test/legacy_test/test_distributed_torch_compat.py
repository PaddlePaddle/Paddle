#   Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""
Single-process tests for the torch-compat additions in ``paddle.distributed``:
``group.WORLD``, ``init_process_group``, ``ProcessGroup``. The full
multi-process behavior of ``init_parallel_env`` (and therefore of
``init_process_group``, which delegates to it) is covered by the existing
``test_collective_*`` suite; these tests cover the wrapper logic and the
``group.WORLD`` accessor that does not need a real distributed runtime.
"""

import os
import unittest
import warnings
from unittest import mock

import paddle.distributed as dist
from paddle.distributed.communication.group import Group


class TestDistributedTorchCompat(unittest.TestCase):
    def setUp(self):
        # The default-group slot is process-global state. ``WORLD = None``
        # clears it across all four registries the setter manages, so each
        # test sees the same starting state.
        self._saved_world = dist.group.WORLD
        dist.group.WORLD = None

    def tearDown(self):
        dist.group.WORLD = self._saved_world

    def test_group_namespace_exists(self):
        self.assertTrue(hasattr(dist, 'group'))
        self.assertTrue(hasattr(dist.group, 'WORLD'))

    def test_group_world_is_none_before_init(self):
        # ``group.WORLD`` resolves to ``None`` before any initialization,
        # matching ``torch.distributed.group.WORLD`` semantics.
        self.assertIsNone(dist.group.WORLD)

    def test_group_world_setter_assigns_default_group(self):
        # ``group.WORLD = pg`` mirrors ``torch.distributed.GroupMember.WORLD``
        # assignment. A subsequent read returns the assigned group, and the
        # group's public attributes (type, rank, id, nranks, name) match
        # what was assigned.
        assigned = Group(
            rank_in_group=0, id=0, ranks=[0], pg=None, name='assigned-world'
        )
        dist.group.WORLD = assigned

        world = dist.group.WORLD
        self.assertIsInstance(world, Group)
        self.assertIs(world, assigned)
        self.assertEqual(world.rank, 0)
        self.assertEqual(world.id, 0)
        self.assertEqual(world.nranks, 1)
        self.assertEqual(world.name, 'assigned-world')

    def test_group_world_setter_clears_with_none(self):
        dist.group.WORLD = Group(
            rank_in_group=0, id=0, ranks=[0], pg=None, name='to-clear'
        )
        self.assertIsNotNone(dist.group.WORLD)
        dist.group.WORLD = None
        self.assertIsNone(dist.group.WORLD)

    def test_group_world_setter_rejects_invalid_type(self):
        with self.assertRaises(TypeError):
            dist.group.WORLD = "not a group"
        with self.assertRaises(TypeError):
            dist.group.WORLD = 42

    def test_group_world_setter_rejects_mismatched_id(self):
        # The default group must be keyed by ``_GroupManager.global_group_id``
        # (== 0); a Group whose own ``id`` differs would break ``get_group``
        # lookups, so the setter must reject it.
        mismatched = Group(
            rank_in_group=0, id=5, ranks=[0], pg=None, name='wrong-id'
        )
        with self.assertRaises(ValueError):
            dist.group.WORLD = mismatched

    def test_group_world_setter_preserves_world_on_invalid_assignment(self):
        # Validation must happen before mutating any registry — a failed
        # assignment must leave the existing WORLD intact.
        original = Group(
            rank_in_group=0, id=0, ranks=[0], pg=None, name='original'
        )
        dist.group.WORLD = original
        self.assertIs(dist.group.WORLD, original)

        with self.assertRaises(TypeError):
            dist.group.WORLD = "not a group"
        self.assertIs(dist.group.WORLD, original)

        with self.assertRaises(ValueError):
            dist.group.WORLD = Group(
                rank_in_group=0, id=5, ranks=[0], pg=None, name='wrong-id'
            )
        self.assertIs(dist.group.WORLD, original)

    def test_destroy_process_group_clears_all_registries(self):
        # destroy_process_group() on the default group must clear every
        # registry (communication ``_GroupManager`` plus collective's
        # ``_group_map`` / ``_group_map_by_name`` / ``_group_map_backend``)
        # so a follow-up init_process_group can re-create it instead of
        # hitting init_parallel_env's early-return path.
        from paddle.distributed import collective as _coll
        from paddle.distributed.communication.group import (
            destroy_process_group,
        )

        dist.group.WORLD = Group(
            rank_in_group=0, id=0, ranks=[0], pg=None, name='_default_pg'
        )
        # Pre-condition: setter populated all four registries.
        self.assertIsNotNone(dist.group.WORLD)
        self.assertIn(_coll._global_env_gid, _coll._group_map)
        self.assertIn(_coll._default_group_name, _coll._group_map_by_name)

        destroy_process_group()

        self.assertIsNone(dist.group.WORLD)
        self.assertNotIn(_coll._global_env_gid, _coll._group_map)
        self.assertNotIn(_coll._default_group_name, _coll._group_map_by_name)

    def test_process_group_re_export(self):
        from paddle.base.core import ProcessGroup as core_pg

        self.assertTrue(hasattr(dist, 'ProcessGroup'))
        self.assertIs(dist.ProcessGroup, core_pg)

    # The init_process_group tests below exercise only the wrapper's
    # env-forwarding behavior. The wrapper delegates to ``init_parallel_env``,
    # whose full distributed behavior (real ProcessGroup creation, default
    # group registration via parallel.py:1188-1192) is covered by
    # ``test_collective_*``. Patching ``init_parallel_env`` here keeps the
    # tests deterministic without a multi-process launch.

    @mock.patch('paddle.distributed.parallel.init_parallel_env')
    @mock.patch.dict(os.environ, {}, clear=False)
    def test_init_process_group_returns_none(self, _):
        # ``torch.distributed.init_process_group`` returns ``None``; the
        # wrapper must too.
        self.assertIsNone(dist.init_process_group(backend='gloo'))

    @mock.patch('paddle.distributed.parallel.init_parallel_env')
    @mock.patch.dict(os.environ, {}, clear=False)
    def test_init_process_group_raises_on_double_init(self, _):
        # Once the default group is in place, a second call must raise
        # rather than silently no-op'ing via ``init_parallel_env``'s
        # early return. Mirrors PyTorch's
        # ``trying to initialize the default process group twice!``.
        dist.group.WORLD = Group(
            rank_in_group=0, id=0, ranks=[0], pg=None, name='_default_pg'
        )
        with self.assertRaises(RuntimeError) as ctx:
            dist.init_process_group(backend='gloo')
        self.assertIn('already been initialized', str(ctx.exception))

    @mock.patch('paddle.distributed.parallel.init_parallel_env')
    @mock.patch.dict(os.environ, {}, clear=False)
    def test_init_process_group_sets_backend_env(self, _):
        dist.init_process_group(backend='nccl')
        self.assertEqual(os.environ.get('PADDLE_DISTRI_BACKEND'), 'nccl')

    @mock.patch('paddle.distributed.parallel.init_parallel_env')
    @mock.patch.dict(
        os.environ,
        {'PADDLE_TRAINERS_NUM': '2'},  # env disagrees with arg below
        clear=False,
    )
    def test_init_process_group_warns_on_world_size_mismatch(self, _):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            dist.init_process_group(backend='gloo', world_size=8)
        messages = [str(w.message) for w in caught]
        self.assertTrue(
            any('world_size=8' in m for m in messages),
            f"expected world_size warning, got: {messages}",
        )

    @mock.patch('paddle.distributed.parallel.init_parallel_env')
    @mock.patch.dict(
        os.environ,
        {'PADDLE_TRAINER_ID': '0'},  # env disagrees with arg below
        clear=False,
    )
    def test_init_process_group_warns_on_rank_mismatch(self, _):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            dist.init_process_group(backend='gloo', rank=3)
        messages = [str(w.message) for w in caught]
        self.assertTrue(
            any('rank=3' in m for m in messages),
            f"expected rank warning, got: {messages}",
        )

    @mock.patch('paddle.distributed.parallel.init_parallel_env')
    @mock.patch.dict(os.environ, {}, clear=False)
    def test_init_process_group_writes_world_size_when_env_unset(self, _):
        os.environ.pop('PADDLE_TRAINERS_NUM', None)
        dist.init_process_group(backend='gloo', world_size=4)
        self.assertEqual(os.environ.get('PADDLE_TRAINERS_NUM'), '4')

    @mock.patch('paddle.distributed.parallel.init_parallel_env')
    @mock.patch.dict(os.environ, {}, clear=False)
    def test_init_process_group_writes_rank_when_env_unset(self, _):
        os.environ.pop('PADDLE_TRAINER_ID', None)
        dist.init_process_group(backend='gloo', rank=2)
        self.assertEqual(os.environ.get('PADDLE_TRAINER_ID'), '2')

    @mock.patch('paddle.distributed.parallel.init_parallel_env')
    @mock.patch.dict(os.environ, {}, clear=False)
    def test_init_process_group_no_warning_on_default_args(self, _):
        # Defaults (-1) must not trigger the mismatch warning.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            dist.init_process_group(backend='gloo')
        unwanted = [
            str(w.message)
            for w in caught
            if 'world_size=' in str(w.message) or 'rank=' in str(w.message)
        ]
        self.assertEqual(unwanted, [])

    @mock.patch('paddle.distributed.parallel.init_parallel_env')
    @mock.patch.dict(
        os.environ,
        {
            'WORLD_SIZE': '2',
            'RANK': '1',
            'LOCAL_RANK': '1',
            'MASTER_ADDR': '127.0.0.1',
            'MASTER_PORT': '29500',
        },
        clear=True,
    )
    def test_init_process_group_does_not_auto_map_torchrun_env(self, _):
        # The wrapper accepts PyTorch-style arguments for source
        # compatibility, but it does not currently translate torchrun env
        # vars into the PADDLE_* variables consumed by ParallelEnv.
        dist.init_process_group(backend='gloo')

        self.assertEqual(os.environ.get('PADDLE_DISTRI_BACKEND'), 'gloo')
        self.assertIsNone(os.environ.get('PADDLE_TRAINERS_NUM'))
        self.assertIsNone(os.environ.get('PADDLE_TRAINER_ID'))
        self.assertIsNone(os.environ.get('PADDLE_CURRENT_ENDPOINT'))
        self.assertIsNone(os.environ.get('PADDLE_TRAINER_ENDPOINTS'))

    def test_in___all__(self):
        for name in ('group', 'init_process_group', 'ProcessGroup'):
            self.assertIn(name, dist.__all__)


if __name__ == '__main__':
    unittest.main()
