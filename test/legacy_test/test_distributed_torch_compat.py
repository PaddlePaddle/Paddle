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
Single-process smoke tests for the torch-compat additions in
``paddle.distributed``: ``group.WORLD``, ``init_process_group``,
``ProcessGroup``. These do not exercise actual collectives - the multi-process
behavior is covered by ``test_collective_*`` - they only verify that the
symbols are exposed and behave correctly without a distributed runtime.
"""

import os
import unittest
import warnings
from unittest import mock

import paddle.distributed as dist


class TestDistributedTorchCompat(unittest.TestCase):
    def test_group_namespace_exists(self):
        self.assertTrue(hasattr(dist, 'group'))
        self.assertTrue(hasattr(dist.group, 'WORLD'))

    def test_group_world_is_none_before_init(self):
        # Before init_parallel_env / init_process_group runs, the default
        # group does not exist yet and ``WORLD`` resolves to ``None`` -
        # matching ``torch.distributed.group.WORLD`` semantics. The single
        # process running this test does not enter distributed mode, so
        # ``_get_global_group`` raises and the metaclass property catches
        # it.
        self.assertIsNone(dist.group.WORLD)

    def test_group_world_reflects_default_group_after_init(self):
        # After initialization, ``WORLD`` must return the actual default
        # Group object (not a static ``None``). We mock the lookup so this
        # runs in a single process without a real distributed launch.
        from paddle.distributed.communication.group import Group

        fake_default = Group(
            rank_in_group=0, id=0, ranks=[0], pg=None, name='fake-world'
        )
        with mock.patch(
            'paddle.distributed.communication.group._get_global_group',
            return_value=fake_default,
        ):
            world = dist.group.WORLD
            self.assertIs(world, fake_default)
            self.assertIsInstance(world, Group)
            self.assertEqual(world.name, 'fake-world')

    def test_process_group_re_export(self):
        from paddle.base.core import ProcessGroup as core_pg

        self.assertTrue(hasattr(dist, 'ProcessGroup'))
        self.assertIs(dist.ProcessGroup, core_pg)

    # The tests below exercise only the wrapper's env-forwarding behavior;
    # init_parallel_env is mocked because actually entering it requires a
    # full distributed launch (PADDLE_TRAINER_ID, PADDLE_CURRENT_ENDPOINT,
    # FLAGS_selected_gpus, ...). The multi-process behavior of
    # init_parallel_env is covered by the existing test_collective_* suite.

    @mock.patch('paddle.distributed.parallel.init_parallel_env')
    @mock.patch.dict(os.environ, {}, clear=False)
    def test_init_process_group_returns_default_group(self, mock_init):
        # Mirrors torch.distributed.init_process_group, which sets up the
        # default group and exposes it via ``group.WORLD``. The wrapper
        # forwards the value produced by ``init_parallel_env``.
        from paddle.distributed.communication.group import Group

        self.assertTrue(hasattr(dist, 'init_process_group'))
        fake_default = Group(
            rank_in_group=0, id=0, ranks=[0], pg=None, name='fake-world'
        )
        mock_init.return_value = fake_default
        result = dist.init_process_group(backend='gloo')
        self.assertIs(result, fake_default)
        # And the same group is exposed through ``dist.group.WORLD``.
        with mock.patch(
            'paddle.distributed.communication.group._get_global_group',
            return_value=fake_default,
        ):
            self.assertIs(dist.group.WORLD, fake_default)

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
        # Patch out init_parallel_env so we test only the env-forwarding
        # behavior of the wrapper, not the full initialization machinery.
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

    def test_in___all__(self):
        for name in ('group', 'init_process_group', 'ProcessGroup'):
            self.assertIn(name, dist.__all__)


if __name__ == '__main__':
    unittest.main()
