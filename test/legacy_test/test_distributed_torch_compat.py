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

import paddle.distributed as dist


class TestDistributedTorchCompat(unittest.TestCase):
    def test_group_namespace_exists(self):
        self.assertTrue(hasattr(dist, 'group'))
        self.assertTrue(hasattr(dist.group, 'WORLD'))

    def test_group_world_pre_init_is_none(self):
        # Pre-init, dist.group.WORLD resolves to None, which every collective
        # treats as the default group. (torch returns a sentinel object;
        # passing the result to a collective is what matters and behaves
        # identically.)
        self.assertIsNone(dist.group.WORLD)

    def test_process_group_re_export(self):
        from paddle.base.core import ProcessGroup as core_pg

        self.assertTrue(hasattr(dist, 'ProcessGroup'))
        self.assertIs(dist.ProcessGroup, core_pg)

    def test_init_process_group_exists_and_returns_none(self):
        self.assertTrue(hasattr(dist, 'init_process_group'))
        # In a single-process REPL, init_parallel_env early-returns. The
        # wrapper still returns None, matching torch's contract.
        result = dist.init_process_group(backend='gloo')
        self.assertIsNone(result)

    def test_init_process_group_sets_backend_env(self):
        prev = os.environ.get('PADDLE_DISTRI_BACKEND')
        try:
            dist.init_process_group(backend='nccl')
            self.assertEqual(os.environ.get('PADDLE_DISTRI_BACKEND'), 'nccl')
        finally:
            if prev is None:
                os.environ.pop('PADDLE_DISTRI_BACKEND', None)
            else:
                os.environ['PADDLE_DISTRI_BACKEND'] = prev

    def test_init_process_group_warns_on_world_size_mismatch(self):
        prev = os.environ.pop('PADDLE_TRAINERS_NUM', None)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                dist.init_process_group(backend='gloo', world_size=8)
            messages = [str(w.message) for w in caught]
            self.assertTrue(
                any('world_size=8' in m for m in messages),
                f"expected world_size warning, got: {messages}",
            )
        finally:
            if prev is not None:
                os.environ['PADDLE_TRAINERS_NUM'] = prev

    def test_init_process_group_warns_on_rank_mismatch(self):
        prev = os.environ.pop('PADDLE_TRAINER_ID', None)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                dist.init_process_group(backend='gloo', rank=3)
            messages = [str(w.message) for w in caught]
            self.assertTrue(
                any('rank=3' in m for m in messages),
                f"expected rank warning, got: {messages}",
            )
        finally:
            if prev is not None:
                os.environ['PADDLE_TRAINER_ID'] = prev

    def test_init_process_group_no_warning_on_default_args(self):
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
