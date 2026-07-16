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

# [AI-EDITED] Unit tests for the sharding-stage1 overlap "no-hook shared color"
# logic newly added to MuonShardingOptimizer.
# cover: _buffer_color / _compute_shared_colors / _select_sync_only_buffers,
# and __init__  sync-only buffer + overlap buffer
# Covered module: paddle/distributed/fleet/meta_optimizers/muon_sharding_optimizer.py

import unittest

from paddle.distributed.fleet.meta_optimizers.muon_sharding_optimizer import (
    MuonShardingOptimizer,
)


class _FakeParam:
    """A minimal stand-in for a Paddle parameter carrying a ``color`` attr."""

    def __init__(self, color, name="p"):
        self.color = color
        self.name = name


class _FakeBuffer:
    """A minimal stand-in for a FusedCommBuffer exposing ``.params``."""

    def __init__(self, params):
        self.params = params


class TestBufferColor(unittest.TestCase):
    """Test the _buffer_color static method."""

    def test_dict_color_moe_expert(self):
        """A dict color returns its 'color' entry."""
        buf = _FakeBuffer(
            [_FakeParam({"color": "moe_expert", "group": object()})]
        )
        self.assertEqual(MuonShardingOptimizer._buffer_color(buf), "moe_expert")

    def test_dict_color_no_hook(self):
        """A no-hook dict color is returned as-is."""
        buf = _FakeBuffer([_FakeParam({"color": "dense_weight_no_hook"})])
        self.assertEqual(
            MuonShardingOptimizer._buffer_color(buf), "dense_weight_no_hook"
        )

    def test_default_color_minus_one(self):
        """A default color of -1 is normalized to None."""
        buf = _FakeBuffer([_FakeParam(-1)])
        self.assertIsNone(MuonShardingOptimizer._buffer_color(buf))

    def test_none_color(self):
        """A None color returns None."""
        buf = _FakeBuffer([_FakeParam(None)])
        self.assertIsNone(MuonShardingOptimizer._buffer_color(buf))

    def test_plain_string_color(self):
        """A plain (non-dict) string color is returned unchanged."""
        buf = _FakeBuffer([_FakeParam("moe_expert")])
        self.assertEqual(MuonShardingOptimizer._buffer_color(buf), "moe_expert")

    def test_empty_buffer(self):
        """An empty buffer returns None."""
        self.assertIsNone(MuonShardingOptimizer._buffer_color(_FakeBuffer([])))

    def test_only_first_param_inspected(self):
        """Only the first param determines the buffer color."""
        buf = _FakeBuffer([_FakeParam(-1), _FakeParam({"color": "moe_expert"})])
        self.assertIsNone(MuonShardingOptimizer._buffer_color(buf))

    def test_dict_without_color_key(self):
        """A dict color missing the 'color' key returns None."""
        buf = _FakeBuffer([_FakeParam({"group": object()})])
        self.assertIsNone(MuonShardingOptimizer._buffer_color(buf))


class TestComputeSharedColors(unittest.TestCase):
    """Test the _compute_shared_colors classmethod."""

    def test_empty_when_no_no_hook_colors(self):
        """Returns empty set when no no-hook color present (MTP sharing off)."""
        params = [
            _FakeParam(-1),
            _FakeParam({"color": "moe_expert"}),
            _FakeParam(None),
        ]
        self.assertEqual(
            MuonShardingOptimizer._compute_shared_colors(params), set()
        )

    def test_dense_only(self):
        """Only the dense no-hook color is present."""
        params = [_FakeParam({"color": "dense_weight_no_hook"}), _FakeParam(-1)]
        self.assertEqual(
            MuonShardingOptimizer._compute_shared_colors(params),
            {"dense_weight_no_hook"},
        )

    def test_moe_only(self):
        """Only the moe no-hook color is present."""
        params = [
            _FakeParam({"color": "moe_weight_no_hook", "group": object()})
        ]
        self.assertEqual(
            MuonShardingOptimizer._compute_shared_colors(params),
            {"moe_weight_no_hook"},
        )

    def test_both_no_hook_colors(self):
        """Both no-hook colors present."""
        params = [
            _FakeParam({"color": "dense_weight_no_hook"}),
            _FakeParam({"color": "moe_weight_no_hook", "group": object()}),
            _FakeParam({"color": "moe_expert"}),  # ignored
            _FakeParam(-1),
        ]
        self.assertEqual(
            MuonShardingOptimizer._compute_shared_colors(params),
            {"dense_weight_no_hook", "moe_weight_no_hook"},
        )

    def test_dict_without_color_key_ignored(self):
        """A dict without a 'color' key does not leak None into the result."""
        params = [_FakeParam({"group": object()})]
        self.assertEqual(
            MuonShardingOptimizer._compute_shared_colors(params), set()
        )


class TestSelectSyncOnlyBuffers(unittest.TestCase):
    """Test the _select_sync_only_buffers classmethod."""

    def _buf(self, color):
        if color is None:
            return _FakeBuffer([_FakeParam(-1)])
        return _FakeBuffer([_FakeParam({"color": color})])

    def test_selects_only_shared_color_buffers(self):
        """Selects only buffers whose color is in shared_colors."""
        shared = {"dense_weight_no_hook", "moe_weight_no_hook"}
        b_dense = self._buf("dense_weight_no_hook")
        b_moe_nohook = self._buf("moe_weight_no_hook")
        b_moe = self._buf("moe_expert")
        b_none = self._buf(None)
        buffers = [b_dense, b_moe_nohook, b_moe, b_none]
        selected = MuonShardingOptimizer._select_sync_only_buffers(
            buffers, shared
        )
        self.assertEqual(selected, [b_dense, b_moe_nohook])

    def test_empty_shared_colors_selects_nothing(self):
        """No buffers are selected when shared_colors is empty."""
        buffers = [self._buf("moe_expert"), self._buf(None)]
        self.assertEqual(
            MuonShardingOptimizer._select_sync_only_buffers(buffers, set()),
            [],
        )

    def test_none_color_never_selected(self):
        """A None-color buffer is never selected."""
        shared = {"dense_weight_no_hook"}
        buffers = [self._buf(None)]
        self.assertEqual(
            MuonShardingOptimizer._select_sync_only_buffers(buffers, shared),
            [],
        )


class TestSyncOnlyVsOverlapPartition(unittest.TestCase):
    """Integration of the real building blocks that __init__ composes: recolored
    params -> shared_colors -> sync-only buffers, and the id-based exclusion
    that leaves only overlap buffers in _comm_buffers."""

    def test_partition_matches_init_logic(self):
        # backbone plain buffers (go through overlap)
        b_plain_dense = _FakeBuffer([_FakeParam(-1)])
        b_plain_moe = _FakeBuffer([_FakeParam({"color": "moe_expert"})])
        # MTP shared-layer re-colored buffers (go through synchronous reduce)
        b_shared_dense = _FakeBuffer(
            [_FakeParam({"color": "dense_weight_no_hook"})]
        )
        b_shared_moe = _FakeBuffer(
            [_FakeParam({"color": "moe_weight_no_hook", "group": object()})]
        )
        all_buffers = [b_plain_dense, b_plain_moe, b_shared_dense, b_shared_moe]

        # shared_colors derived from all params' colors
        all_params = [p for b in all_buffers for p in b.params]
        shared_colors = MuonShardingOptimizer._compute_shared_colors(all_params)
        self.assertEqual(
            shared_colors, {"dense_weight_no_hook", "moe_weight_no_hook"}
        )

        sync_only = MuonShardingOptimizer._select_sync_only_buffers(
            all_buffers, shared_colors
        )
        self.assertEqual(sync_only, [b_shared_dense, b_shared_moe])

        # __init__ uses an id set to exclude sync-only from the overlap buffers
        sync_only_ids = {id(b) for b in sync_only}
        overlap_buffers = [b for b in all_buffers if id(b) not in sync_only_ids]
        self.assertEqual(overlap_buffers, [b_plain_dense, b_plain_moe])

    def test_no_sharing_all_overlap(self):
        """With sharing off there are no no-hook colors, so nothing is sync-only."""
        buffers = [
            _FakeBuffer([_FakeParam(-1)]),
            _FakeBuffer([_FakeParam({"color": "moe_expert"})]),
        ]
        params = [p for b in buffers for p in b.params]
        shared_colors = MuonShardingOptimizer._compute_shared_colors(params)
        self.assertEqual(shared_colors, set())
        self.assertEqual(
            MuonShardingOptimizer._select_sync_only_buffers(
                buffers, shared_colors
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
