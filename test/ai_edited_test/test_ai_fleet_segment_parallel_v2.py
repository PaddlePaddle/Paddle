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

# [AUTO-GENERATED] Tests for paddle/distributed/fleet/meta_parallel/segment_parallel.py
# Target: SegmentParallel (_prepare_for_model)
# Coverage target: ~75.0% -> improved

"""
测试 paddle/distributed/fleet/meta_parallel/segment_parallel.py 中的 SegmentParallel 类。

Tests for SegmentParallel class in paddle/distributed/fleet/meta_parallel/segment_parallel.py.
Covers _prepare_for_model with various broadcast scenarios (sep, sharding, dp).
All distributed operations are mocked.
"""

import unittest
from unittest.mock import MagicMock, patch


class TestSegmentParallelPrepareForModel(unittest.TestCase):
    """测试 SegmentParallel._prepare_for_model / Test _prepare_for_model method."""

    def setUp(self):
        """设置 mock 环境 / Set up mock environment."""
        self.mock_layers = MagicMock()
        self.mock_layers.full_name.return_value = "test_layer"

    def tearDown(self):
        patch.stopall()

    def test_prepare_for_model_sep_only(self):
        """测试仅序列并行广播 / Test prepare with only sequence parallel."""
        from paddle.distributed.fleet.meta_parallel.segment_parallel import (
            SegmentParallel,
        )

        mock_hcg = MagicMock()
        mock_hcg.get_sharding_parallel_world_size.return_value = 1
        mock_hcg.get_data_parallel_world_size.return_value = 1

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.segment_parallel.broadcast_sep_parameters"
            ) as mock_sep,
            patch(
                "paddle.distributed.fleet.meta_parallel.segment_parallel.broadcast_sharding_parameters"
            ) as mock_sharding,
            patch(
                "paddle.distributed.fleet.meta_parallel.segment_parallel.broadcast_dp_parameters"
            ) as mock_dp,
        ):
            sp = SegmentParallel(self.mock_layers, mock_hcg, strategy=None)
            mock_sep.assert_called_once_with(self.mock_layers, mock_hcg)
            mock_sharding.assert_not_called()
            mock_dp.assert_not_called()

    def test_prepare_for_model_with_sharding(self):
        """测试包含分片并行广播 / Test prepare with sharding parallel."""
        from paddle.distributed.fleet.meta_parallel.segment_parallel import (
            SegmentParallel,
        )

        mock_hcg = MagicMock()
        mock_hcg.get_sharding_parallel_world_size.return_value = 2
        mock_hcg.get_data_parallel_world_size.return_value = 1

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.segment_parallel.broadcast_sep_parameters"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.segment_parallel.broadcast_sharding_parameters"
            ) as mock_sharding,
            patch(
                "paddle.distributed.fleet.meta_parallel.segment_parallel.broadcast_dp_parameters"
            ),
        ):
            sp = SegmentParallel(self.mock_layers, mock_hcg, strategy=None)
            mock_sharding.assert_called_once_with(self.mock_layers, mock_hcg)

    def test_prepare_for_model_with_dp(self):
        """测试包含数据并行广播 / Test prepare with data parallel."""
        from paddle.distributed.fleet.meta_parallel.segment_parallel import (
            SegmentParallel,
        )

        mock_hcg = MagicMock()
        mock_hcg.get_sharding_parallel_world_size.return_value = 1
        mock_hcg.get_data_parallel_world_size.return_value = 2

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.segment_parallel.broadcast_sep_parameters"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.segment_parallel.broadcast_sharding_parameters"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.segment_parallel.broadcast_dp_parameters"
            ) as mock_dp,
        ):
            sp = SegmentParallel(self.mock_layers, mock_hcg, strategy=None)
            mock_dp.assert_called_once_with(self.mock_layers, mock_hcg)

    def test_prepare_for_model_all_enabled(self):
        """测试所有广播都启用 / Test prepare with all broadcasts enabled."""
        from paddle.distributed.fleet.meta_parallel.segment_parallel import (
            SegmentParallel,
        )

        mock_hcg = MagicMock()
        mock_hcg.get_sharding_parallel_world_size.return_value = 2
        mock_hcg.get_data_parallel_world_size.return_value = 2

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.segment_parallel.broadcast_sep_parameters"
            ) as mock_sep,
            patch(
                "paddle.distributed.fleet.meta_parallel.segment_parallel.broadcast_sharding_parameters"
            ) as mock_sharding,
            patch(
                "paddle.distributed.fleet.meta_parallel.segment_parallel.broadcast_dp_parameters"
            ) as mock_dp,
        ):
            sp = SegmentParallel(self.mock_layers, mock_hcg, strategy=None)
            mock_sep.assert_called_once()
            mock_sharding.assert_called_once()
            mock_dp.assert_called_once()


class TestSegmentParallelForward(unittest.TestCase):
    """测试 SegmentParallel.forward / Test forward method."""

    def setUp(self):
        self.mock_layers = MagicMock()
        self.mock_layers.full_name.return_value = "test_layer"

    def tearDown(self):
        patch.stopall()

    def test_forward(self):
        """测试完整前向传播 / Test full forward."""
        from paddle.distributed.fleet.meta_parallel.segment_parallel import (
            SegmentParallel,
        )

        mock_hcg = MagicMock()
        mock_hcg.get_sharding_parallel_world_size.return_value = 1
        mock_hcg.get_data_parallel_world_size.return_value = 1

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.segment_parallel.broadcast_sep_parameters"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.segment_parallel.broadcast_sharding_parameters"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.segment_parallel.broadcast_dp_parameters"
            ),
        ):
            self.mock_layers.return_value = "layer_output"

            sp = SegmentParallel(self.mock_layers, mock_hcg, strategy=None)
            result = sp.forward("input")
            self.assertEqual(result, "layer_output")
            self.mock_layers.assert_called_once_with("input")


if __name__ == '__main__':
    unittest.main()
