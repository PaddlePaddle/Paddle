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

# [AUTO-GENERATED] Tests for paddle/distributed/fleet/meta_parallel/tensor_parallel.py
# Target: TensorParallel (_prepare_for_model, _pre_forward)
# Coverage target: ~74.2% -> improved

"""
测试 paddle/distributed/fleet/meta_parallel/tensor_parallel.py 中的 TensorParallel 类。

Tests for TensorParallel class in paddle/distributed/fleet/meta_parallel/tensor_parallel.py.
Covers _prepare_for_model (various broadcast scenarios) and _pre_forward (broadcast_input_data).
All distributed operations are mocked.
"""

import unittest
from unittest.mock import MagicMock, patch


class TestTensorParallelPrepareForModel(unittest.TestCase):
    """测试 TensorParallel._prepare_for_model / Test _prepare_for_model method."""

    def setUp(self):
        """设置 mock 环境 / Set up mock environment."""
        self.mock_layers = MagicMock()
        self.mock_layers.full_name.return_value = "test_layer"

    def tearDown(self):
        patch.stopall()

    def test_prepare_for_model_mp_only(self):
        """测试仅模型并行广播 / Test prepare with only model parallel."""
        from paddle.distributed.fleet.meta_parallel.tensor_parallel import (
            TensorParallel,
        )

        mock_hcg = MagicMock()
        mock_hcg.get_sep_parallel_world_size.return_value = 1
        mock_hcg.get_sharding_parallel_world_size.return_value = 1
        mock_hcg.get_data_parallel_world_size.return_value = 1
        mock_hcg.get_moe_sharding_parallel_world_size.return_value = 1

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_mp_parameters"
            ) as mock_mp,
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_sep_parameters"
            ) as mock_sep,
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_sharding_parameters"
            ) as mock_sharding,
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_dp_parameters"
            ) as mock_dp,
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_moe_sharding_parameters"
            ) as mock_moe,
        ):
            tp = TensorParallel(self.mock_layers, mock_hcg, strategy=None)
            mock_mp.assert_called_once_with(self.mock_layers, mock_hcg)
            mock_sep.assert_not_called()
            mock_sharding.assert_not_called()
            mock_dp.assert_not_called()
            mock_moe.assert_not_called()

    def test_prepare_for_model_with_sep(self):
        """测试包含序列并行广播 / Test prepare with sequence parallel."""
        from paddle.distributed.fleet.meta_parallel.tensor_parallel import (
            TensorParallel,
        )

        mock_hcg = MagicMock()
        mock_hcg.get_sep_parallel_world_size.return_value = 2
        mock_hcg.get_sharding_parallel_world_size.return_value = 1
        mock_hcg.get_data_parallel_world_size.return_value = 1
        mock_hcg.get_moe_sharding_parallel_world_size.return_value = 1

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_mp_parameters"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_sep_parameters"
            ) as mock_sep,
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_sharding_parameters"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_dp_parameters"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_moe_sharding_parameters"
            ),
        ):
            TensorParallel(self.mock_layers, mock_hcg, strategy=None)
            mock_sep.assert_called_once_with(
                self.mock_layers, mock_hcg, fuse_params=False
            )

    def test_prepare_for_model_with_sharding(self):
        """测试包含分片并行广播 / Test prepare with sharding parallel."""
        from paddle.distributed.fleet.meta_parallel.tensor_parallel import (
            TensorParallel,
        )

        mock_hcg = MagicMock()
        mock_hcg.get_sep_parallel_world_size.return_value = 1
        mock_hcg.get_sharding_parallel_world_size.return_value = 2
        mock_hcg.get_data_parallel_world_size.return_value = 1
        mock_hcg.get_moe_sharding_parallel_world_size.return_value = 1

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_mp_parameters"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_sep_parameters"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_sharding_parameters"
            ) as mock_sharding,
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_dp_parameters"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_moe_sharding_parameters"
            ),
        ):
            TensorParallel(self.mock_layers, mock_hcg, strategy=None)
            mock_sharding.assert_called_once_with(
                self.mock_layers, mock_hcg, fuse_params=False
            )

    def test_prepare_for_model_with_dp(self):
        """测试包含数据并行广播 / Test prepare with data parallel."""
        from paddle.distributed.fleet.meta_parallel.tensor_parallel import (
            TensorParallel,
        )

        mock_hcg = MagicMock()
        mock_hcg.get_sep_parallel_world_size.return_value = 1
        mock_hcg.get_sharding_parallel_world_size.return_value = 1
        mock_hcg.get_data_parallel_world_size.return_value = 2
        mock_hcg.get_moe_sharding_parallel_world_size.return_value = 1

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_mp_parameters"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_sep_parameters"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_sharding_parameters"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_dp_parameters"
            ) as mock_dp,
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_moe_sharding_parameters"
            ),
        ):
            TensorParallel(self.mock_layers, mock_hcg, strategy=None)
            mock_dp.assert_called_once_with(
                self.mock_layers, mock_hcg, fuse_params=False
            )

    def test_prepare_for_model_with_moe_sharding(self):
        """测试包含MOE分片并行广播 / Test prepare with MoE sharding parallel."""
        from paddle.distributed.fleet.meta_parallel.tensor_parallel import (
            TensorParallel,
        )

        mock_hcg = MagicMock()
        mock_hcg.get_sep_parallel_world_size.return_value = 1
        mock_hcg.get_sharding_parallel_world_size.return_value = 1
        mock_hcg.get_data_parallel_world_size.return_value = 1
        mock_hcg.get_moe_sharding_parallel_world_size.return_value = 2

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_mp_parameters"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_sep_parameters"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_sharding_parameters"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_dp_parameters"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_moe_sharding_parameters"
            ) as mock_moe,
        ):
            TensorParallel(self.mock_layers, mock_hcg, strategy=None)
            mock_moe.assert_called_once_with(
                self.mock_layers, mock_hcg, fuse_params=False
            )

    def test_prepare_for_model_all_enabled(self):
        """测试所有广播都启用 / Test prepare with all broadcasts enabled."""
        from paddle.distributed.fleet.meta_parallel.tensor_parallel import (
            TensorParallel,
        )

        mock_hcg = MagicMock()
        mock_hcg.get_sep_parallel_world_size.return_value = 2
        mock_hcg.get_sharding_parallel_world_size.return_value = 2
        mock_hcg.get_data_parallel_world_size.return_value = 2
        mock_hcg.get_moe_sharding_parallel_world_size.return_value = 2

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_mp_parameters"
            ) as mock_mp,
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_sep_parameters"
            ) as mock_sep,
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_sharding_parameters"
            ) as mock_sharding,
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_dp_parameters"
            ) as mock_dp,
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_moe_sharding_parameters"
            ) as mock_moe,
        ):
            TensorParallel(self.mock_layers, mock_hcg, strategy=None)
            mock_mp.assert_called_once()
            mock_sep.assert_called_once()
            mock_sharding.assert_called_once()
            mock_dp.assert_called_once()
            mock_moe.assert_called_once()


class TestTensorParallelPreForward(unittest.TestCase):
    """测试 TensorParallel._pre_forward / Test _pre_forward method."""

    def setUp(self):
        """设置 mock 环境 / Set up mock environment."""
        self.mock_layers = MagicMock()
        self.mock_layers.full_name.return_value = "test_layer"

    def tearDown(self):
        patch.stopall()

    def test_pre_forward_no_strategy(self):
        """测试无策略时默认广播输入 / Test pre_forward without strategy."""
        from paddle.distributed.fleet.meta_parallel.tensor_parallel import (
            TensorParallel,
        )

        mock_hcg = MagicMock()
        mock_hcg.get_sep_parallel_world_size.return_value = 1
        mock_hcg.get_sharding_parallel_world_size.return_value = 1
        mock_hcg.get_data_parallel_world_size.return_value = 1
        mock_hcg.get_moe_sharding_parallel_world_size.return_value = 1

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_mp_parameters"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_input_data"
            ) as mock_broadcast,
        ):
            mock_broadcast.return_value = ("broadcasted_input",)

            tp = TensorParallel(self.mock_layers, mock_hcg, strategy=None)
            result = tp._pre_forward("input_data")
            mock_broadcast.assert_called_once_with(mock_hcg, "input_data")
            self.assertEqual(result, ("broadcasted_input",))

    def test_pre_forward_with_strategy_need_broadcast(self):
        """测试策略指定需要广播 / Test pre_forward with strategy needing broadcast."""
        from paddle.distributed.fleet.meta_parallel.tensor_parallel import (
            TensorParallel,
        )

        mock_hcg = MagicMock()
        mock_hcg.get_sep_parallel_world_size.return_value = 1
        mock_hcg.get_sharding_parallel_world_size.return_value = 1
        mock_hcg.get_data_parallel_world_size.return_value = 1
        mock_hcg.get_moe_sharding_parallel_world_size.return_value = 1

        mock_strategy = MagicMock()
        mock_strategy.hybrid_configs = {"mp_configs": MagicMock()}
        mock_strategy.hybrid_configs["mp_configs"].need_broadcast_data = True

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_mp_parameters"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_input_data"
            ) as mock_broadcast,
        ):
            mock_broadcast.return_value = ("broadcasted",)

            tp = TensorParallel(
                self.mock_layers, mock_hcg, strategy=mock_strategy
            )
            result = tp._pre_forward("data")
            mock_broadcast.assert_called_once_with(mock_hcg, "data")

    def test_pre_forward_with_strategy_no_broadcast(self):
        """测试策略指定不需要广播 / Test pre_forward with strategy not needing broadcast."""
        from paddle.distributed.fleet.meta_parallel.tensor_parallel import (
            TensorParallel,
        )

        mock_hcg = MagicMock()
        mock_hcg.get_sep_parallel_world_size.return_value = 1
        mock_hcg.get_sharding_parallel_world_size.return_value = 1
        mock_hcg.get_data_parallel_world_size.return_value = 1
        mock_hcg.get_moe_sharding_parallel_world_size.return_value = 1

        mock_strategy = MagicMock()
        mock_strategy.hybrid_configs = {"mp_configs": MagicMock()}
        mock_strategy.hybrid_configs["mp_configs"].need_broadcast_data = False

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_mp_parameters"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_input_data"
            ) as mock_broadcast,
        ):
            tp = TensorParallel(
                self.mock_layers, mock_hcg, strategy=mock_strategy
            )
            result = tp._pre_forward("data")
            mock_broadcast.assert_not_called()
            # _pre_forward returns None when not broadcasting
            self.assertIsNone(result)

    def test_pre_forward_with_kwargs(self):
        """测试带关键字参数的前向传播 / Test pre_forward with kwargs."""
        from paddle.distributed.fleet.meta_parallel.tensor_parallel import (
            TensorParallel,
        )

        mock_hcg = MagicMock()
        mock_hcg.get_sep_parallel_world_size.return_value = 1
        mock_hcg.get_sharding_parallel_world_size.return_value = 1
        mock_hcg.get_data_parallel_world_size.return_value = 1
        mock_hcg.get_moe_sharding_parallel_world_size.return_value = 1

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_mp_parameters"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_input_data"
            ) as mock_broadcast,
        ):
            mock_broadcast.return_value = ("broadcasted",)

            tp = TensorParallel(self.mock_layers, mock_hcg, strategy=None)
            result = tp._pre_forward("data", key="value")
            mock_broadcast.assert_called_once_with(
                mock_hcg, "data", key="value"
            )


class TestTensorParallelForward(unittest.TestCase):
    """测试 TensorParallel.forward / Test forward method."""

    def setUp(self):
        self.mock_layers = MagicMock()
        self.mock_layers.full_name.return_value = "test_layer"

    def tearDown(self):
        patch.stopall()

    def test_forward(self):
        """测试完整前向传播 / Test full forward."""
        from paddle.distributed.fleet.meta_parallel.tensor_parallel import (
            TensorParallel,
        )

        mock_hcg = MagicMock()
        mock_hcg.get_sep_parallel_world_size.return_value = 1
        mock_hcg.get_sharding_parallel_world_size.return_value = 1
        mock_hcg.get_data_parallel_world_size.return_value = 1
        mock_hcg.get_moe_sharding_parallel_world_size.return_value = 1

        with (
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_mp_parameters"
            ),
            patch(
                "paddle.distributed.fleet.meta_parallel.tensor_parallel.broadcast_input_data"
            ) as mock_broadcast,
        ):
            self.mock_layers.return_value = "layer_output"

            tp = TensorParallel(self.mock_layers, mock_hcg, strategy=None)
            result = tp.forward("input")
            self.assertEqual(result, "layer_output")
            # Note: MetaParallelBase.forward ignores _pre_forward return value,
            # so _layers is called with original "input", not "broadcasted"
            self.mock_layers.assert_called_once_with("input")
            # Verify broadcast was called (for coverage)
            mock_broadcast.assert_called()


if __name__ == '__main__':
    unittest.main()
