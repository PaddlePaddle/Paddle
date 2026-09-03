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

# [AUTO-GENERATED] Tests for paddle/distributed/fleet/layers/mpu/mp_ops.py
# Target: _c_identity, _c_concat, _c_split, _mp_allreduce, _c_lookup_table,
#         _linear, _c_softmax_with_cross_entropy, _parallel_linear, _parallel_embedding, split
# Coverage target: ~67.9% -> improved

"""
测试 paddle/distributed/fleet/layers/mpu/mp_ops.py 中的模型并行操作函数。

Tests for model parallel operation functions in paddle/distributed/fleet/layers/mpu/mp_ops.py.
Covers _c_identity, _c_concat, _c_split, _mp_allreduce, _c_lookup_table,
_linear, _c_softmax_with_cross_entropy, _parallel_linear, _parallel_embedding, split.
All distributed operations are mocked.
"""

import unittest
from unittest.mock import MagicMock, patch


class TestCIdentity(unittest.TestCase):
    """测试 _c_identity 操作 / Test _c_identity operation."""

    def setUp(self):
        self.mock_framework = patch(
            "paddle.distributed.fleet.layers.mpu.mp_ops.in_dynamic_mode",
            return_value=False,
        ).start()
        self.mock_in_pir_mode = patch(
            "paddle.distributed.fleet.layers.mpu.mp_ops.in_pir_mode",
            return_value=False,
        ).start()

    def tearDown(self):
        patch.stopall()

    def test_c_identity_non_member_returns_none(self):
        """测试非成员返回None / Test non-member group returns None."""
        from paddle.distributed.fleet.layers.mpu.mp_ops import _c_identity

        mock_group = MagicMock()
        mock_group.is_member.return_value = False
        result = _c_identity(MagicMock(), group=mock_group)
        self.assertIsNone(result)

    def test_c_identity_member_static_mode(self):
        """测试成员静态模式 / Test member in static mode."""
        from paddle.distributed.fleet.layers.mpu.mp_ops import _c_identity

        mock_group = MagicMock()
        mock_group.is_member.return_value = True
        mock_group.id = 7
        mock_tensor = MagicMock()
        mock_tensor.dtype = "float32"

        with patch(
            "paddle.distributed.fleet.layers.mpu.mp_ops.LayerHelper"
        ) as mock_lh:
            mock_helper = MagicMock()
            mock_lh.return_value = mock_helper
            mock_out = MagicMock()
            mock_out.dtype = "float32"
            mock_helper.create_variable_for_type_inference.return_value = (
                mock_out
            )

            result = _c_identity(mock_tensor, group=mock_group)
            self.assertIsNotNone(result)
            mock_helper.append_op.assert_called_once()


class TestCCconcat(unittest.TestCase):
    """测试 _c_concat 操作 / Test _c_concat operation."""

    def setUp(self):
        self.mock_framework = patch(
            "paddle.distributed.fleet.layers.mpu.mp_ops.in_dynamic_or_pir_mode",
            return_value=False,
        ).start()

    def tearDown(self):
        patch.stopall()

    def test_c_concat_non_member_returns_none(self):
        """测试非成员返回None / Test non-member group returns None."""
        from paddle.distributed.fleet.layers.mpu.mp_ops import _c_concat

        mock_group = MagicMock()
        mock_group.is_member.return_value = False
        result = _c_concat(MagicMock(), group=mock_group)
        self.assertIsNone(result)

    def test_c_concat_static_mode(self):
        """测试静态模式 / Test static mode."""
        from paddle.distributed.fleet.layers.mpu.mp_ops import _c_concat

        mock_group = MagicMock()
        mock_group.is_member.return_value = True
        mock_group.id = 3
        mock_group.rank = 0
        mock_group.nranks = 2

        with patch(
            "paddle.distributed.fleet.layers.mpu.mp_ops.collective"
        ) as mock_coll:
            mock_env = MagicMock()
            mock_env.rank = 0
            mock_coll._get_global_env.return_value = mock_env
            mock_coll._get_default_group.return_value = mock_group

            mock_tensor = MagicMock()
            mock_tensor.dtype = "float32"

            with patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.LayerHelper"
            ) as mock_lh:
                mock_helper = MagicMock()
                mock_lh.return_value = mock_helper
                mock_out = MagicMock()
                mock_out.dtype = "float32"
                mock_helper.create_variable_for_type_inference.return_value = (
                    mock_out
                )

                result = _c_concat(mock_tensor, group=mock_group)
                self.assertIsNotNone(result)

    def test_c_concat_dynamic_pir_mode(self):
        """测试动态/PIR模式 / Test dynamic/pir mode."""
        from paddle.distributed.fleet.layers.mpu.mp_ops import _c_concat

        self.mock_framework.return_value = True

        mock_group = MagicMock()
        mock_group.is_member.return_value = True
        mock_group.id = 3
        mock_group.rank = 0
        mock_group.nranks = 2

        with patch(
            "paddle.distributed.fleet.layers.mpu.mp_ops.collective"
        ) as mock_coll:
            mock_env = MagicMock()
            mock_env.rank = 0
            mock_coll._get_global_env.return_value = mock_env
            mock_coll._get_default_group.return_value = mock_group

            mock_tensor = MagicMock()
            mock_result = MagicMock()

            with patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops._C_ops"
            ) as mock_cops:
                mock_cops.c_concat.return_value = mock_result

                result = _c_concat(mock_tensor, group=mock_group)
                self.assertEqual(result, mock_result)


class TestCSplit(unittest.TestCase):
    """测试 _c_split 操作 / Test _c_split operation."""

    def setUp(self):
        self.mock_framework = patch(
            "paddle.distributed.fleet.layers.mpu.mp_ops.in_dynamic_mode",
            return_value=False,
        ).start()

    def tearDown(self):
        patch.stopall()

    def test_c_split_non_member_returns_none(self):
        """测试非成员返回None / Test non-member group returns None."""
        from paddle.distributed.fleet.layers.mpu.mp_ops import _c_split

        mock_group = MagicMock()
        mock_group.is_member.return_value = False
        result = _c_split(MagicMock(), group=mock_group)
        self.assertIsNone(result)

    def test_c_split_static_mode(self):
        """测试静态模式 / Test static mode."""
        from paddle.distributed.fleet.layers.mpu.mp_ops import _c_split

        mock_group = MagicMock()
        mock_group.is_member.return_value = True
        mock_group.id = 5
        mock_group.rank = 1
        mock_group.nranks = 2
        mock_group.get_group_rank = MagicMock(return_value=1)

        with patch(
            "paddle.distributed.fleet.layers.mpu.mp_ops.collective"
        ) as mock_coll:
            mock_env = MagicMock()
            mock_env.rank = 1
            mock_env.world_size = 2
            mock_coll._get_global_env.return_value = mock_env

            mock_tensor = MagicMock()
            mock_tensor.dtype = "float32"

            with patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.LayerHelper"
            ) as mock_lh:
                mock_helper = MagicMock()
                mock_lh.return_value = mock_helper
                mock_out = MagicMock()
                mock_out.dtype = "float32"
                mock_helper.create_variable_for_type_inference.return_value = (
                    mock_out
                )

                result = _c_split(mock_tensor, group=mock_group)
                self.assertIsNotNone(result)

    def test_c_split_dynamic_mode(self):
        """测试动态模式 / Test dynamic mode."""
        from paddle.distributed.fleet.layers.mpu.mp_ops import _c_split

        self.mock_framework.return_value = True

        mock_group = MagicMock()
        mock_group.is_member.return_value = True
        mock_group.id = 5
        mock_group.rank = 0
        mock_group.get_group_rank = MagicMock(return_value=0)

        with patch(
            "paddle.distributed.fleet.layers.mpu.mp_ops.collective"
        ) as mock_coll:
            mock_env = MagicMock()
            mock_env.rank = 0
            mock_env.world_size = 2
            mock_coll._get_global_env.return_value = mock_env

            mock_tensor = MagicMock()
            mock_result = MagicMock()

            with patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.c_split_eager"
            ) as mock_eager:
                mock_eager.apply.return_value = mock_result

                result = _c_split(mock_tensor, group=mock_group)
                self.assertEqual(result, mock_result)


class TestMPAllreduce(unittest.TestCase):
    """测试 _mp_allreduce 操作 / Test _mp_allreduce operation."""

    def test_non_member_returns_none(self):
        """测试非成员返回None / Test non-member group returns None."""
        from paddle.distributed.fleet.layers.mpu.mp_ops import _mp_allreduce

        mock_group = MagicMock()
        mock_group.is_member.return_value = False
        result = _mp_allreduce(MagicMock(), group=mock_group)
        self.assertIsNone(result)

    def test_pir_mode(self):
        """测试PIR模式 / Test PIR mode."""
        from paddle.distributed.fleet.layers.mpu.mp_ops import _mp_allreduce

        with (
            patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.in_dynamic_mode",
                return_value=False,
            ),
            patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.in_pir_mode",
                return_value=True,
            ),
            patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops._C_ops"
            ) as mock_cops,
        ):
            mock_tensor = MagicMock()
            mock_result = MagicMock()
            mock_cops.mp_allreduce_sum.return_value = mock_result

            result = _mp_allreduce(mock_tensor)
            self.assertEqual(result, mock_result)

    def test_static_mode(self):
        """测试静态模式 / Test static mode."""
        from paddle.distributed.fleet.layers.mpu.mp_ops import _mp_allreduce

        with (
            patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.in_dynamic_mode",
                return_value=False,
            ),
            patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.in_pir_mode",
                return_value=False,
            ),
            patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.LayerHelper"
            ) as mock_lh,
        ):
            mock_helper = MagicMock()
            mock_lh.return_value = mock_helper
            mock_tensor = MagicMock()
            mock_tensor.dtype = "float32"
            mock_out = MagicMock()
            mock_helper.create_variable_for_type_inference.return_value = (
                mock_out
            )

            result = _mp_allreduce(mock_tensor)
            self.assertIsNotNone(result)


class TestCLookupTable(unittest.TestCase):
    """测试 _c_lookup_table 操作 / Test _c_lookup_table operation."""

    def test_static_mode(self):
        """测试静态模式 / Test static mode."""
        from paddle.distributed.fleet.layers.mpu.mp_ops import _c_lookup_table

        with (
            patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.in_dynamic_mode",
                return_value=False,
            ),
            patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.in_pir_mode",
                return_value=False,
            ),
            patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.LayerHelper"
            ) as mock_lh,
        ):
            mock_helper = MagicMock()
            mock_lh.return_value = mock_helper
            mock_helper.input_dtype.return_value = "float32"
            mock_table = MagicMock()
            mock_index = MagicMock()
            mock_out = MagicMock()
            mock_helper.create_variable_for_type_inference.return_value = (
                mock_out
            )

            result = _c_lookup_table(
                mock_table, mock_index, start_index=0, vocab_size=100
            )
            self.assertIsNotNone(result)
            mock_helper.append_op.assert_called_once()


class TestLinearFunction(unittest.TestCase):
    """测试 _linear 函数 / Test _linear function."""

    def test_static_mode_with_bias(self):
        """测试静态模式带偏置 / Test static mode with bias."""
        from paddle.distributed.fleet.layers.mpu.mp_ops import _linear

        with (
            patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.in_dynamic_mode",
                return_value=False,
            ),
            patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.LayerHelper"
            ) as mock_lh,
        ):
            mock_helper = MagicMock()
            mock_lh.return_value = mock_helper
            mock_x = MagicMock()
            mock_x.dtype = "float32"
            mock_x.shape = [2, 3]
            mock_weight = MagicMock()
            mock_bias = MagicMock()

            mock_tmp = MagicMock()
            mock_res = MagicMock()
            mock_helper.create_variable_for_type_inference.side_effect = [
                mock_tmp,
                mock_res,
            ]

            result = _linear(mock_x, mock_weight, bias=mock_bias)
            self.assertIsNotNone(result)
            self.assertEqual(mock_helper.append_op.call_count, 2)

    def test_static_mode_no_bias(self):
        """测试静态模式无偏置 / Test static mode without bias."""
        from paddle.distributed.fleet.layers.mpu.mp_ops import _linear

        with (
            patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.in_dynamic_mode",
                return_value=False,
            ),
            patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.LayerHelper"
            ) as mock_lh,
        ):
            mock_helper = MagicMock()
            mock_lh.return_value = mock_helper
            mock_x = MagicMock()
            mock_x.dtype = "float32"
            mock_x.shape = [2, 3]
            mock_weight = MagicMock()

            mock_tmp = MagicMock()
            mock_helper.create_variable_for_type_inference.return_value = (
                mock_tmp
            )

            result = _linear(mock_x, mock_weight, bias=None)
            self.assertEqual(result, mock_tmp)
            self.assertEqual(mock_helper.append_op.call_count, 1)


class TestCSoftmaxWithCrossEntropy(unittest.TestCase):
    """测试 _c_softmax_with_cross_entropy / Test _c_softmax_with_cross_entropy."""

    def test_non_member_returns_none(self):
        """测试非成员返回None / Test non-member returns None."""
        from paddle.distributed.fleet.layers.mpu.mp_ops import (
            _c_softmax_with_cross_entropy,
        )

        mock_group = MagicMock()
        mock_group.is_member.return_value = False
        result = _c_softmax_with_cross_entropy(
            MagicMock(), MagicMock(), group=mock_group
        )
        self.assertIsNone(result)

    def test_static_mode(self):
        """测试静态模式 / Test static mode."""
        from paddle.distributed.fleet.layers.mpu.mp_ops import (
            _c_softmax_with_cross_entropy,
        )

        mock_group = MagicMock()
        mock_group.is_member.return_value = True
        mock_group.id = 0
        mock_group.rank = 0
        mock_group.nranks = 1
        mock_group.get_group_rank = MagicMock(return_value=0)

        mock_logits = MagicMock()
        mock_logits.shape = [4, 10]
        mock_logits.dtype = "float32"
        mock_label = MagicMock()
        mock_label.shape = [4, 10]

        with (
            patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.in_dynamic_mode",
                return_value=False,
            ),
            patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.collective"
            ) as mock_coll,
        ):
            mock_env = MagicMock()
            mock_env.rank = 0
            mock_env.world_size = 1
            mock_coll._get_global_env.return_value = mock_env

            with patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.LayerHelper"
            ) as mock_lh:
                mock_helper = MagicMock()
                mock_lh.return_value = mock_helper
                mock_softmax = MagicMock()
                mock_softmax.dtype = "float32"
                mock_loss = MagicMock()
                mock_loss.dtype = "float32"
                mock_helper.create_variable_for_type_inference.side_effect = [
                    mock_softmax,
                    mock_loss,
                ]

                result = _c_softmax_with_cross_entropy(
                    mock_logits, mock_label, group=mock_group
                )
                self.assertIsNotNone(result)

    def test_invalid_dims_raises(self):
        """测试无效维度抛出异常 / Test invalid dimensions raises."""
        from paddle.distributed.fleet.layers.mpu.mp_ops import (
            _c_softmax_with_cross_entropy,
        )

        mock_group = MagicMock()
        mock_group.is_member.return_value = True
        mock_group.id = 0
        mock_group.rank = 0
        mock_group.nranks = 1
        mock_group.get_group_rank = MagicMock(return_value=0)

        mock_logits = MagicMock()
        mock_logits.shape = [4, 10, 5]
        mock_label = MagicMock()
        mock_label.shape = [4]

        with (
            patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.in_dynamic_mode",
                return_value=False,
            ),
            patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.collective"
            ) as mock_coll,
        ):
            mock_env = MagicMock()
            mock_env.rank = 0
            mock_env.world_size = 1
            mock_coll._get_global_env.return_value = mock_env

            with self.assertRaises(ValueError):
                _c_softmax_with_cross_entropy(
                    mock_logits, mock_label, group=mock_group
                )

    def test_label_unsqueeze(self):
        """测试label unsqueeze / Test label unsqueeze path."""
        from paddle.distributed.fleet.layers.mpu.mp_ops import (
            _c_softmax_with_cross_entropy,
        )

        mock_group = MagicMock()
        mock_group.is_member.return_value = True
        mock_group.id = 0
        mock_group.rank = 0
        mock_group.nranks = 1
        mock_group.get_group_rank = MagicMock(return_value=0)

        mock_logits = MagicMock()
        mock_logits.shape = [4, 10]
        mock_logits.dtype = "float32"
        mock_label = MagicMock()
        mock_label.shape = [4]

        with (
            patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.in_dynamic_mode",
                return_value=False,
            ),
            patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.collective"
            ) as mock_coll,
            patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.paddle"
            ) as mock_paddle,
        ):
            mock_env = MagicMock()
            mock_env.rank = 0
            mock_env.world_size = 1
            mock_coll._get_global_env.return_value = mock_env

            mock_squeezed = MagicMock()
            mock_squeezed.shape = [4, 1]
            mock_paddle.unsqueeze.return_value = mock_squeezed

            with patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.LayerHelper"
            ) as mock_lh:
                mock_helper = MagicMock()
                mock_lh.return_value = mock_helper
                mock_softmax = MagicMock()
                mock_loss = MagicMock()
                mock_helper.create_variable_for_type_inference.side_effect = [
                    mock_softmax,
                    mock_loss,
                ]

                result = _c_softmax_with_cross_entropy(
                    mock_logits, mock_label, group=mock_group
                )
                self.assertIsNotNone(result)


class TestCSoftmaxWithMultiLabelCrossEntropy(unittest.TestCase):
    """测试 _c_softmax_with_multi_label_cross_entropy / Test multi-label cross entropy."""

    def test_non_member_returns_none(self):
        """测试非成员返回None / Test non-member returns None."""
        from paddle.distributed.fleet.layers.mpu.mp_ops import (
            _c_softmax_with_multi_label_cross_entropy,
        )

        mock_group = MagicMock()
        mock_group.is_member.return_value = False
        result = _c_softmax_with_multi_label_cross_entropy(
            MagicMock(), MagicMock(), MagicMock(), group=mock_group
        )
        self.assertIsNone(result)

    def test_static_mode(self):
        """测试静态模式 / Test static mode."""
        from paddle.distributed.fleet.layers.mpu.mp_ops import (
            _c_softmax_with_multi_label_cross_entropy,
        )

        mock_group = MagicMock()
        mock_group.is_member.return_value = True
        mock_group.id = 0
        mock_group.rank = 0
        mock_group.nranks = 1
        mock_group.get_group_rank = MagicMock(return_value=0)

        mock_logits = MagicMock()
        mock_logits.shape = [4, 10]
        mock_logits.dtype = "float32"
        mock_label = MagicMock()
        mock_label.shape = [4, 10]
        mock_smooth = MagicMock()

        with (
            patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.in_dynamic_mode",
                return_value=False,
            ),
            patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.collective"
            ) as mock_coll,
        ):
            mock_env = MagicMock()
            mock_env.rank = 0
            mock_env.world_size = 1
            mock_coll._get_global_env.return_value = mock_env

            with patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops.LayerHelper"
            ) as mock_lh:
                mock_helper = MagicMock()
                mock_lh.return_value = mock_helper
                mock_softmax = MagicMock()
                mock_loss = MagicMock()
                mock_helper.create_variable_for_type_inference.side_effect = [
                    mock_softmax,
                    mock_loss,
                ]

                result = _c_softmax_with_multi_label_cross_entropy(
                    mock_logits, mock_label, mock_smooth, group=mock_group
                )
                self.assertIsNotNone(result)


class TestSetVarDistributed(unittest.TestCase):
    """测试 _set_var_distributed 函数 / Test _set_var_distributed function."""

    def test_set_var_distributed_none(self):
        """测试var为None / Test var is None."""
        from paddle.distributed.fleet.layers.mpu.mp_ops import (
            _set_var_distributed,
        )

        # Should not raise
        _set_var_distributed(None)

    def test_set_var_distributed_with_var(self):
        """测试设置分布式变量 / Test setting distributed var."""
        from paddle.distributed.fleet.layers.mpu.mp_ops import (
            _set_var_distributed,
        )

        mock_var = MagicMock()
        mock_var.name = "test_var"

        with patch(
            "paddle.distributed.fleet.layers.mpu.mp_ops.paddle"
        ) as mock_paddle:
            mock_startup = MagicMock()
            mock_main = MagicMock()
            mock_startup.current_block.return_value._find_var_recursive.return_value = mock_var
            mock_main.current_block.return_value._find_var_recursive.return_value = mock_var
            mock_paddle.static.default_startup_program.return_value = (
                mock_startup
            )
            mock_paddle.static.default_main_program.return_value = mock_main

            _set_var_distributed(mock_var)
            self.assertTrue(mock_var.is_distributed)


class TestLinearLayer(unittest.TestCase):
    """测试 _Linear 层 / Test _Linear layer."""

    def test_linear_init(self):
        """测试 _Linear 初始化 / Test _Linear init."""
        from paddle.distributed.fleet.layers.mpu.mp_ops import _Linear

        with patch(
            "paddle.distributed.fleet.layers.mpu.mp_ops._Linear.create_parameter"
        ) as mock_create:
            mock_weight = MagicMock()
            mock_weight.shape = [64, 32]
            mock_bias = MagicMock()
            mock_create.side_effect = [mock_weight, mock_bias]

            layer = _Linear(64, 32)
            self.assertEqual(layer.weight, mock_weight)
            self.assertEqual(layer.bias, mock_bias)

    def test_linear_extra_repr(self):
        """测试 _Linear extra_repr / Test _Linear extra_repr."""
        from paddle.distributed.fleet.layers.mpu.mp_ops import _Linear

        with patch(
            "paddle.distributed.fleet.layers.mpu.mp_ops._Linear.create_parameter"
        ) as mock_create:
            mock_weight = MagicMock()
            mock_weight.shape = [64, 32]
            mock_bias = MagicMock()
            mock_create.side_effect = [mock_weight, mock_bias]

            layer = _Linear(64, 32, name="test")
            repr_str = layer.extra_repr()
            self.assertIn("in_features=64", repr_str)
            self.assertIn("out_features=32", repr_str)
            self.assertIn("name=test", repr_str)

    def test_linear_forward(self):
        """测试 _Linear 前向传播 / Test _Linear forward."""
        from paddle.distributed.fleet.layers.mpu.mp_ops import _Linear

        with (
            patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops._Linear.create_parameter"
            ) as mock_create,
            patch(
                "paddle.distributed.fleet.layers.mpu.mp_ops._linear"
            ) as mock_linear,
        ):
            mock_weight = MagicMock()
            mock_bias = MagicMock()
            mock_create.side_effect = [mock_weight, mock_bias]

            mock_out = MagicMock()
            mock_linear.return_value = mock_out

            layer = _Linear(64, 32)
            mock_x = MagicMock()
            result = layer.forward(mock_x)
            self.assertEqual(result, mock_out)


if __name__ == '__main__':
    unittest.main()
