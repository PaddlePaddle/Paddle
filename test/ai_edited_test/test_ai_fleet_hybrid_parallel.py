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

# [AUTO-GENERATED] Test file for paddle.distributed.fleet.utils.hybrid_parallel_util
# Target file: paddle/distributed/fleet/utils/hybrid_parallel_util.py
# 覆盖模块: paddle/distributed/fleet/utils/hybrid_parallel_util.py
# Covered module: paddle/distributed/fleet/utils/hybrid_parallel_util.py

import unittest
from unittest.mock import MagicMock, patch


class TestObtainOptimizerParametersList(unittest.TestCase):
    """测试 obtain_optimizer_parameters_list 函数
    Test obtain_optimizer_parameters_list function"""

    def test_obtain_params_from_param_groups(self):
        """测试从 _param_groups 获取参数 / Test obtaining params from _param_groups"""
        from paddle.distributed.fleet.utils.hybrid_parallel_util import (
            obtain_optimizer_parameters_list,
        )

        mock_param1 = MagicMock()
        mock_param2 = MagicMock()
        mock_opt = MagicMock()
        mock_opt._param_groups = [{'params': [mock_param1, mock_param2]}]
        result = obtain_optimizer_parameters_list(mock_opt)
        self.assertEqual(result, [mock_param1, mock_param2])

    def test_obtain_params_from_param_groups_multiple_groups(self):
        """测试从多个参数组获取参数 / Test obtaining params from multiple groups"""
        from paddle.distributed.fleet.utils.hybrid_parallel_util import (
            obtain_optimizer_parameters_list,
        )

        mock_param1 = MagicMock()
        mock_param2 = MagicMock()
        mock_param3 = MagicMock()
        mock_opt = MagicMock()
        mock_opt._param_groups = [
            {'params': [mock_param1]},
            {'params': [mock_param2, mock_param3]},
        ]
        result = obtain_optimizer_parameters_list(mock_opt)
        self.assertEqual(result, [mock_param1, mock_param2, mock_param3])

    def test_obtain_params_from_parameter_list(self):
        """测试从 _parameter_list 获取参数 / Test obtaining params from _parameter_list"""
        from paddle.distributed.fleet.utils.hybrid_parallel_util import (
            obtain_optimizer_parameters_list,
        )

        mock_param1 = MagicMock()
        mock_param2 = MagicMock()
        mock_opt = MagicMock()
        mock_opt._param_groups = None
        mock_opt._parameter_list = [mock_param1, mock_param2]
        result = obtain_optimizer_parameters_list(mock_opt)
        self.assertEqual(result, [mock_param1, mock_param2])


class TestUnwrapOptimizer(unittest.TestCase):
    """测试 unwrap_optimizer 函数 / Test unwrap_optimizer function"""

    def test_unwrap_optimizer_no_wrapping(self):
        """测试无包装时的解包 / Test unwrap with no wrapping"""
        from paddle.distributed.fleet.utils.hybrid_parallel_util import (
            unwrap_optimizer,
        )

        mock_opt = MagicMock()
        result = unwrap_optimizer(mock_opt)
        self.assertIs(result, mock_opt)

    def test_unwrap_optimizer_single_wrap(self):
        """测试单层包装时的解包 / Test unwrap with single wrapping"""
        from paddle.distributed.fleet.utils.hybrid_parallel_util import (
            unwrap_optimizer,
        )

        inner = MagicMock()
        wrapper = MagicMock()
        wrapper._inner_opt = inner
        result = unwrap_optimizer(wrapper, (type(wrapper),))
        self.assertIs(result, inner)

    def test_unwrap_optimizer_multi_wrap(self):
        """测试多层包装时的解包 / Test unwrap with multiple wrappings"""
        from paddle.distributed.fleet.utils.hybrid_parallel_util import (
            unwrap_optimizer,
        )

        innermost = MagicMock()
        middle = MagicMock()
        middle._inner_opt = innermost
        outer = MagicMock()
        outer._inner_opt = middle
        result = unwrap_optimizer(outer, (type(outer), type(middle)))
        self.assertIs(result, innermost)


class TestBroadcastNestedData(unittest.TestCase):
    """测试 _broadcast_nested_data 函数 / Test _broadcast_nested_data function"""

    @patch(
        'paddle.distributed.fleet.utils.hybrid_parallel_util._broadcast_object_list_help'
    )
    def test_broadcast_nested_data_unsupported_type(self, mock_broadcast_obj):
        """测试不支持的类型抛出 TypeError / Test unsupported type raises TypeError"""
        from paddle.distributed.fleet.utils.hybrid_parallel_util import (
            _broadcast_nested_data,
        )

        mock_hcg = MagicMock()
        with self.assertRaises(TypeError):
            _broadcast_nested_data(mock_hcg, 'gpu', MagicMock(), 12345)


class TestBroadcastInputData(unittest.TestCase):
    """测试 broadcast_input_data 函数 / Test broadcast_input_data function"""

    @patch('paddle.device.get_all_custom_device_type', return_value=[])
    @patch('paddle.get_device', return_value='gpu:0')
    def test_broadcast_input_data_gpu(self, mock_get_device, mock_custom):
        """测试 GPU 设备上的 broadcast_input_data
        Test broadcast_input_data on GPU device"""
        from paddle.distributed.fleet.utils.hybrid_parallel_util import (
            broadcast_input_data,
        )

        mock_hcg = MagicMock()
        with patch(
            'paddle.distributed.fleet.utils.hybrid_parallel_util._broadcast_nested_data'
        ) as mock_nested:
            # First call for inputs, second for kwargs
            mock_nested.side_effect = [('input1', 'input2'), {'lr': 0.001}]
            inputs, kwargs = broadcast_input_data(
                mock_hcg, "data1", "data2", lr=0.001
            )
            self.assertEqual(inputs, ('input1', 'input2'))
            self.assertEqual(kwargs, {'lr': 0.001})

    @patch(
        'paddle.device.get_all_custom_device_type', return_value=['custom_dev']
    )
    @patch('paddle.get_device', return_value='custom_dev:0')
    @patch('paddle.CustomPlace')
    def test_broadcast_input_data_custom_device(
        self, mock_cp, mock_get_device, mock_custom
    ):
        """测试自定义设备上的 broadcast_input_data
        Test broadcast_input_data on custom device"""
        from paddle.distributed.fleet.utils.hybrid_parallel_util import (
            broadcast_input_data,
        )

        mock_hcg = MagicMock()
        with patch(
            'paddle.distributed.fleet.utils.hybrid_parallel_util._broadcast_nested_data'
        ) as mock_nested:
            mock_nested.side_effect = [('d1',), {}]
            inputs, kwargs = broadcast_input_data(mock_hcg, "data1")
            self.assertEqual(inputs, ('d1',))
            self.assertEqual(kwargs, {})

    @patch('paddle.get_device', return_value='cpu:0')
    @patch('paddle.device.get_all_custom_device_type', return_value=[])
    def test_broadcast_input_data_cpu_raises(
        self, mock_custom, mock_get_device
    ):
        """测试 CPU 设备上的 broadcast_input_data 抛出异常
        Test broadcast_input_data on CPU raises assertion"""
        from paddle.distributed.fleet.utils.hybrid_parallel_util import (
            broadcast_input_data,
        )

        mock_hcg = MagicMock()
        with self.assertRaises(AssertionError):
            broadcast_input_data(mock_hcg)


class TestFusedAllreduceGradients(unittest.TestCase):
    """测试 fused_allreduce_gradients 函数 / Test fused_allreduce_gradients function"""

    @patch('paddle.distributed.in_auto_parallel_align_mode', return_value=False)
    @patch(
        'paddle.distributed.fleet.utils.hybrid_parallel_util.fused_allreduce_gradients_with_group'
    )
    def test_fused_allreduce_gradients_none_hcg(self, mock_fused, mock_auto):
        """测试 hcg 为 None 时 fused_allreduce_gradients
        Test fused_allreduce_gradients with None hcg"""
        from paddle.distributed.fleet.utils.hybrid_parallel_util import (
            fused_allreduce_gradients,
        )

        fused_allreduce_gradients([], None)
        mock_fused.assert_called_once()

    @patch('paddle.distributed.in_auto_parallel_align_mode', return_value=True)
    @patch(
        'paddle.distributed.fleet.utils.hybrid_parallel_util.fused_allreduce_gradients_with_group'
    )
    def test_fused_allreduce_gradients_auto_parallel(
        self, mock_fused, mock_auto
    ):
        """测试自动并行模式下的 fused_allreduce_gradients
        Test fused_allreduce_gradients in auto parallel mode"""
        from paddle.distributed.fleet.utils.hybrid_parallel_util import (
            fused_allreduce_gradients,
        )

        fused_allreduce_gradients([], None)
        # scale should be 1.0 in auto_parallel_align_mode
        call_args = mock_fused.call_args
        self.assertEqual(call_args[1].get('scale'), 1.0)

    @patch('paddle.distributed.in_auto_parallel_align_mode', return_value=False)
    @patch(
        'paddle.distributed.fleet.utils.hybrid_parallel_util.fused_allreduce_gradients_with_group'
    )
    def test_fused_allreduce_gradients_dp_enabled(self, mock_fused, mock_auto):
        """测试数据并行启用时的 fused_allreduce_gradients
        Test fused_allreduce_gradients with DP enabled"""
        from paddle.distributed.fleet.utils.hybrid_parallel_util import (
            fused_allreduce_gradients,
        )

        mock_hcg = MagicMock()
        mock_dp_group = MagicMock()
        mock_dp_group.nranks = 4
        mock_hcg.get_data_parallel_world_size.return_value = 4
        mock_hcg.get_sep_parallel_world_size.return_value = 1
        mock_hcg.get_data_parallel_group.return_value = mock_dp_group
        fused_allreduce_gradients([], mock_hcg)
        mock_fused.assert_called_once()

    @patch('paddle.distributed.in_auto_parallel_align_mode', return_value=False)
    @patch(
        'paddle.distributed.fleet.utils.hybrid_parallel_util.fused_allreduce_gradients_with_group'
    )
    def test_fused_allreduce_gradients_sep_enabled(self, mock_fused, mock_auto):
        """测试序列并行启用时的 fused_allreduce_gradients
        Test fused_allreduce_gradients with SEP enabled"""
        from paddle.distributed.fleet.utils.hybrid_parallel_util import (
            fused_allreduce_gradients,
        )

        mock_hcg = MagicMock()
        mock_sep_group = MagicMock()
        mock_dp_sep_group = MagicMock()
        mock_hcg.get_data_parallel_world_size.return_value = 1
        mock_hcg.get_sep_parallel_world_size.return_value = 2
        mock_hcg.get_sep_parallel_group.return_value = mock_sep_group
        mock_hcg.get_dp_sep_parallel_group.return_value = mock_dp_sep_group
        fused_allreduce_gradients([], mock_hcg)
        mock_fused.assert_called_once()

    @patch('paddle.distributed.in_auto_parallel_align_mode', return_value=False)
    def test_fused_allreduce_gradients_assertion(self, mock_auto):
        """测试两者都禁用时 fused_allreduce_gradients 抛出异常
        Test fused_allreduce_gradients raises assertion when both disabled"""
        from paddle.distributed.fleet.utils.hybrid_parallel_util import (
            fused_allreduce_gradients,
        )

        mock_hcg = MagicMock()
        mock_hcg.get_data_parallel_world_size.return_value = 1
        mock_hcg.get_sep_parallel_world_size.return_value = 1
        with self.assertRaises(AssertionError):
            fused_allreduce_gradients([], mock_hcg)


class TestBroadcastMpDpSepShardingParameters(unittest.TestCase):
    """测试 broadcast 参数函数 / Test broadcast parameter functions"""

    @patch(
        'paddle.distributed.fleet.utils.hybrid_parallel_util.sync_params_buffers'
    )
    def test_broadcast_mp_parameters(self, mock_sync):
        """测试 broadcast_mp_parameters / Test broadcast_mp_parameters"""
        from paddle.distributed.fleet.utils.hybrid_parallel_util import (
            broadcast_mp_parameters,
        )

        mock_model = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg.get_model_parallel_group.return_value = "mp_group"
        mock_hcg.get_model_parallel_group_src_rank.return_value = 0
        broadcast_mp_parameters(mock_model, mock_hcg)
        mock_sync.assert_called_once()

    @patch(
        'paddle.distributed.fleet.utils.hybrid_parallel_util.sync_params_buffers'
    )
    def test_broadcast_dp_parameters(self, mock_sync):
        """测试 broadcast_dp_parameters / Test broadcast_dp_parameters"""
        from paddle.distributed.fleet.utils.hybrid_parallel_util import (
            broadcast_dp_parameters,
        )

        mock_model = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg.get_data_parallel_group.return_value = "dp_group"
        mock_hcg.get_data_parallel_group_src_rank.return_value = 0
        broadcast_dp_parameters(mock_model, mock_hcg)
        mock_sync.assert_called_once()

    @patch(
        'paddle.distributed.fleet.utils.hybrid_parallel_util.sync_params_buffers'
    )
    def test_broadcast_sharding_parameters(self, mock_sync):
        """测试 broadcast_sharding_parameters / Test broadcast_sharding_parameters"""
        from paddle.distributed.fleet.utils.hybrid_parallel_util import (
            broadcast_sharding_parameters,
        )

        mock_model = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg.get_sharding_parallel_group.return_value = "sharding_group"
        mock_hcg.get_sharding_parallel_group_src_rank.return_value = 0
        broadcast_sharding_parameters(mock_model, mock_hcg)
        mock_sync.assert_called_once()

    @patch(
        'paddle.distributed.fleet.utils.hybrid_parallel_util.sync_params_buffers'
    )
    def test_broadcast_sep_parameters(self, mock_sync):
        """测试 broadcast_sep_parameters / Test broadcast_sep_parameters"""
        from paddle.distributed.fleet.utils.hybrid_parallel_util import (
            broadcast_sep_parameters,
        )

        mock_model = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg.get_sep_parallel_group.return_value = "sep_group"
        mock_hcg.get_sep_parallel_group_src_rank.return_value = 0
        broadcast_sep_parameters(mock_model, mock_hcg)
        mock_sync.assert_called_once()

    @patch(
        'paddle.distributed.fleet.utils.hybrid_parallel_util.sync_params_buffers'
    )
    def test_broadcast_moe_sharding_parameters(self, mock_sync):
        """测试 broadcast_moe_sharding_parameters
        Test broadcast_moe_sharding_parameters"""
        from paddle.distributed.fleet.utils.hybrid_parallel_util import (
            broadcast_moe_sharding_parameters,
        )

        mock_model = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg.get_moe_sharding_parallel_group.return_value = "moe_group"
        mock_hcg.get_moe_sharding_parallel_group_src_rank.return_value = 0
        broadcast_moe_sharding_parameters(mock_model, mock_hcg)
        mock_sync.assert_called_once()
        # Check is_moe_sharding_parallel is passed
        call_kwargs = mock_sync.call_args[1]
        self.assertTrue(call_kwargs.get('is_moe_sharding_parallel', False))

    @patch(
        'paddle.distributed.fleet.utils.hybrid_parallel_util.sync_params_buffers'
    )
    def test_broadcast_mp_parameters_no_fuse(self, mock_sync):
        """测试 broadcast_mp_parameters 不融合参数
        Test broadcast_mp_parameters with fuse_params=False"""
        from paddle.distributed.fleet.utils.hybrid_parallel_util import (
            broadcast_mp_parameters,
        )

        mock_model = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg.get_model_parallel_group.return_value = "mp_group"
        mock_hcg.get_model_parallel_group_src_rank.return_value = 0
        broadcast_mp_parameters(mock_model, mock_hcg, fuse_params=False)
        call_kwargs = mock_sync.call_args[1]
        self.assertFalse(call_kwargs['fuse_params'])


class TestModuleImport(unittest.TestCase):
    """测试模块导入 / Test module import"""

    def test_module_import(self):
        """测试 hybrid_parallel_util 模块可导入
        Test hybrid_parallel_util module can be imported"""
        from paddle.distributed.fleet.utils import hybrid_parallel_util

        self.assertIsNotNone(hybrid_parallel_util)

    def test_module_has_functions(self):
        """测试 hybrid_parallel_util 模块有函数
        Test hybrid_parallel_util module has functions"""
        from paddle.distributed.fleet.utils import hybrid_parallel_util

        self.assertTrue(len(dir(hybrid_parallel_util)) > 0)

    def test_all_export(self):
        """测试 __all__ 导出 / Test __all__ exports"""
        from paddle.distributed.fleet.utils import hybrid_parallel_util

        self.assertEqual(hybrid_parallel_util.__all__, [])


if __name__ == '__main__':
    unittest.main()
