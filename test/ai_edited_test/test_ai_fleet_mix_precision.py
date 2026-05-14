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

# [AUTO-GENERATED] Test file for paddle.distributed.fleet.utils.mix_precision_utils
# Target file: paddle/distributed/fleet/utils/mix_precision_utils.py
# 覆盖模块: paddle/distributed/fleet/utils/mix_precision_utils.py
# Covered module: paddle/distributed/fleet/utils/mix_precision_utils.py

import unittest
from unittest.mock import MagicMock, patch


class TestMixPrecisionLayer(unittest.TestCase):
    """测试 MixPrecisionLayer 类 / Test MixPrecisionLayer class"""

    def test_mix_precision_layer_import(self):
        """测试 MixPrecisionLayer 可导入 / Test MixPrecisionLayer can be imported"""
        from paddle.distributed.fleet.utils.mix_precision_utils import (
            MixPrecisionLayer,
        )

        self.assertIsNotNone(MixPrecisionLayer)

    def test_mix_precision_layer_invalid_dtype(self):
        """测试无效 dtype 抛出断言异常 / Test invalid dtype raises assertion error"""
        from paddle.distributed.fleet.utils.mix_precision_utils import (
            MixPrecisionLayer,
        )

        with patch('paddle.device.synchronize'):
            layer = MagicMock()
            layer.full_name.return_value = "test_layer"
            layer.parameters.return_value = []
            with self.assertRaises(AssertionError):
                MixPrecisionLayer(layer, dtype="float32")

    def test_mix_precision_layer_bfloat16(self):
        """测试 bfloat16 类型的 MixPrecisionLayer
        Test MixPrecisionLayer with bfloat16 dtype"""
        from paddle.distributed.fleet.utils.mix_precision_utils import (
            MixPrecisionLayer,
        )

        with patch('paddle.device.synchronize'):
            layer = MagicMock()
            layer.full_name.return_value = "test_layer"
            layer.parameters.return_value = []
            mp_layer = MixPrecisionLayer(layer, dtype="bfloat16")
            self.assertIsNotNone(mp_layer)

    def test_mix_precision_layer_float16(self):
        """测试 float16 类型的 MixPrecisionLayer
        Test MixPrecisionLayer with float16 dtype"""
        from paddle.distributed.fleet.utils.mix_precision_utils import (
            MixPrecisionLayer,
        )

        with patch('paddle.device.synchronize'):
            layer = MagicMock()
            layer.full_name.return_value = "test_layer"
            layer.parameters.return_value = []
            mp_layer = MixPrecisionLayer(layer, dtype="float16")
            self.assertIsNotNone(mp_layer)


class TestMixPrecisionOptimizer(unittest.TestCase):
    """测试 MixPrecisionOptimizer 类 / Test MixPrecisionOptimizer class"""

    def test_mix_precision_optimizer_import(self):
        """测试 MixPrecisionOptimizer 可导入 / Test MixPrecisionOptimizer can be imported"""
        from paddle.distributed.fleet.utils.mix_precision_utils import (
            MixPrecisionOptimizer,
        )

        self.assertIsNotNone(MixPrecisionOptimizer)

    def test_mix_precision_optimizer_getattr(self):
        """测试 MixPrecisionOptimizer 属性代理 / Test MixPrecisionOptimizer attribute proxy"""
        from paddle.distributed.fleet.utils.mix_precision_utils import (
            MixPrecisionOptimizer,
        )

        mock_opt = MagicMock()
        mock_opt._parameter_list = []
        mock_opt._param_groups = []
        mock_opt.learning_rate = 0.001
        opt = MixPrecisionOptimizer(mock_opt)
        self.assertEqual(opt.learning_rate, 0.001)

    @patch(
        'paddle.distributed.fleet.utils.mix_precision_utils.obtain_optimizer_parameters_list'
    )
    def test_mix_precision_optimizer_init(self, mock_obtain):
        """测试 MixPrecisionOptimizer 初始化 / Test MixPrecisionOptimizer initialization"""
        from paddle.distributed.fleet.utils.mix_precision_utils import (
            MixPrecisionOptimizer,
        )

        mock_param = MagicMock()
        mock_param.stop_gradient = False
        mock_obtain.return_value = [mock_param]
        mock_opt = MagicMock()
        mock_opt._parameter_list = [mock_param]
        mock_opt._param_groups = []
        opt = MixPrecisionOptimizer(mock_opt)
        self.assertEqual(opt._inner_opt, mock_opt)

    @patch(
        'paddle.distributed.fleet.utils.mix_precision_utils.obtain_optimizer_parameters_list'
    )
    @patch('paddle.in_dynamic_mode', return_value=True)
    @patch('paddle.base.framework.in_dygraph_mode', return_value=True)
    def test_mix_precision_optimizer_clear_grad_no_param_list(
        self, mock_dygraph, mock_dyn, mock_obtain
    ):
        """测试 clear_grad 空参数列表 / Test clear_grad with empty param list"""
        from paddle.distributed.fleet.utils.mix_precision_utils import (
            MixPrecisionOptimizer,
        )

        mock_obtain.return_value = []
        mock_opt = MagicMock()
        opt = MixPrecisionOptimizer(mock_opt)
        # Empty list will cause IndexError - this is expected behavior
        with self.assertRaises(IndexError):
            opt.clear_grad()

    @patch(
        'paddle.distributed.fleet.utils.mix_precision_utils.obtain_optimizer_parameters_list'
    )
    @patch('paddle.in_dynamic_mode', return_value=True)
    @patch('paddle.base.framework.in_dygraph_mode', return_value=True)
    def test_mix_precision_optimizer_clear_grad_stop_gradient(
        self, mock_dygraph, mock_dyn, mock_obtain
    ):
        """测试 clear_grad 跳过 stop_gradient 参数
        Test clear_grad skips stop_gradient params"""
        from paddle.distributed.fleet.utils.mix_precision_utils import (
            MixPrecisionOptimizer,
        )

        mock_param = MagicMock()
        mock_param.stop_gradient = True
        mock_obtain.return_value = [mock_param]
        mock_opt = MagicMock()
        opt = MixPrecisionOptimizer(mock_opt)
        opt.clear_grad()

    @patch(
        'paddle.distributed.fleet.utils.mix_precision_utils.obtain_optimizer_parameters_list'
    )
    @patch('paddle.in_dynamic_mode', return_value=True)
    @patch('paddle.base.framework.in_dygraph_mode', return_value=True)
    def test_mix_precision_optimizer_clear_grad_set_to_zero(
        self, mock_dygraph, mock_dyn, mock_obtain
    ):
        """测试 clear_grad set_to_zero=True / Test clear_grad set_to_zero=True"""
        from paddle.distributed.fleet.utils.mix_precision_utils import (
            MixPrecisionOptimizer,
        )

        mock_param = MagicMock()
        mock_param.stop_gradient = False
        mock_param.main_grad = MagicMock()
        mock_obtain.return_value = [mock_param]
        mock_opt = MagicMock()
        opt = MixPrecisionOptimizer(mock_opt)
        opt.clear_grad(set_to_zero=True)
        mock_param.main_grad.zero_.assert_called_once()

    @patch(
        'paddle.distributed.fleet.utils.mix_precision_utils.obtain_optimizer_parameters_list'
    )
    @patch('paddle.in_dynamic_mode', return_value=True)
    @patch('paddle.base.framework.in_dygraph_mode', return_value=True)
    def test_mix_precision_optimizer_clear_grad_not_set_to_zero(
        self, mock_dygraph, mock_dyn, mock_obtain
    ):
        """测试 clear_grad set_to_zero=False / Test clear_grad set_to_zero=False"""
        from paddle.distributed.fleet.utils.mix_precision_utils import (
            MixPrecisionOptimizer,
        )

        main_grad_mock = MagicMock()
        mock_param = MagicMock()
        mock_param.stop_gradient = False
        mock_param.main_grad = main_grad_mock
        mock_obtain.return_value = [mock_param]
        mock_opt = MagicMock()
        opt = MixPrecisionOptimizer(mock_opt)
        opt.clear_grad(set_to_zero=False)
        main_grad_mock._clear.assert_called_once()

    @patch(
        'paddle.distributed.fleet.utils.mix_precision_utils.obtain_optimizer_parameters_list'
    )
    @patch('paddle.in_dynamic_mode', return_value=True)
    @patch('paddle.base.framework.in_dygraph_mode', return_value=True)
    def test_mix_precision_optimizer_clear_grad_no_main_grad(
        self, mock_dygraph, mock_dyn, mock_obtain
    ):
        """测试 clear_grad 无 main_grad 属性 / Test clear_grad without main_grad"""
        from paddle.distributed.fleet.utils.mix_precision_utils import (
            MixPrecisionOptimizer,
        )

        mock_param = MagicMock()
        mock_param.stop_gradient = False
        del mock_param.main_grad
        mock_obtain.return_value = [mock_param]
        mock_opt = MagicMock()
        opt = MixPrecisionOptimizer(mock_opt)
        opt.clear_grad()


class TestUnscaleMethod(unittest.TestCase):
    """测试 unscale_method 函数 / Test unscale_method function"""

    def test_unscale_method_import(self):
        """测试 unscale_method 可导入 / Test unscale_method can be imported"""
        from paddle.distributed.fleet.utils.mix_precision_utils import (
            unscale_method,
        )

        self.assertIsNotNone(unscale_method)

    def test_unscale_method_disabled(self):
        """测试 AMP 未启用时的 unscale_method / Test unscale_method when AMP disabled"""
        from paddle.distributed.fleet.utils.mix_precision_utils import (
            unscale_method,
        )

        mock_scaler = MagicMock()
        mock_scaler._enable = False
        mock_optimizer = MagicMock()
        # Should return early
        unscale_method(mock_scaler, mock_optimizer)

    @patch('paddle.to_tensor')
    @patch(
        'paddle.distributed.fleet.get_hybrid_communicate_group',
        return_value=None,
    )
    def test_unscale_method_enabled_no_params(self, mock_hcg, mock_to_tensor):
        """测试启用但无参数时的 unscale_method / Test unscale_method enabled with no params"""
        from paddle.distributed.fleet.utils.mix_precision_utils import (
            unscale_method,
        )

        mock_scaler = MagicMock()
        mock_scaler._enable = True
        mock_optimizer = MagicMock()
        mock_optimizer._parameter_list = []
        mock_to_tensor.return_value = MagicMock()
        unscale_method(mock_scaler, mock_optimizer)

    @patch('paddle.distributed.all_reduce')
    @patch('paddle.to_tensor')
    @patch('paddle.distributed.fleet.get_hybrid_communicate_group')
    def test_unscale_method_with_hcg(
        self, mock_get_hcg, mock_to_tensor, mock_all_reduce
    ):
        """测试有 hcg 时的 unscale_method / Test unscale_method with hcg"""
        from paddle.distributed.fleet.utils.mix_precision_utils import (
            unscale_method,
        )

        mock_scaler = MagicMock()
        mock_scaler._enable = True
        mock_scaler._scale = MagicMock()
        mock_optimizer = MagicMock()
        mock_optimizer._parameter_list = []
        mock_to_tensor.return_value = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg.nranks = 8
        mock_hcg.get_data_parallel_world_size.return_value = 4
        mock_get_hcg.return_value = mock_hcg
        unscale_method(mock_scaler, mock_optimizer)

    @patch('paddle.distributed.all_reduce')
    @patch('paddle.to_tensor')
    @patch('paddle.distributed.fleet.get_hybrid_communicate_group')
    def test_unscale_method_hcg_single_dp(
        self, mock_get_hcg, mock_to_tensor, mock_all_reduce
    ):
        """测试 hcg 只有数据并行时的 unscale_method
        Test unscale_method with hcg having single dp"""
        from paddle.distributed.fleet.utils.mix_precision_utils import (
            unscale_method,
        )

        mock_scaler = MagicMock()
        mock_scaler._enable = True
        mock_scaler._scale = MagicMock()
        mock_optimizer = MagicMock()
        mock_optimizer._parameter_list = []
        mock_to_tensor.return_value = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg.nranks = 4
        mock_hcg.get_data_parallel_world_size.return_value = 4
        mock_get_hcg.return_value = mock_hcg
        unscale_method(mock_scaler, mock_optimizer)

    @patch('paddle.to_tensor')
    @patch(
        'paddle.distributed.fleet.get_hybrid_communicate_group',
        return_value=None,
    )
    def test_unscale_method_param_groups(self, mock_hcg, mock_to_tensor):
        """测试使用参数组的 unscale_method / Test unscale_method with param groups"""
        from paddle.distributed.fleet.utils.mix_precision_utils import (
            unscale_method,
        )

        mock_scaler = MagicMock()
        mock_scaler._enable = True
        mock_optimizer = MagicMock()
        mock_optimizer._param_groups = [{'params': []}]
        mock_to_tensor.return_value = MagicMock()
        unscale_method(mock_scaler, mock_optimizer)


class TestMixPrecisionScaler(unittest.TestCase):
    """测试 MixPrecisionScaler 类 / Test MixPrecisionScaler class"""

    def test_mix_precision_scaler_import(self):
        """测试 MixPrecisionScaler 可导入 / Test MixPrecisionScaler can be imported"""
        from paddle.distributed.fleet.utils.mix_precision_utils import (
            MixPrecisionScaler,
        )

        self.assertIsNotNone(MixPrecisionScaler)

    def test_mix_precision_scaler_init(self):
        """测试 MixPrecisionScaler 初始化 / Test MixPrecisionScaler initialization"""
        from paddle.distributed.fleet.utils.mix_precision_utils import (
            MixPrecisionScaler,
        )

        mock_scaler = MagicMock()
        scaler = MixPrecisionScaler(mock_scaler)
        self.assertEqual(scaler._inner_scaler, mock_scaler)

    def test_mix_precision_scaler_getattr(self):
        """测试 MixPrecisionScaler 属性代理 / Test MixPrecisionScaler attribute proxy"""
        from paddle.distributed.fleet.utils.mix_precision_utils import (
            MixPrecisionScaler,
        )

        mock_scaler = MagicMock()
        mock_scaler.scale_value = 65536.0
        scaler = MixPrecisionScaler(mock_scaler)
        self.assertEqual(scaler.scale_value, 65536.0)


if __name__ == '__main__':
    unittest.main()
