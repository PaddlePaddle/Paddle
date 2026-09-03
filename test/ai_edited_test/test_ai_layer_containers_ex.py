# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# [AUTO-GENERATED] Unit test for paddle.nn.layer.layers, common, container
# 自动生成的单测，覆盖 paddle.nn.layer 中 Linear, Sequential, LayerList 等核心类
# Target: paddle/nn/layer/layers.py (and common.py, container.py)

"""
测试模块：paddle.nn.layer (Linear, Sequential, LayerList, Layer base)
Test Module: paddle.nn.layer

本测试覆盖以下功能：
This test covers the following functions:
1. Linear - 线性层 / Linear layer with different configurations
2. Sequential - 序列容器 / Sequential container with named sublayers
3. LayerList - 层列表 / LayerList with append, extend, insert
4. LayerDict - 层字典 / LayerDict with ordered access
5. Layer base - 基础层 / Layer class with eval/train, sublayers, add_sublayer
6. Identity - 恒等层 / Identity layer
"""

import unittest

import numpy as np

import paddle
from paddle import nn


class TestLinearComprehensive(unittest.TestCase):
    """测试Linear线性层
    Test Linear"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_linear_basic(self):
        """测试基本线性层 / Test basic Linear"""
        linear = nn.Linear(10, 5)
        x = paddle.randn([2, 10])
        out = linear(x)
        self.assertEqual(out.shape, [2, 5])

    def test_linear_no_bias(self):
        """测试无偏置 / Test Linear without bias"""
        linear = nn.Linear(10, 5, bias_attr=False)
        self.assertIsNone(linear.bias)
        x = paddle.randn([2, 10])
        out = linear(x)
        self.assertEqual(out.shape, [2, 5])

    def test_linear_2d_input(self):
        """测试2D输入 / Test with 2D input"""
        linear = nn.Linear(10, 5)
        x = paddle.randn([4, 10])
        out = linear(x)
        self.assertEqual(out.shape, [4, 5])

    def test_linear_3d_input(self):
        """测试3D输入 / Test with 3D input (batch, seq, features)"""
        linear = nn.Linear(10, 5)
        x = paddle.randn([2, 6, 10])
        out = linear(x)
        self.assertEqual(out.shape, [2, 6, 5])

    def test_linear_float64(self):
        """测试float64输入 / Test with float64 input"""
        linear = nn.Linear(10, 5)
        x = paddle.randn([2, 10], dtype='float32')
        out = linear(x)
        self.assertEqual(out.shape, [2, 5])

    def test_linear_weight_init(self):
        """测试权重初始化 / Test weight initialization"""
        linear = nn.Linear(10, 5)
        self.assertIsNotNone(linear.weight)
        self.assertIsNotNone(linear.bias)


class TestSequentialComprehensive(unittest.TestCase):
    """测试Sequential序列容器
    Test Sequential"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_sequential_list(self):
        """测试列表形式 / Test Sequential with list"""
        seq = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        x = paddle.randn([2, 10])
        out = seq(x)
        self.assertEqual(out.shape, [2, 5])

    def test_sequential_ordered_dict(self):
        """测试OrderedDict形式 / Test Sequential with OrderedDict"""
        from collections import OrderedDict

        layers = OrderedDict(
            [
                ('linear1', nn.Linear(10, 20)),
                ('relu', nn.ReLU()),
                ('linear2', nn.Linear(20, 5)),
            ]
        )
        seq = nn.Sequential(layers)
        x = paddle.randn([2, 10])
        out = seq(x)
        self.assertEqual(out.shape, [2, 5])

    def test_sequential_len(self):
        """测试长度 / Test len(Sequential)"""
        seq = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        self.assertEqual(len(seq), 2)

    def test_sequential_indexing(self):
        """测试索引 / Test indexing"""
        seq = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 3))
        self.assertIsInstance(seq[0], nn.Linear)
        self.assertIsInstance(seq[1], nn.ReLU)

    def test_sequential_iter(self):
        """测试迭代 / Test iteration"""
        seq = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        layers = list(seq)
        self.assertEqual(len(layers), 2)

    def test_sequential_repr(self):
        """测试字符串表示 / Test __repr__"""
        seq = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        r = str(seq)
        self.assertIn('Linear', r)

    def test_sequential_empty(self):
        """测试空Sequential / Test empty Sequential"""
        seq = nn.Sequential()
        self.assertEqual(len(seq), 0)

    def test_sequential_add_module(self):
        """测试动态添加模块 / Test add_module via indexing"""
        seq = nn.Sequential()
        seq.add_sublayer('linear', nn.Linear(10, 5))
        x = paddle.randn([2, 10])
        out = seq(x)
        self.assertEqual(out.shape, [2, 5])


class TestLayerListComprehensive(unittest.TestCase):
    """测试LayerList层列表
    Test LayerList"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_layer_list_basic(self):
        """测试基本LayerList / Test basic LayerList"""
        layer_list = nn.LayerList([nn.Linear(10, 5), nn.Linear(5, 3)])
        self.assertEqual(len(layer_list), 2)

    def test_layer_list_append(self):
        """测试append / Test append"""
        layer_list = nn.LayerList([nn.Linear(10, 5)])
        layer_list.append(nn.ReLU())
        self.assertEqual(len(layer_list), 2)

    def test_layer_list_extend(self):
        """测试extend / Test extend"""
        layer_list = nn.LayerList([nn.Linear(10, 5)])
        layer_list.extend([nn.ReLU(), nn.Linear(5, 3)])
        self.assertEqual(len(layer_list), 3)

    def test_layer_list_insert(self):
        """测试insert / Test insert"""
        layer_list = nn.LayerList([nn.Linear(10, 5), nn.Linear(5, 3)])
        layer_list.insert(1, nn.ReLU())
        self.assertEqual(len(layer_list), 3)
        self.assertIsInstance(layer_list[1], nn.ReLU)

    def test_layer_list_indexing(self):
        """测试索引 / Test indexing"""
        layer_list = nn.LayerList([nn.Linear(10, 5), nn.ReLU()])
        self.assertIsInstance(layer_list[0], nn.Linear)
        self.assertIsInstance(layer_list[1], nn.ReLU)

    def test_layer_list_iter(self):
        """测试迭代 / Test iteration"""
        layer_list = nn.LayerList([nn.Linear(10, 5), nn.ReLU()])
        layers = list(layer_list)
        self.assertEqual(len(layers), 2)

    def test_layer_list_forward(self):
        """测试forward / Test forward by iterating"""
        layer_list = nn.LayerList([nn.Linear(10, 5), nn.Linear(5, 3)])
        x = paddle.randn([2, 10])
        for layer in layer_list:
            x = layer(x)
        self.assertEqual(x.shape, [2, 3])


class TestLayerDictComprehensive(unittest.TestCase):
    """测试LayerDict层字典
    Test LayerDict"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_layer_dict_basic(self):
        """测试基本LayerDict / Test basic LayerDict"""
        from collections import OrderedDict

        layers = OrderedDict(
            [('linear1', nn.Linear(10, 5)), ('linear2', nn.Linear(5, 3))]
        )
        layer_dict = nn.LayerDict(layers)
        self.assertEqual(len(layer_dict), 2)
        self.assertIn('linear1', layer_dict)

    def test_layer_dict_keys(self):
        """测试keys / Test keys"""
        from collections import OrderedDict

        layers = OrderedDict([('a', nn.Linear(10, 5)), ('b', nn.ReLU())])
        layer_dict = nn.LayerDict(layers)
        keys = list(layer_dict.keys())
        self.assertEqual(keys, ['a', 'b'])

    def test_layer_dict_update(self):
        """测试update / Test update"""
        layer_dict = nn.LayerDict({'a': nn.Linear(10, 5)})
        layer_dict.update({'b': nn.ReLU()})
        self.assertEqual(len(layer_dict), 2)

    def test_layer_dict_pop(self):
        """测试pop / Test pop"""
        layer_dict = nn.LayerDict({'a': nn.Linear(10, 5), 'b': nn.ReLU()})
        removed = layer_dict.pop('a')
        self.assertIsInstance(removed, nn.Linear)
        self.assertEqual(len(layer_dict), 1)


class TestIdentityLayer(unittest.TestCase):
    """测试Identity恒等层
    Test Identity"""

    def setUp(self):
        paddle.disable_static()

    def tearDown(self):
        paddle.enable_static()

    def test_identity_basic(self):
        """测试Identity / Test Identity"""
        identity = nn.Identity()
        x = paddle.randn([2, 3, 4])
        out = identity(x)
        self.assertTrue(paddle.allclose(x, out))


class TestLayerBase(unittest.TestCase):
    """测试Layer基类
    Test Layer base class"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_layer_train_eval(self):
        """测试train/eval模式切换 / Test train/eval mode switch"""
        layer = nn.Linear(10, 5)
        self.assertTrue(layer.training)
        layer.eval()
        self.assertFalse(layer.training)
        layer.train()
        self.assertTrue(layer.training)

    def test_layer_sublayers(self):
        """测试sublayers / Test sublayers"""
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 3))
        children = list(model.sublayers())
        self.assertTrue(len(children) >= 3)

    def test_layer_named_sublayers(self):
        """测试named_sublayers / Test named_sublayers (named_children)"""
        from collections import OrderedDict

        layers = OrderedDict(
            [('l1', nn.Linear(10, 5)), ('l2', nn.Linear(5, 3))]
        )
        model = nn.Sequential(layers)
        names = [name for name, _ in model.named_children()]
        self.assertEqual(names, ['l1', 'l2'])

    def test_layer_parameters(self):
        """测试parameters / Test parameters"""
        layer = nn.Linear(10, 5)
        params = list(layer.parameters())
        self.assertEqual(len(params), 2)

    def test_layer_state_dict(self):
        """测试state_dict / Test state_dict"""
        layer = nn.Linear(10, 5)
        sd = layer.state_dict()
        self.assertIn('weight', sd)
        self.assertIn('bias', sd)

    def test_layer_to_static_dict(self):
        """测试load_state_dict / Test load_state_dict"""
        layer1 = nn.Linear(10, 5)
        layer2 = nn.Linear(10, 5)
        sd = layer1.state_dict()
        layer2.load_state_dict(sd)
        # Weights should match
        np.testing.assert_allclose(
            layer1.weight.numpy(), layer2.weight.numpy(), atol=1e-6
        )

    def test_layer_full_name(self):
        """测试full_name / Test _full_name"""
        layer = nn.Linear(10, 5)
        self.assertIsNotNone(layer._full_name)

    def test_layer_dtype(self):
        """测试dtype / Test default dtype"""
        layer = nn.Linear(10, 5)
        # _dtype should be 'float32' by default
        self.assertEqual(layer._dtype, 'float32')

    def test_layer_repr(self):
        """测试__repr__ / Test __repr__"""
        layer = nn.Linear(10, 5)
        r = str(layer)
        self.assertIn('Linear', r)

    def test_layer_add_buffer(self):
        """测试register_buffer / Test register_buffer"""
        layer = nn.Linear(10, 5)
        buffer = paddle.randn([5])
        layer.register_buffer('my_buffer', buffer)
        self.assertEqual(layer.my_buffer.shape, [5])


class TestIncompatibleKeys(unittest.TestCase):
    """测试_IncompatibleKeys
    Test _IncompatibleKeys"""

    def test_repr_all_matched(self):
        """测试所有key匹配 / Test repr when all keys matched"""
        from paddle.nn.layer.layers import _IncompatibleKeys

        result = _IncompatibleKeys([], [])
        self.assertEqual(str(result), '<All keys matched successfully>')

    def test_repr_missing_keys(self):
        """测试缺失key / Test repr with missing keys"""
        from paddle.nn.layer.layers import _IncompatibleKeys

        result = _IncompatibleKeys(['weight'], [])
        r = str(result)
        self.assertIn('weight', r)


if __name__ == '__main__':
    unittest.main()
