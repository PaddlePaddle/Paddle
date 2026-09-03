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

# Unit test for paddle.nn.layer containers (Sequential, LayerList, etc.)
# Target: cover Sequential, LayerList, LayerDict, ParameterList

import unittest

import paddle
from paddle import nn


class TestSequentialAdvanced(unittest.TestCase):
    """Test Sequential advanced patterns.
    Sequential accepts: Layer objects, (name, Layer) tuples, or OrderedDict.
    Does NOT accept keyword args like fc1=nn.Linear(...).
    """

    def setUp(self):
        paddle.disable_static()

    def test_sequential_ordered_dict(self):
        """Sequential with OrderedDict initialization."""
        from collections import OrderedDict

        layers = OrderedDict(
            [
                ('fc1', nn.Linear(10, 20)),
                ('relu1', nn.ReLU()),
                ('fc2', nn.Linear(20, 5)),
            ]
        )
        seq = nn.Sequential(layers)
        x = paddle.randn([4, 10])
        out = seq(x)
        self.assertEqual(out.shape, [4, 5])

    def test_sequential_named_tuples(self):
        """Sequential with (name, layer) tuples."""
        seq = nn.Sequential(
            ('fc1', nn.Linear(10, 20)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(20, 5)),
        )
        x = paddle.randn([4, 10])
        out = seq(x)
        self.assertEqual(out.shape, [4, 5])

    def test_sequential_positional_args(self):
        """Sequential with positional Layer arguments (unnamed)."""
        seq = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        x = paddle.randn([4, 10])
        out = seq(x)
        self.assertEqual(out.shape, [4, 5])

    def test_sequential_append(self):
        """Sequential append method."""
        seq = nn.Sequential(nn.Linear(10, 20))
        seq.append(nn.ReLU())
        seq.append(nn.Linear(20, 5))
        x = paddle.randn([4, 10])
        out = seq(x)
        self.assertEqual(out.shape, [4, 5])

    def test_sequential_len(self):
        """Sequential __len__."""
        seq = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        self.assertEqual(len(seq), 3)

    def test_sequential_indexing(self):
        """Sequential __getitem__ and __iter__."""
        seq = nn.Sequential(
            ('fc1', nn.Linear(10, 20)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(20, 5)),
        )
        self.assertIsInstance(seq[0], nn.Linear)
        self.assertIsInstance(seq['relu'], nn.ReLU)
        layers = list(seq)
        self.assertEqual(len(layers), 3)

    def test_sequential_insert(self):
        """Sequential insert method."""
        seq = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))
        seq.insert(1, nn.ReLU())
        self.assertEqual(len(seq), 3)

    def test_sequential_extend(self):
        """Sequential extend method."""
        seq = nn.Sequential(nn.Linear(10, 20))
        seq.extend([nn.ReLU(), nn.Linear(20, 5)])
        self.assertEqual(len(seq), 3)
        x = paddle.randn([4, 10])
        out = seq(x)
        self.assertEqual(out.shape, [4, 5])


class TestLayerList(unittest.TestCase):
    """Test LayerList."""

    def setUp(self):
        paddle.disable_static()

    def test_layer_list_basic(self):
        """LayerList basic usage."""
        layers = nn.LayerList(
            [
                nn.Linear(10, 20),
                nn.Linear(20, 5),
            ]
        )
        x = paddle.randn([4, 10])
        for layer in layers:
            x = layer(x)
        self.assertEqual(x.shape, [4, 5])

    def test_layer_list_append(self):
        """LayerList append."""
        layers = nn.LayerList()
        layers.append(nn.Linear(10, 20))
        layers.append(nn.Linear(20, 5))
        self.assertEqual(len(layers), 2)

    def test_layer_list_extend(self):
        """LayerList extend."""
        layers = nn.LayerList()
        layers.extend([nn.Linear(10, 20), nn.Linear(20, 5)])
        self.assertEqual(len(layers), 2)

    def test_layer_list_insert(self):
        """LayerList insert."""
        layers = nn.LayerList([nn.Linear(10, 20), nn.Linear(20, 5)])
        layers.insert(1, nn.ReLU())
        self.assertEqual(len(layers), 3)

    def test_layer_list_indexing(self):
        """LayerList indexing."""
        layers = nn.LayerList([nn.Linear(10, 20), nn.ReLU()])
        self.assertIsInstance(layers[0], nn.Linear)
        self.assertIsInstance(layers[1], nn.ReLU)


class TestLayerDict(unittest.TestCase):
    """Test LayerDict."""

    def setUp(self):
        paddle.disable_static()

    def test_layer_dict_basic(self):
        """LayerDict basic usage."""
        layers = nn.LayerDict(
            {
                'fc1': nn.Linear(10, 20),
                'fc2': nn.Linear(20, 5),
            }
        )
        self.assertIsInstance(layers['fc1'], nn.Linear)

    def test_layer_dict_keys(self):
        """LayerDict keys."""
        layers = nn.LayerDict(
            {
                'fc1': nn.Linear(10, 20),
                'relu': nn.ReLU(),
            }
        )
        keys = list(layers.keys())
        self.assertIn('fc1', keys)
        self.assertIn('relu', keys)


class TestParameterList(unittest.TestCase):
    """Test ParameterList."""

    def setUp(self):
        paddle.disable_static()

    def test_parameter_list_basic(self):
        """ParameterList basic usage."""
        params = nn.ParameterList(
            [
                paddle.create_parameter(shape=[10, 20], dtype='float32'),
                paddle.create_parameter(shape=[20, 5], dtype='float32'),
            ]
        )
        self.assertEqual(len(params), 2)
        self.assertEqual(params[0].shape, [10, 20])

    def test_parameter_list_append(self):
        """ParameterList append."""
        params = nn.ParameterList()
        params.append(paddle.create_parameter(shape=[10, 20], dtype='float32'))
        self.assertEqual(len(params), 1)


if __name__ == '__main__':
    unittest.main()
