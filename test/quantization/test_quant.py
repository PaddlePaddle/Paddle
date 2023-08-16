# copyright (c) 2022 paddlepaddle authors. all rights reserved.
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

import unittest

import paddle
import paddle.nn.functional as F
from paddle.nn import Conv2D, Linear, ReLU, Sequential
from paddle.quantization import QuantConfig
from paddle.quantization.base_quanter import BaseQuanter
from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver


class LeNetDygraph(paddle.nn.Layer):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.features = Sequential(
            Conv2D(3, 6, 3, stride=1, padding=1),
            ReLU(),
            paddle.nn.MaxPool2D(2, 2),
            Conv2D(6, 16, 5, stride=1, padding=0),
            ReLU(),
            paddle.nn.MaxPool2D(2, 2),
        )

        if num_classes > 0:
            self.fc = Sequential(
                Linear(576, 120), Linear(120, 84), Linear(84, 10)
            )

    def forward(self, inputs):
        x = self.features(inputs)
        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x)
        out = F.relu(x)
        out = F.relu(out)
        return out


class TestQuantConfig(unittest.TestCase):
    def setUp(self):
        self.model = LeNetDygraph()
        self.quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.9)

    def test_global_config(self):
        self.q_config = QuantConfig(
            activation=self.quanter, weight=self.quanter
        )
        self.q_config._specify(self.model)
        self.assertIsNotNone(self.q_config.global_config.activation)
        self.assertIsNotNone(self.q_config.global_config.weight)
        for layer in self.model.sublayers():
            config = self.q_config._get_config_by_layer(layer)
            self.assertTrue(config.activation == self.quanter)
            self.assertTrue(config.weight == self.quanter)

    def assert_just_linear_weight_configure(self, model, config):
        for layer in model.sublayers():
            layer_config = config._get_config_by_layer(layer)
            if type(layer) == Linear:
                self.assertIsNone(layer_config.activation)
                self.assertEqual(layer_config.weight, self.quanter)
                self.assertTrue(config._is_quantifiable(layer))
            elif type(layer) == Conv2D:
                self.assertIsNone(layer_config)
                self.assertFalse(config._is_quantifiable(layer))

    def test_add_layer_config(self):
        self.q_config = QuantConfig(activation=None, weight=None)
        self.q_config.add_layer_config(
            [self.model.fc], activation=None, weight=self.quanter
        )
        self.q_config._specify(self.model)
        self.assert_just_linear_weight_configure(self.model, self.q_config)

    def test_add_name_config(self):
        self.q_config = QuantConfig(activation=None, weight=None)
        self.q_config.add_name_config(
            [self.model.fc.full_name()], activation=None, weight=self.quanter
        )
        self.q_config._specify(self.model)
        self.assert_just_linear_weight_configure(self.model, self.q_config)

    def test_add_type_config(self):
        self.q_config = QuantConfig(activation=None, weight=None)
        self.q_config.add_type_config(
            [Linear], activation=None, weight=self.quanter
        )
        self.q_config._specify(self.model)
        self.assert_just_linear_weight_configure(self.model, self.q_config)

    def test_add_qat_layer_mapping(self):
        self.q_config = QuantConfig(activation=None, weight=None)
        self.q_config.add_qat_layer_mapping(Sequential, Conv2D)
        self.assertTrue(Sequential in self.q_config.qat_layer_mappings)
        self.assertTrue(
            Sequential not in self.q_config.default_qat_layer_mapping
        )

    def test_add_customized_leaf(self):
        self.q_config = QuantConfig(activation=None, weight=None)
        self.q_config.add_customized_leaf(Sequential)
        self.assertTrue(Sequential in self.q_config.customized_leaves)
        self.assertTrue(self.q_config._is_customized_leaf(self.model.fc))
        self.assertTrue(self.q_config._is_leaf(self.model.fc))
        self.assertFalse(self.q_config._is_default_leaf(self.model.fc))
        self.assertFalse(self.q_config._is_real_leaf(self.model.fc))

    def test_need_observe(self):
        self.q_config = QuantConfig(activation=None, weight=None)
        self.q_config.add_layer_config(
            [self.model.fc], activation=self.quanter, weight=self.quanter
        )
        self.q_config.add_customized_leaf(Sequential)
        self.q_config._specify(self.model)
        self.assertTrue(self.q_config._has_observer_config(self.model.fc))
        self.assertTrue(self.q_config._need_observe(self.model.fc))

    def test__get_observer(self):
        self.q_config = QuantConfig(activation=None, weight=None)
        self.q_config.add_layer_config(
            [self.model.fc], activation=self.quanter, weight=self.quanter
        )
        self.q_config._specify(self.model)
        observer = self.q_config._get_observer(self.model.fc)
        self.assertIsInstance(observer, BaseQuanter)

    def test_details(self):
        self.q_config = QuantConfig(
            activation=self.quanter, weight=self.quanter
        )
        self.q_config._specify(self.model)
        self.assertIsNotNone(self.q_config.details())
        self.assertIsNotNone(self.q_config.__str__())


if __name__ == '__main__':
    unittest.main()
