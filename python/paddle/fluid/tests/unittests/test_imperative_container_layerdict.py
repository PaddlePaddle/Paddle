# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from collections import OrderedDict

import paddle


class TestLayerDict(unittest.TestCase):
    def test_layer_dict(self):
        layers = OrderedDict(
            [
                ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
                ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
            ]
        )

        layers_dicts = paddle.nn.LayerDict(sublayers=layers)

        def check_layer_dict():
            self.assertEqual(len(layers), len(layers_dicts))

            for k1, k2 in zip(layers, layers_dicts):
                self.assertIs(layers[k1], layers_dicts[k2])

            for k, v in zip(layers, layers_dicts.children()):
                self.assertIs(layers[k], v)

            for k in layers_dicts:
                self.assertIs(layers[k], layers_dicts[k])

            for k in layers.keys():
                self.assertTrue(k in layers_dicts)

            for k1, k2 in zip(layers.keys(), layers_dicts.keys()):
                self.assertEqual(k1, k2)

            for k, v in layers_dicts.items():
                self.assertIs(layers[k], v)

            for v1, v2 in zip(layers.values(), layers_dicts.values()):
                self.assertIs(v1, v2)

        check_layer_dict()

        layers['linear'] = paddle.nn.Linear(2, 4)
        layers_dicts['linear'] = layers['linear']
        check_layer_dict()

        sublayer = OrderedDict(
            [
                ('sigmod', paddle.nn.Sigmoid()),
                ('relu', paddle.nn.ReLU()),
            ]
        )
        layers.update(sublayer)
        layers_dicts.update(sublayer)
        check_layer_dict()

        del layers['conv1d']
        del layers_dicts['conv1d']
        check_layer_dict()

        l = layers_dicts.pop('linear')
        self.assertIs(layers['linear'], l)
        layers.pop('linear')
        check_layer_dict()

        layers_dicts.clear()
        self.assertEqual(0, len(layers_dicts))
        layers.clear()
        check_layer_dict()

        list_format_layers = [
            ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
            ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
        ]
        layers = OrderedDict(list_format_layers)
        layers_dicts.update(list_format_layers)
        check_layer_dict()

    def test_layer_dict_error_inputs(self):
        layers = [
            ('conv1d', paddle.nn.Conv1D(3, 2, 3), "conv1d"),
            ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
        ]

        layers_dicts = paddle.nn.LayerDict()
        self.assertRaises(ValueError, layers_dicts.update, layers)

        self.assertRaises(AssertionError, layers_dicts.update, 1)


if __name__ == '__main__':
    unittest.main()
