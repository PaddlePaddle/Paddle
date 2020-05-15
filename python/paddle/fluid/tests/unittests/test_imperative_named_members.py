# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import paddle.fluid as fluid
import paddle


class MyLayer(fluid.Layer):
    def __init__(self, num_channel, dim, num_filter=5):
        super(MyLayer, self).__init__()
        self.fc = fluid.dygraph.Linear(dim, dim)
        self.conv = fluid.dygraph.Conv2D(num_channel, num_channel, num_filter)

    def forward(self, x):
        x = self.fc(x)
        x = self.conv(x)
        return x


class TestImperativeNamedSubLayers(unittest.TestCase):
    def test_named_sublayers(self):
        with fluid.dygraph.guard():
            fc1 = fluid.Linear(10, 3)
            fc2 = fluid.Linear(3, 10, bias_attr=False)
            custom = MyLayer(3, 10)
            model = fluid.dygraph.Sequential(fc1, fc2, custom)
            named_sublayers = model.named_sublayers()
            list_named_sublayers = list(named_sublayers)

            expected_sublayers = [fc1, fc2, custom, custom.fc, custom.conv]
            self.assertEqual(len(list_named_sublayers), len(expected_sublayers))
            for (name, sublayer), expected_sublayer in zip(list_named_sublayers,
                                                           expected_sublayers):
                self.assertEqual(sublayer, expected_sublayer)

            list_sublayers = list(model.sublayers())
            self.assertEqual(len(list_named_sublayers), len(list_sublayers))
            for (name, sublayer), expected_sublayer in zip(list_named_sublayers,
                                                           list_sublayers):
                self.assertEqual(sublayer, expected_sublayer)

            for name, sublayer in model.named_sublayers(
                    include_sublayers=False):
                self.assertEqual(model[name], sublayer)

            self.assertListEqual(
                [l for _, l in list(model.named_sublayers(include_self=True))],
                [model] + expected_sublayers)


class TestImperativeNamedParameters(unittest.TestCase):
    def test_named_parameters(self):
        with fluid.dygraph.guard():
            fc1 = fluid.Linear(10, 3)
            fc2 = fluid.Linear(3, 10, bias_attr=False)
            custom = MyLayer(3, 10)
            model = paddle.nn.Sequential(fc1, fc2, custom)

            named_parameters = list(model.named_parameters())
            expected_named_parameters = list()
            for prefix, layer in model.named_sublayers(include_sublayers=True):
                for name, param in layer.named_parameters(
                        include_sublayers=False):
                    full_name = prefix + ('.' if prefix else '') + name
                    expected_named_parameters.append((full_name, param))

            self.assertListEqual(expected_named_parameters, named_parameters)


if __name__ == '__main__':
    unittest.main()
