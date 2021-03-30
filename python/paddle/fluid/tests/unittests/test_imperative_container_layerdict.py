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

from __future__ import print_function

import unittest
import numpy as np
import paddle
from collections import OrderedDict


class TestLayerDict(unittest.TestCase):
    def test_layer_dict(self):
        layers = OrderedDict([
            ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
            ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
        ])

        laers_dicts = paddle.nn.LayerDict(sublayers=layers)

        self.assertEqual(len(layers), len(laers_dicts))


if __name__ == '__main__':
    unittest.main()
