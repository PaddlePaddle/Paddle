# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import os
import tempfile
import unittest

import numpy as np

import paddle
from paddle import nn


class LinearNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(784, 10)

    def forward(self, x):
        return self._linear(x)


class TestSaveLoadSafetensors(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_safetensors(self):
        # enable dygraph mode
        paddle.disable_static()
        # create network
        layer = LinearNet()
        save_dict = layer.state_dict()

        path = os.path.join(
            self.temp_dir.name,
            "test_paddle_save_load_safetensors",
            "layer.safetensors",
        )

        paddle.save(save_dict, path, safetensors=True)
        numpy_load = paddle.load(path, return_numpy=True, safetensors=True)
        # compare results before and after saving
        for key, value in save_dict.items():
            self.assertTrue(isinstance(numpy_load[key], np.ndarray))
            np.testing.assert_array_equal(numpy_load[key], value)

        tensor_load = paddle.load(path, return_numpy=False, safetensors=True)
        # compare results before and after saving
        for key, value in save_dict.items():
            self.assertTrue(isinstance(tensor_load[key], paddle.Tensor))
            np.testing.assert_array_equal(tensor_load[key].numpy(), value)
