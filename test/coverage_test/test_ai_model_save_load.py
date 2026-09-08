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

"""
模型保存加载高级测试 / Advanced Model Save/Load Tests

测试目标 / Test Target:
  paddle 模型保存和加载功能

覆盖的模块 / Covered Modules:
  - paddle.save/load: 张量和字典保存
  - paddle.jit.save/load: JIT模型保存
  - model.state_dict/set_state_dict: 模型状态
  - paddle.Model.save/load: 高级模型API

作用 / Purpose:
  补充模型持久化API的测试，提升覆盖率。
"""

import os
import tempfile
import unittest

import numpy as np

import paddle
from paddle import nn

paddle.disable_static()


class SimpleModel(nn.Layer):
    """简单测试模型 / Simple test model"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = paddle.nn.functional.relu(self.fc1(x))
        return self.fc2(x)


class TestModelStatDict(unittest.TestCase):
    """测试模型状态字典 / Test model state dict"""

    def test_state_dict_keys(self):
        """测试状态字典键 / Test state dict keys"""
        model = SimpleModel()
        state_dict = model.state_dict()
        self.assertIn('fc1.weight', state_dict)
        self.assertIn('fc1.bias', state_dict)
        self.assertIn('fc2.weight', state_dict)
        self.assertIn('fc2.bias', state_dict)

    def test_state_dict_shapes(self):
        """测试状态字典形状 / Test state dict shapes"""
        model = SimpleModel()
        state_dict = model.state_dict()
        self.assertEqual(list(state_dict['fc1.weight'].shape), [4, 8])
        self.assertEqual(list(state_dict['fc1.bias'].shape), [8])

    def test_set_state_dict(self):
        """测试设置状态字典 / Test set state dict"""
        model1 = SimpleModel()
        model2 = SimpleModel()
        # Copy weights from model1 to model2
        state_dict = model1.state_dict()
        model2.set_state_dict(state_dict)
        # Verify weights are same
        for key in state_dict:
            np.testing.assert_allclose(
                model1.state_dict()[key].numpy(),
                model2.state_dict()[key].numpy(),
            )


class TestSaveLoad(unittest.TestCase):
    """测试保存加载 / Test save and load"""

    def test_save_load_tensor(self):
        """测试张量保存加载 / Test tensor save and load"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'tensor.pd')
            x = paddle.randn([3, 4])
            paddle.save(x, path)
            loaded = paddle.load(path)
            np.testing.assert_allclose(x.numpy(), loaded.numpy())

    def test_save_load_dict(self):
        """测试字典保存加载 / Test dict save and load"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'data.pd')
            data = {'weights': paddle.randn([4, 8]), 'bias': paddle.zeros([8])}
            paddle.save(data, path)
            loaded = paddle.load(path)
            np.testing.assert_allclose(
                data['weights'].numpy(), loaded['weights'].numpy()
            )
            np.testing.assert_allclose(
                data['bias'].numpy(), loaded['bias'].numpy()
            )

    def test_save_load_model_weights(self):
        """测试模型权重保存加载 / Test model weights save and load"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pd')
            model = SimpleModel()
            original_weights = {
                k: v.numpy().copy() for k, v in model.state_dict().items()
            }
            paddle.save(model.state_dict(), path)

            new_model = SimpleModel()
            new_model.set_state_dict(paddle.load(path))
            for key in original_weights:
                np.testing.assert_allclose(
                    original_weights[key], new_model.state_dict()[key].numpy()
                )


class TestJITSaveLoad(unittest.TestCase):
    """测试JIT保存加载 / Test JIT save and load"""

    def test_jit_save_load(self):
        """测试JIT模型保存加载 / Test JIT model save and load"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            x = paddle.randn([2, 4])

            # Save with JIT
            save_path = os.path.join(tmpdir, 'model')
            net = paddle.jit.to_static(
                model,
                input_spec=[
                    paddle.static.InputSpec(shape=[None, 4], dtype='float32')
                ],
            )
            paddle.jit.save(net, save_path)

            # Load and run
            loaded_model = paddle.jit.load(save_path)
            result = loaded_model(x)
            self.assertEqual(result.shape, [2, 2])

    def test_jit_save_preserves_output(self):
        """测试JIT保存保留输出 / Test JIT save preserves output"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SimpleModel()
            model.eval()
            x = paddle.randn([3, 4])
            original_output = model(x)

            save_path = os.path.join(tmpdir, 'model')
            net = paddle.jit.to_static(
                model,
                input_spec=[
                    paddle.static.InputSpec(shape=[None, 4], dtype='float32')
                ],
            )
            paddle.jit.save(net, save_path)

            loaded_model = paddle.jit.load(save_path)
            loaded_output = loaded_model(x)
            np.testing.assert_allclose(
                original_output.numpy(), loaded_output.numpy(), rtol=1e-5
            )


if __name__ == '__main__':
    unittest.main()
