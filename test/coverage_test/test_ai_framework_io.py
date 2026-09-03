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
框架IO与模型保存加载单元测试 / Framework IO and Model Save/Load Unit Tests

测试目标 / Test Target:
  paddle.framework.io 模块 (python/paddle/framework/io.py, 覆盖率约67.3%)

覆盖的模块 / Covered Modules:
  - paddle.save: 保存张量/模型
  - paddle.load: 加载张量/模型
  - paddle.jit.save / paddle.jit.load: 动转静模型保存加载
  - paddle.nn.Layer state_dict: 模型状态字典操作

作用 / Purpose:
  覆盖模型和张量的序列化/反序列化代码路径，补充模型保存加载功能的测试。
"""

import os
import tempfile
import unittest

import numpy as np

import paddle
from paddle import nn

paddle.disable_static()


class TestPaddleSaveLoad(unittest.TestCase):
    """测试paddle.save和paddle.load / Test paddle.save and paddle.load"""

    def setUp(self):
        """初始化临时目录 / Initialize temp directory"""
        self.tmp_dir = tempfile.mkdtemp()

    def test_save_load_tensor(self):
        """测试张量保存和加载 / Test tensor save and load"""
        x = paddle.randn([3, 4])
        path = os.path.join(self.tmp_dir, 'tensor.pdparams')
        paddle.save(x, path)
        loaded = paddle.load(path)
        np.testing.assert_allclose(x.numpy(), loaded.numpy(), rtol=1e-5)

    def test_save_load_dict(self):
        """测试字典保存和加载 / Test dict save and load"""
        data = {
            'tensor1': paddle.randn([2, 3]),
            'tensor2': paddle.randn([4]),
            'scalar': paddle.to_tensor(1.0),
        }
        path = os.path.join(self.tmp_dir, 'dict.pdparams')
        paddle.save(data, path)
        loaded = paddle.load(path)
        for k in data:
            np.testing.assert_allclose(
                data[k].numpy(), loaded[k].numpy(), rtol=1e-5
            )

    def test_save_load_list(self):
        """测试列表保存和加载 / Test list save and load"""
        data = [paddle.randn([2, 3]), paddle.randn([4])]
        path = os.path.join(self.tmp_dir, 'list.pdparams')
        paddle.save(data, path)
        loaded = paddle.load(path)
        for orig, load in zip(data, loaded):
            np.testing.assert_allclose(orig.numpy(), load.numpy(), rtol=1e-5)

    def test_save_load_numpy(self):
        """测试numpy数组保存和加载 / Test numpy array save and load"""
        data = np.random.randn(3, 4).astype('float32')
        path = os.path.join(self.tmp_dir, 'numpy.pdparams')
        paddle.save(data, path)
        loaded = paddle.load(path)
        np.testing.assert_allclose(data, loaded, rtol=1e-5)


class TestModelStateDictIO(unittest.TestCase):
    """测试模型状态字典IO / Test model state dict IO"""

    def setUp(self):
        """初始化模型和临时目录 / Initialize model and temp dir"""
        self.tmp_dir = tempfile.mkdtemp()
        self.model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

    def test_state_dict_save_load(self):
        """测试状态字典保存和加载 / Test state dict save and load"""
        state_dict = self.model.state_dict()
        path = os.path.join(self.tmp_dir, 'model.pdparams')
        paddle.save(state_dict, path)

        # 创建新模型并加载参数
        new_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        loaded_state = paddle.load(path)
        new_model.set_state_dict(loaded_state)

        # 验证参数一致
        for (k1, v1), (k2, v2) in zip(
            self.model.state_dict().items(), new_model.state_dict().items()
        ):
            np.testing.assert_allclose(v1.numpy(), v2.numpy(), rtol=1e-5)

    def test_set_state_dict(self):
        """测试设置状态字典 / Test setting state dict"""
        original_state = self.model.state_dict()
        # 修改参数
        for key in original_state:
            original_state[key] = paddle.zeros_like(original_state[key])
        self.model.set_state_dict(original_state)
        # 验证参数已更新
        for key, param in self.model.state_dict().items():
            np.testing.assert_allclose(
                param.numpy(), np.zeros_like(param.numpy()), rtol=1e-5
            )

    def test_optimizer_state_dict(self):
        """测试优化器状态字典 / Test optimizer state dict"""
        optimizer = paddle.optimizer.Adam(
            learning_rate=0.01, parameters=self.model.parameters()
        )
        # 先执行一步
        x = paddle.randn([4, 10])
        output = self.model(x)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        # 保存优化器状态
        state_dict = optimizer.state_dict()
        path = os.path.join(self.tmp_dir, 'optimizer.pdopt')
        paddle.save(state_dict, path)
        loaded = paddle.load(path)
        self.assertIsNotNone(loaded)


class TestJITSaveLoad(unittest.TestCase):
    """测试JIT动转静保存加载 / Test JIT dynamic-to-static save and load"""

    def setUp(self):
        """初始化模型和临时目录 / Initialize model and temp dir"""
        self.tmp_dir = tempfile.mkdtemp()

    def test_jit_save_load(self):
        """测试JIT模型保存和加载 / Test JIT model save and load"""

        class SimpleModel(nn.Layer):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2)

            @paddle.jit.to_static(
                input_spec=[
                    paddle.static.InputSpec(shape=[None, 4], dtype='float32')
                ]
            )
            def forward(self, x):
                return self.fc(x)

        model = SimpleModel()
        path = os.path.join(self.tmp_dir, 'jit_model')
        paddle.jit.save(model, path)

        # 加载模型
        loaded_model = paddle.jit.load(path)
        x = paddle.randn([3, 4])
        output = loaded_model(x)
        self.assertEqual(output.shape, [3, 2])

    def test_jit_save_with_input_spec(self):
        """测试带InputSpec的JIT保存 / Test JIT save with InputSpec"""

        class LinearModel(nn.Layer):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)

            def forward(self, x):
                return self.linear(x)

        model = LinearModel()
        path = os.path.join(self.tmp_dir, 'linear_jit')
        # 使用input_spec保存
        input_spec = [paddle.static.InputSpec(shape=[None, 3], dtype='float32')]
        paddle.jit.save(model, path, input_spec=input_spec)

        loaded = paddle.jit.load(path)
        x = paddle.randn([5, 3])
        output = loaded(x)
        self.assertEqual(output.shape, [5, 2])


class TestInputSpec(unittest.TestCase):
    """测试InputSpec / Test InputSpec"""

    def test_input_spec_basic(self):
        """测试基本InputSpec / Test basic InputSpec"""
        spec = paddle.static.InputSpec(shape=[None, 4], dtype='float32')
        self.assertEqual(spec.shape, (-1, 4))
        self.assertEqual(spec.dtype, paddle.float32)

    def test_input_spec_with_name(self):
        """测试带名称的InputSpec / Test InputSpec with name"""
        spec = paddle.static.InputSpec(
            shape=[None, 3, 224, 224], dtype='float32', name='image'
        )
        self.assertEqual(spec.name, 'image')

    def test_to_static_decorator(self):
        """测试to_static装饰器 / Test to_static decorator"""

        @paddle.jit.to_static
        def simple_func(x):
            return x * 2 + 1

        x = paddle.randn([3, 4])
        result = simple_func(x)
        self.assertEqual(result.shape, [3, 4])


if __name__ == '__main__':
    unittest.main()
