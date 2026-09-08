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
模型训练循环测试 / Model Training Loop Tests

测试目标 / Test Target:
  完整训练循环组件

覆盖的模块 / Covered Modules:
  - paddle.Model 高级API
  - model.fit / model.evaluate / model.predict
  - callback机制

作用 / Purpose:
  补充高级模型训练API的测试，提升覆盖率。
"""

import unittest

import numpy as np

import paddle
from paddle import nn
from paddle.io import DataLoader, Dataset

paddle.disable_static()


class SimpleDataset(Dataset):
    def __init__(self, size=200):
        self.data = np.random.randn(size, 4).astype('float32')
        self.labels = np.random.randint(0, 2, size).astype('int64')

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)


class SimpleNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


class TestPaddleModelAPI(unittest.TestCase):
    """测试paddle.Model高级API / Test paddle.Model high-level API"""

    def setUp(self):
        """设置测试环境 / Setup test environment"""
        self.model = paddle.Model(SimpleNet())
        self.model.prepare(
            optimizer=paddle.optimizer.Adam(
                parameters=self.model.parameters(), learning_rate=0.001
            ),
            loss=nn.CrossEntropyLoss(),
            metrics=paddle.metric.Accuracy(),
        )
        self.train_dataset = SimpleDataset(100)
        self.val_dataset = SimpleDataset(40)

    def test_model_fit(self):
        """测试model.fit / Test model.fit"""
        self.model.fit(
            self.train_dataset,
            eval_data=self.val_dataset,
            batch_size=32,
            epochs=2,
            verbose=0,
        )

    def test_model_evaluate(self):
        """测试model.evaluate / Test model.evaluate"""
        result = self.model.evaluate(self.val_dataset, batch_size=16, verbose=0)
        self.assertIsNotNone(result)

    def test_model_predict(self):
        """测试model.predict / Test model.predict"""
        result = self.model.predict(self.val_dataset, batch_size=16, verbose=0)
        self.assertIsNotNone(result)


class TestManualTrainingLoop(unittest.TestCase):
    """测试手动训练循环 / Test manual training loop"""

    def test_basic_training(self):
        """测试基本训练循环 / Test basic training loop"""
        model = SimpleNet()
        optimizer = paddle.optimizer.Adam(parameters=model.parameters())
        criterion = nn.CrossEntropyLoss()
        dataset = SimpleDataset(100)
        loader = DataLoader(dataset, batch_size=32)

        model.train()
        for x, y in loader:
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

    def test_training_with_eval(self):
        """测试含评估的训练循环 / Test training loop with evaluation"""
        model = SimpleNet()
        optimizer = paddle.optimizer.Adam(parameters=model.parameters())
        criterion = nn.CrossEntropyLoss()
        train_set = SimpleDataset(80)
        val_set = SimpleDataset(20)
        train_loader = DataLoader(train_set, batch_size=32)
        val_loader = DataLoader(val_set, batch_size=20)

        # Train one epoch
        model.train()
        for x, y in train_loader:
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        # Evaluate
        model.eval()
        total_correct = 0
        total_samples = 0
        with paddle.no_grad():
            for x, y in val_loader:
                pred = model(x)
                correct = (pred.argmax(axis=1) == y).sum()
                total_correct += int(correct.numpy())
                total_samples += len(y)
        accuracy = total_correct / total_samples
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)


if __name__ == '__main__':
    unittest.main()
