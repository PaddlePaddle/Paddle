# copyright (c) 2020 paddlepaddle authors. all rights reserved.
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
import shutil
import tempfile
import unittest

import numpy as np

import paddle
from paddle import Model, fluid
from paddle.io import Dataset
from paddle.nn import Conv2D, Linear, ReLU, Sequential
from paddle.nn.layer.loss import CrossEntropyLoss
from paddle.static import InputSpec
from paddle.vision.datasets import MNIST
from paddle.vision.models import LeNet


class LeNetDygraph(paddle.nn.Layer):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.features = Sequential(
            Conv2D(1, 6, 3, stride=1, padding=1),
            ReLU(),
            paddle.nn.MaxPool2D(2, 2),
            Conv2D(6, 16, 5, stride=1, padding=0),
            ReLU(),
            paddle.nn.MaxPool2D(2, 2),
        )

        if num_classes > 0:
            self.fc = Sequential(
                Linear(400, 120), Linear(120, 84), Linear(84, 10)
            )

    def forward(self, inputs):
        x = self.features(inputs)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1, -1)
            x = self.fc(x)
        return x


class ModelInner(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc = paddle.nn.Linear(3, 4)

    def forward(self, x):
        y = self.fc(x)
        return y, 0


class ModelOutter(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.module1 = ModelInner()
        self.module2 = paddle.nn.Linear(4, 5)

    def forward(self, x):
        y, dummpy = self.module1(x)
        y = self.module2(y)
        return y, 3


class LeNetListInput(paddle.nn.Layer):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.cov = Conv2D(1, 6, 3, stride=1, padding=1)
        for param in self.cov.parameters():
            param.trainable = False
        self.features = Sequential(
            self.cov,
            ReLU(),
            paddle.nn.MaxPool2D(2, 2),
            Conv2D(6, 16, 5, stride=1, padding=0),
            ReLU(),
            paddle.nn.MaxPool2D(2, 2),
        )

        if num_classes > 0:
            self.fc = Sequential(
                Linear(400, 120), Linear(120, 84), Linear(84, 10)
            )

    def forward(self, inputs):
        x = inputs[0]
        x = self.features(x)

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x + inputs[1])
        return x


class LeNetDictInput(LeNetDygraph):
    def forward(self, inputs):
        x = self.features(inputs['x1'])

        if self.num_classes > 0:
            x = paddle.flatten(x, 1)
            x = self.fc(x + inputs['x2'])
        return x


class MnistDataset(MNIST):
    def __init__(self, mode, return_label=True, sample_num=None):
        super().__init__(mode=mode)
        self.return_label = return_label
        if sample_num:
            self.images = self.images[:sample_num]
            self.labels = self.labels[:sample_num]

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        img = np.reshape(img, [1, 28, 28])
        if self.return_label:
            return img, np.array(self.labels[idx]).astype('int64')
        return (img,)

    def __len__(self):
        return len(self.images)


def compute_acc(pred, label):
    pred = np.argmax(pred, -1)
    label = np.array(label)
    correct = pred[:, np.newaxis] == label
    return np.sum(correct) / correct.shape[0]


def dynamic_train(model, dataloader):
    optim = fluid.optimizer.Adam(
        learning_rate=0.001, parameter_list=model.parameters()
    )
    model.train()
    for inputs, labels in dataloader:
        outputs = model(inputs)
        loss = CrossEntropyLoss(reduction="sum")(outputs, labels)
        avg_loss = paddle.sum(loss)
        avg_loss.backward()
        optim.minimize(avg_loss)
        model.clear_gradients()


def dynamic_evaluate(model, dataloader):
    with fluid.dygraph.no_grad():
        model.eval()
        cnt = 0
        for inputs, labels in dataloader:
            outputs = model(inputs)

            cnt += (
                (
                    np.argmax(outputs.numpy(), -1)[:, np.newaxis]
                    == labels.numpy()
                )
                .astype('int')
                .sum()
            )

    return cnt / len(dataloader.dataset)


class MyModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self._fc = Linear(20, 10)

    def forward(self, x):
        y = self._fc(x)
        return y


class MyDataset(Dataset):
    def __getitem__(self, idx):
        return np.random.random(size=(20,)).astype(
            np.float32
        ), np.random.randint(0, 10, size=(1,)).astype(np.int64)

    def __len__(self):
        return 40


import sys
import time


class TestModelFunction(unittest.TestCase):
    def set_seed(self, seed=1024):
        paddle.seed(seed)
        paddle.framework.random._manual_program_seed(seed)

    def test_dygraph_export_deploy_model_about_inputs(self):
        start = time.time()
        self.set_seed()
        np.random.seed(201)
        mnist_data = MnistDataset(mode='train')
        paddle.disable_static()
        # without inputs
        save_dir = os.path.join(
            tempfile.mkdtemp(), '.cache_test_dygraph_export_deploy'
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for initial in ["fit"]:
            net = LeNet()
            model = Model(net)
            optim = fluid.optimizer.Adam(
                learning_rate=0.001, parameter_list=model.parameters()
            )
            model.prepare(
                optimizer=optim, loss=CrossEntropyLoss(reduction="sum")
            )
            print(
                "DEBUG fit DEBUG test_dygraph_export_deploy_model_about_inputs begin fit",
                time.time() - start,
            )
            sys.stdout.flush()
            model.fit(mnist_data, batch_size=64, verbose=0)

            model.save(save_dir, training=False)
        shutil.rmtree(save_dir)
        # with inputs, and the type of inputs is InputSpec
        save_dir = os.path.join(
            tempfile.mkdtemp(), '.cache_test_dygraph_export_deploy_2'
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        net = LeNet()
        inputs = InputSpec([None, 1, 28, 28], 'float32', 'x')
        model = Model(net, inputs)
        optim = fluid.optimizer.Adam(
            learning_rate=0.001, parameter_list=model.parameters()
        )
        model.prepare(optimizer=optim, loss=CrossEntropyLoss(reduction="sum"))
        model.save(save_dir, training=False)
        shutil.rmtree(save_dir)
        print(
            "DEBUG test_dygraph_export_deploy_model_about_inputs end",
            time.time() - start,
        )
        sys.stdout.flush()


if __name__ == '__main__':
    unittest.main()
