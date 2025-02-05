#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import tempfile
import time
import unittest

import numpy as np

EPOCH_NUM = 1
BATCH_SIZE = 1024


def train_func_base(epoch_id, train_loader, model, cost, optimizer):
    total_step = len(train_loader)
    epoch_start = time.time()
    for batch_id, (images, labels) in enumerate(train_loader()):
        # forward
        outputs = model(images)
        loss = cost(outputs, labels)
        # backward and optimize
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        print(
            f"Epoch [{epoch_id + 1}/{EPOCH_NUM}], Step [{batch_id + 1}/{total_step}], Loss: {loss.numpy()}"
        )
    epoch_end = time.time()
    print(
        f"Epoch ID: {epoch_id + 1}, FP32 train epoch time: {(epoch_end - epoch_start) * 1000} ms"
    )


def train_func_ampo1(epoch_id, train_loader, model, cost, optimizer, scaler):
    import paddle

    total_step = len(train_loader)
    epoch_start = time.time()
    for batch_id, (images, labels) in enumerate(train_loader()):
        # forward
        with paddle.amp.auto_cast(
            custom_black_list={
                "flatten_contiguous_range",
                "greater_than",
                "matmul_v2",
            },
            level='O1',
        ):
            outputs = model(images)
            loss = cost(outputs, labels)
        # backward and optimize
        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.minimize(optimizer, scaled)
        optimizer.clear_grad()
        print(
            f"Epoch [{epoch_id + 1}/{EPOCH_NUM}], Step [{batch_id + 1}/{total_step}], Loss: {loss.numpy()}"
        )
    epoch_end = time.time()
    print(
        f"Epoch ID: {epoch_id + 1}, AMPO1 train epoch time: {(epoch_end - epoch_start) * 1000} ms"
    )


def test_func(epoch_id, test_loader, model, cost):
    import paddle

    # evaluation every epoch finish
    model.eval()
    avg_acc = [[], []]
    for batch_id, (images, labels) in enumerate(test_loader()):
        # forward
        outputs = model(images)
        loss = cost(outputs, labels)
        # accuracy
        acc_top1 = paddle.metric.accuracy(input=outputs, label=labels, k=1)
        acc_top5 = paddle.metric.accuracy(input=outputs, label=labels, k=5)
        avg_acc[0].append(acc_top1.numpy())
        avg_acc[1].append(acc_top5.numpy())
    model.train()
    print(
        f"Epoch ID: {epoch_id + 1}, Top1 accuracy: {np.array(avg_acc[0]).mean()}, Top5 accuracy: {np.array(avg_acc[1]).mean()}"
    )


class TestCustomCPUPlugin(unittest.TestCase):
    def setUp(self):
        # compile so and set to current path
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.temp_dir = tempfile.TemporaryDirectory()
        cmd = 'cd {} \
            && git clone --depth 1 {} \
            && cd PaddleCustomDevice \
            && git fetch origin \
            && git checkout {} -b dev \
            && cd backends/custom_cpu \
            && mkdir build && cd build && cmake .. -DPython_EXECUTABLE={} -DWITH_TESTING=OFF && make -j8'.format(
            self.temp_dir.name,
            os.getenv('PLUGIN_URL'),
            os.getenv('PLUGIN_TAG'),
            sys.executable,
        )
        os.system(cmd)

        # set environment for loading and registering compiled custom kernels
        # only valid in current process
        os.environ['CUSTOM_DEVICE_ROOT'] = os.path.join(
            cur_dir,
            f'{self.temp_dir.name}/PaddleCustomDevice/backends/custom_cpu/build',
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_custom_cpu_plugin(self):
        self._test_to_static()
        self._test_amp_o1()

    def _test_to_static(self):
        import paddle

        class LeNet5(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.fc = paddle.nn.Linear(in_features=1024, out_features=10)
                self.relu = paddle.nn.ReLU()
                self.fc1 = paddle.nn.Linear(in_features=10, out_features=10)

            def forward(self, x):
                out = paddle.flatten(x, 1)
                out = self.fc(out)
                out = self.relu(out)
                out = self.fc1(out)
                return out

        # set device
        paddle.set_device('custom_cpu')

        # model
        model = LeNet5()

        # cost and optimizer
        cost = paddle.nn.CrossEntropyLoss()
        optimizer = paddle.optimizer.Adam(
            learning_rate=0.001, parameters=model.parameters()
        )

        # convert to static model
        build_strategy = paddle.static.BuildStrategy()
        mnist = paddle.jit.to_static(
            model, build_strategy=build_strategy, full_graph=True
        )

        # data loader
        transform = paddle.vision.transforms.Compose(
            [
                paddle.vision.transforms.Resize((32, 32)),
                paddle.vision.transforms.ToTensor(),
                paddle.vision.transforms.Normalize(
                    mean=(0.1307,), std=(0.3081,)
                ),
            ]
        )
        train_dataset = paddle.vision.datasets.MNIST(
            mode='train', transform=transform, download=True
        )
        test_dataset = paddle.vision.datasets.MNIST(
            mode='test', transform=transform, download=True
        )
        train_loader = paddle.io.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=2,
        )
        test_loader = paddle.io.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=2,
        )

        # train and eval
        for epoch_id in range(EPOCH_NUM):
            train_func_base(epoch_id, train_loader, model, cost, optimizer)
            test_func(epoch_id, test_loader, model, cost)

    def _test_amp_o1(self):
        import paddle

        class LeNet5(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.fc = paddle.nn.Linear(in_features=1024, out_features=10)
                self.relu = paddle.nn.ReLU()
                self.fc1 = paddle.nn.Linear(in_features=10, out_features=10)

            def forward(self, x):
                out = paddle.flatten(x, 1)
                out = self.fc(out)
                out = self.relu(out)
                out = self.fc1(out)
                return out

        # set device
        paddle.set_device('custom_cpu')

        # model
        model = LeNet5()

        # cost and optimizer
        cost = paddle.nn.CrossEntropyLoss()
        optimizer = paddle.optimizer.Adam(
            learning_rate=0.001, parameters=model.parameters()
        )

        # convert to static model
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        model, optimizer = paddle.amp.decorate(
            models=model, optimizers=optimizer, level='O1'
        )

        # data loader
        transform = paddle.vision.transforms.Compose(
            [
                paddle.vision.transforms.Resize((32, 32)),
                paddle.vision.transforms.ToTensor(),
                paddle.vision.transforms.Normalize(
                    mean=(0.1307,), std=(0.3081,)
                ),
            ]
        )
        train_dataset = paddle.vision.datasets.MNIST(
            mode='train', transform=transform, download=True
        )
        test_dataset = paddle.vision.datasets.MNIST(
            mode='test', transform=transform, download=True
        )
        train_loader = paddle.io.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=2,
        )
        test_loader = paddle.io.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=2,
        )

        # train and eval
        for epoch_id in range(EPOCH_NUM):
            train_func_ampo1(
                epoch_id, train_loader, model, cost, optimizer, scaler
            )
            test_func(epoch_id, test_loader, model, cost)


if __name__ == '__main__':
    if os.name == 'nt' or sys.platform.startswith('darwin'):
        # only support Linux now
        sys.exit()
    unittest.main()
