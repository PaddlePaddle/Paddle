# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import random
import unittest

import numpy as np

import paddle
import paddle.nn as nn
from paddle.distributed.fleet import auto
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.vision.datasets import MNIST

paddle.enable_static()


def apply_pass(use_bf16=False):
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True
    if use_bf16:
        amp = strategy.amp
        amp.enable = True
        amp.enable_bf16 = True
        amp.custom_bf16_list = {'relu', 'scale', 'elementwise_add', 'reshape2'}
        amp.custom_fp32_list = {'pool2d', 'reduce_mean', 'matmul_v2', 'conv2d'}
    return strategy


class MnistDataset(MNIST):
    def __init__(self, mode, return_label=True):
        super().__init__(mode=mode)
        self.return_label = return_label

    def __getitem__(self, idx):
        img = np.reshape(self.images[idx], [1, 28, 28])
        if self.return_label:
            return img, np.array(self.labels[idx]).astype('int64')
        return (img,)

    def __len__(self):
        return len(self.images)


def reset_prog():
    paddle.fluid.framework.switch_main_program(paddle.static.Program())
    paddle.fluid.framework.switch_startup_program(paddle.static.Program())


class Model(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(1, 6, 3, 1, 1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2D(2, 2)
        self.conv2 = nn.Conv2D(6, 16, 5, 1, 0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2D(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        input.stop_gradient = True
        x = self.maxpool1(self.relu1(self.conv1(input)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        return self.fc3(self.fc2(self.fc1(x)))


class TestBF16Pass(unittest.TestCase):
    def setUp(self):
        self.rtol = 1e-5
        self.atol = 1e-8
        self.batch_size = 256
        self.batch_num = 10
        self.dataset = MnistDataset("train")
        self.eval_dataset = MnistDataset("test")

    def init(self, engine):
        paddle.seed(2021)
        np.random.seed(2021)
        random.seed(2021)
        place = paddle.fluid.CUDAPlace(ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(self, use_bf16=False):
        reset_prog()

        strategy = apply_pass(use_bf16)
        model = Model()
        opt = paddle.optimizer.SGD(0.001, parameters=model.parameters())
        loss = nn.CrossEntropyLoss()
        engine = auto.Engine(model, loss, opt, strategy=strategy)
        self.init(engine)
        return engine

    def check_results(self, ref_losses, check_losses, rtol=None, atol=None):
        np.testing.assert_allclose(
            ref_losses,
            check_losses,
            rtol=rtol or self.rtol,
            atol=atol or self.atol,
            err_msg='pass {} has wrong results!, \nu={}\nv={}\ndiff={}'.format(
                __class__, ref_losses, check_losses, ref_losses - check_losses
            ),
        )

    def test_bf16_pass(self):
        fp32_engine = self.get_engine()
        history = fp32_engine.fit(
            self.dataset, 1, batch_size=self.batch_size, steps_per_epoch=10
        )
        fp32_losses = np.array(history.history["loss"])

        bf16_o1_engine = self.get_engine(False)
        history = bf16_o1_engine.fit(
            self.dataset, 1, batch_size=self.batch_size, steps_per_epoch=10
        )
        bf16_o1_losses = np.array(history.history["loss"])
        # bf16_o1_engine.evaluate(
        #     self.eval_dataset, 1, batch_size=self.batch_size
        # )
        # self.check_results(mp_losses, bf16_o1_losses)


if __name__ == "__main__":
    unittest.main()
