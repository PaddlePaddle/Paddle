# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
import warnings

import numpy as np

import paddle
from paddle import base, nn, optimizer, static
from paddle.distributed.auto_parallel.static.auto_align_tool import (
    AutoAlignTool,
)
from paddle.vision.datasets import MNIST

warnings.filterwarnings("ignore")
paddle.enable_static()
paddle.set_device("gpu")

startup_program = base.default_startup_program()
main_program = base.default_main_program()


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


dataset = MnistDataset("train")
place = paddle.CUDAPlace(0)
with base.program_guard(main_program, startup_program):
    inputs = static.data(name="image", shape=[-1, 1, 28, 28], dtype="float32")
    labels = static.data(name="label", shape=[-1, 1], dtype="int64")
    z = nn.Conv2D(1, 6, 3, 1, 1).forward(inputs)
    z = nn.ReLU().forward(x=z)
    z = nn.MaxPool2D(2, 2).forward(x=z)
    z = nn.Conv2D(6, 16, 5, 1, 0).forward(x=z)
    z = nn.ReLU().forward(x=z)
    z = nn.MaxPool2D(2, 2).forward(x=z)
    z = nn.Flatten().forward(z)
    z = static.nn.fc(name="fc1", x=z, size=120)
    z = static.nn.fc(name="fc2", x=z, size=84)
    z = static.nn.fc(name="fc3", x=z, size=10)
    losses = nn.CrossEntropyLoss()(z, labels)

    optim = optimizer.SGD(0.001)
    optim.minimize(losses)


class TestAlignTool(unittest.TestCase):
    def test_align_tool(self):
        executor = base.Executor()
        executor.run(startup_program)
        align_tool = AutoAlignTool(main_program, 1, [losses.name])

        for epoch in range(5):
            images = np.zeros([32, 1, 28, 28], np.float32)
            labels = np.zeros([32, 1], np.int64)
            for i, data in enumerate(dataset):
                images[i % 32] = data[0]
                labels[i % 32] = data[1]
                if i % 31 == 0 and i > 0:
                    fetch_list = align_tool.get_var(0, 1)
                    fetch_list = align_tool.get_var(1, 1)
                    fetch_list = align_tool.get_var(2, 1)
                    fetch_list = align_tool.get_var(3, 1)
                    fetch_list = align_tool.get_var(4, 1)
                    fetch_list = align_tool.get_var(5, 1)
                    vars = executor.run(
                        main_program,
                        feed={"image": images, "label": labels},
                        fetch_list=fetch_list,
                    )
                    if os.path.exists("./serial") is False:
                        os.mkdir("./serial")
                    align_tool.save("./serial", vars, fetch_list)
                    break
            AutoAlignTool.diff_information("./serial", "./serial")
            AutoAlignTool.diff_information_from_dirs(["./serial"], ["./serial"])
            break

        print("test auto parallel align tool successfully!")


if __name__ == "__main__":
    unittest.main()
