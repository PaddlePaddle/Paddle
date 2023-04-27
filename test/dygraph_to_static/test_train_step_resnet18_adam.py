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

import platform
import unittest

from test_train_step import (
    TestTrainStepTinyModel,
    loss_fn_tiny_model,
    train_step_tiny_model,
)

import paddle
from paddle.vision.models import resnet18


class TestTrainStepResNet18Adam(TestTrainStepTinyModel):
    def setUp(self):
        self.input = paddle.randn([64, 3, 224, 224])
        self.net_creator = resnet18
        self.lr_creator = lambda: 0.001
        self.optimizer_creator = paddle.optimizer.Adam
        self.loss_fn = loss_fn_tiny_model
        self.train_step_func = train_step_tiny_model
        self.steps = 3
        self.rtol = 1e-4
        if platform.system() == 'Windows':
            self.rtol = 1e-3


if __name__ == "__main__":
    unittest.main()
