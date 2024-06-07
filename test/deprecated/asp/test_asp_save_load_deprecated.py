# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
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

import unittest

import numpy as np

import paddle
from paddle import base
from paddle.base import core
from paddle.incubate.asp import ASPHelper


class MyLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = paddle.nn.Conv2D(
            in_channels=3, out_channels=4, kernel_size=3, padding=2
        )
        self.linear1 = paddle.nn.Linear(4624, 32)
        self.linear2 = paddle.nn.Linear(32, 32)
        self.linear3 = paddle.nn.Linear(32, 10)

    def forward(self, img):
        hidden = self.conv1(img)
        hidden = paddle.flatten(hidden, start_axis=1)
        hidden = self.linear1(hidden)
        hidden = self.linear2(hidden)
        prediction = self.linear3(hidden)
        return prediction


class TestASPStaticOptimize(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()

        self.main_program = base.Program()
        self.startup_program = base.Program()

        def build_model():
            img = paddle.static.data(
                name='img', shape=[None, 3, 32, 32], dtype='float32'
            )
            label = paddle.static.data(
                name='label', shape=[None, 1], dtype='int64'
            )
            hidden = paddle.static.nn.conv2d(
                input=img, num_filters=4, filter_size=3, padding=2, act="relu"
            )
            hidden = paddle.static.nn.fc(x=hidden, size=32, activation='relu')
            prediction = paddle.static.nn.fc(
                x=hidden, size=10, activation='softmax'
            )
            return img, label, prediction

        with base.program_guard(self.main_program, self.startup_program):
            self.img, self.label, predict = build_model()
            self.loss = paddle.mean(
                paddle.nn.functional.cross_entropy(
                    input=predict,
                    label=self.label,
                    reduction='none',
                    use_softmax=False,
                )
            )
            self.optimizer = paddle.optimizer.SGD(learning_rate=0.01)
            self.optimizer = paddle.incubate.asp.decorate(self.optimizer)
            self.optimizer.minimize(self.loss, self.startup_program)

        self.place = paddle.CPUPlace()
        if core.is_compiled_with_cuda():
            self.place = paddle.CUDAPlace(0)
        self.exe = base.Executor(self.place)
        self.exe.run(self.startup_program)

        paddle.incubate.asp.prune_model(self.main_program)

    def test_save_and_load(self):
        path = "/tmp/paddle_asp_save_st/"
        param_path = path + "asp.pdparams"
        model_path = path + "asp.pdmodel"

        paddle.save(self.main_program.state_dict(), param_path)
        paddle.save(self.main_program, model_path)

        prog = paddle.load(model_path)

        state_dict = paddle.load(param_path)
        prog.set_state_dict(state_dict)

        feeder = base.DataFeeder(
            feed_list=[self.img, self.label], place=self.place
        )

        data = (
            np.random.randn(64, 3, 32, 32),
            np.random.randint(10, size=(64, 1)),
        )
        self.exe.run(prog, feed=feeder.feed([data]))

        for param in prog.global_block().all_parameters():
            if ASPHelper._is_supported_layer(prog, param.name):
                mat = np.array(
                    base.global_scope().find_var(param.name).get_tensor()
                )
                if (len(param.shape) == 4 and param.shape[1] < 4) or (
                    len(param.shape) == 2 and param.shape[0] < 4
                ):
                    self.assertFalse(
                        paddle.incubate.asp.check_sparsity(mat.T, n=2, m=4)
                    )
                else:
                    self.assertTrue(
                        paddle.incubate.asp.check_sparsity(mat.T, n=2, m=4)
                    )


if __name__ == '__main__':
    unittest.main()
