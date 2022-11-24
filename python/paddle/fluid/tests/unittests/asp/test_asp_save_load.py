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

<<<<<<< HEAD
from __future__ import print_function

=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
import unittest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.contrib.sparsity.asp import ASPHelper
import numpy as np


class MyLayer(paddle.nn.Layer):
<<<<<<< HEAD

    def __init__(self):
        super(MyLayer, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=3,
                                      out_channels=4,
                                      kernel_size=3,
                                      padding=2)
=======
    def __init__(self):
        super().__init__()
        self.conv1 = paddle.nn.Conv2D(
            in_channels=3, out_channels=4, kernel_size=3, padding=2
        )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
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


class TestASPDynamicOptimize(unittest.TestCase):
<<<<<<< HEAD

=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    def setUp(self):
        paddle.disable_static()

        self.layer = MyLayer()

        self.place = paddle.CPUPlace()
        if core.is_compiled_with_cuda():
            self.place = paddle.CUDAPlace(0)

        self.optimizer = paddle.optimizer.SGD(
<<<<<<< HEAD
            learning_rate=0.01, parameters=self.layer.parameters())
=======
            learning_rate=0.01, parameters=self.layer.parameters()
        )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
        self.optimizer = paddle.incubate.asp.decorate(self.optimizer)
        paddle.incubate.asp.prune_model(self.layer)

    def test_save_and_load(self):
        path = "/tmp/paddle_asp_save_dy/"
        net_path = path + "asp_net.pdparams"
        opt_path = path + "asp_opt.pdopt"

        paddle.save(self.layer.state_dict(), net_path)
        paddle.save(self.optimizer.state_dict(), opt_path)

        asp_info = ASPHelper._get_program_asp_info(
<<<<<<< HEAD
            paddle.static.default_main_program())
        for param_name in asp_info.mask_vars:
            mask = asp_info.mask_vars[param_name]
            asp_info.update_mask_vars(
                param_name, paddle.ones(shape=mask.shape, dtype=mask.dtype))
=======
            paddle.static.default_main_program()
        )
        for param_name in asp_info.mask_vars:
            mask = asp_info.mask_vars[param_name]
            asp_info.update_mask_vars(
                param_name, paddle.ones(shape=mask.shape, dtype=mask.dtype)
            )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
            asp_info.update_masks(param_name, np.ones(shape=mask.shape))

        net_state_dict = paddle.load(net_path)
        opt_state_dict = paddle.load(opt_path)

        self.layer.set_state_dict(net_state_dict)
        self.optimizer.set_state_dict(opt_state_dict)

<<<<<<< HEAD
        imgs = paddle.to_tensor(np.random.randn(64, 3, 32, 32),
                                dtype='float32',
                                place=self.place,
                                stop_gradient=False)
        labels = paddle.to_tensor(np.random.randint(10, size=(64, 1)),
                                  dtype='float32',
                                  place=self.place,
                                  stop_gradient=False)
=======
        imgs = paddle.to_tensor(
            np.random.randn(64, 3, 32, 32),
            dtype='float32',
            place=self.place,
            stop_gradient=False,
        )
        labels = paddle.to_tensor(
            np.random.randint(10, size=(64, 1)),
            dtype='float32',
            place=self.place,
            stop_gradient=False,
        )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

        loss_fn = paddle.nn.MSELoss(reduction='mean')

        output = self.layer(imgs)
        loss = loss_fn(output, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.clear_grad()

        for param in self.layer.parameters():
            if ASPHelper._is_supported_layer(
<<<<<<< HEAD
                    paddle.static.default_main_program(), param.name):
                mat = param.numpy()
                if (len(param.shape) == 4
                        and param.shape[1] < 4) or (len(param.shape) == 2
                                                    and param.shape[0] < 4):
                    self.assertFalse(
                        paddle.fluid.contrib.sparsity.check_sparsity(mat.T,
                                                                     n=2,
                                                                     m=4))
                else:
                    self.assertTrue(
                        paddle.fluid.contrib.sparsity.check_sparsity(mat.T,
                                                                     n=2,
                                                                     m=4))


class TestASPStaticOptimize(unittest.TestCase):

=======
                paddle.static.default_main_program(), param.name
            ):
                mat = param.numpy()
                if (len(param.shape) == 4 and param.shape[1] < 4) or (
                    len(param.shape) == 2 and param.shape[0] < 4
                ):
                    self.assertFalse(
                        paddle.fluid.contrib.sparsity.check_sparsity(
                            mat.T, n=2, m=4
                        )
                    )
                else:
                    self.assertTrue(
                        paddle.fluid.contrib.sparsity.check_sparsity(
                            mat.T, n=2, m=4
                        )
                    )


class TestASPStaticOptimize(unittest.TestCase):
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    def setUp(self):
        paddle.enable_static()

        self.main_program = fluid.Program()
        self.startup_program = fluid.Program()

        def build_model():
<<<<<<< HEAD
            img = fluid.data(name='img',
                             shape=[None, 3, 32, 32],
                             dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            hidden = fluid.layers.conv2d(input=img,
                                         num_filters=4,
                                         filter_size=3,
                                         padding=2,
                                         act="relu")
=======
            img = fluid.data(
                name='img', shape=[None, 3, 32, 32], dtype='float32'
            )
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            hidden = fluid.layers.conv2d(
                input=img, num_filters=4, filter_size=3, padding=2, act="relu"
            )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
            hidden = fluid.layers.fc(input=hidden, size=32, act='relu')
            prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
            return img, label, prediction

        with fluid.program_guard(self.main_program, self.startup_program):
            self.img, self.label, predict = build_model()
            self.loss = paddle.mean(
<<<<<<< HEAD
                fluid.layers.cross_entropy(input=predict, label=self.label))
=======
                fluid.layers.cross_entropy(input=predict, label=self.label)
            )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
            self.optimizer = fluid.optimizer.SGD(learning_rate=0.01)
            self.optimizer = paddle.incubate.asp.decorate(self.optimizer)
            self.optimizer.minimize(self.loss, self.startup_program)

        self.place = paddle.CPUPlace()
        if core.is_compiled_with_cuda():
            self.place = paddle.CUDAPlace(0)
        self.exe = fluid.Executor(self.place)
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

<<<<<<< HEAD
        feeder = fluid.DataFeeder(feed_list=[self.img, self.label],
                                  place=self.place)

        data = (np.random.randn(64, 3, 32,
                                32), np.random.randint(10, size=(64, 1)))
=======
        feeder = fluid.DataFeeder(
            feed_list=[self.img, self.label], place=self.place
        )

        data = (
            np.random.randn(64, 3, 32, 32),
            np.random.randint(10, size=(64, 1)),
        )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
        self.exe.run(prog, feed=feeder.feed([data]))

        for param in prog.global_block().all_parameters():
            if ASPHelper._is_supported_layer(prog, param.name):
<<<<<<< HEAD
                mat = np.array(fluid.global_scope().find_var(
                    param.name).get_tensor())
                if (len(param.shape) == 4
                        and param.shape[1] < 4) or (len(param.shape) == 2
                                                    and param.shape[0] < 4):
                    self.assertFalse(
                        paddle.fluid.contrib.sparsity.check_sparsity(mat.T,
                                                                     n=2,
                                                                     m=4))
                else:
                    self.assertTrue(
                        paddle.fluid.contrib.sparsity.check_sparsity(mat.T,
                                                                     n=2,
                                                                     m=4))
=======
                mat = np.array(
                    fluid.global_scope().find_var(param.name).get_tensor()
                )
                if (len(param.shape) == 4 and param.shape[1] < 4) or (
                    len(param.shape) == 2 and param.shape[0] < 4
                ):
                    self.assertFalse(
                        paddle.fluid.contrib.sparsity.check_sparsity(
                            mat.T, n=2, m=4
                        )
                    )
                else:
                    self.assertTrue(
                        paddle.fluid.contrib.sparsity.check_sparsity(
                            mat.T, n=2, m=4
                        )
                    )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f


if __name__ == '__main__':
    unittest.main()
