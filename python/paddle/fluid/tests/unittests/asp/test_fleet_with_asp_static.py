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

import paddle.distributed.fleet as fleet
<<<<<<< HEAD
import unittest
import paddle
import paddle.fluid as fluid
=======
import paddle.distributed.fleet.base.role_maker as role_maker
import unittest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
import os
from paddle.static import sparsity
from paddle.fluid.contrib.sparsity.asp import ASPHelper
import numpy as np

cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
if cuda_visible_devices is None or cuda_visible_devices == "":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices.split(',')[0]

paddle.enable_static()


class TestFleetWithASPStatic(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def setUp(self):
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36213"
        os.environ["PADDLE_CURRENT_ENDPOINTS"] = "127.0.0.1:36213"
        os.environ["PADDLE_TRAINERS_NUM"] = "1"
        os.environ["PADDLE_TRAINER_ID"] = "0"

    def net(self, main_prog, startup_prog):
        with fluid.program_guard(main_prog, startup_prog):
<<<<<<< HEAD
            input_x = paddle.static.data(
                name="x", shape=[-1, 32], dtype='float32'
            )
=======
            input_x = paddle.static.data(name="x",
                                         shape=[-1, 32],
                                         dtype='float32')
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
            input_y = paddle.static.data(name="y", shape=[-1, 1], dtype='int64')

            fc_1 = fluid.layers.fc(input=input_x, size=64, act='tanh')
            prediction = fluid.layers.fc(input=fc_1, size=2, act='softmax')
            cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
            avg_cost = paddle.mean(x=cost)

            strategy = paddle.distributed.fleet.DistributedStrategy()
            strategy.asp = True
        return avg_cost, strategy, input_x, input_y

    def test_with_asp(self):
        fleet.init(is_collective=True)
        train_prog, startup_prog = fluid.Program(), fluid.Program()
<<<<<<< HEAD
        avg_cost, strategy, input_x, input_y = self.net(
            train_prog, startup_prog
        )

        with fluid.program_guard(train_prog, startup_prog):
            optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.01)
            optimizer = fleet.distributed_optimizer(
                optimizer, strategy=strategy
            )
            optimizer.minimize(avg_cost)

        place = (
            fluid.CUDAPlace(0)
            if paddle.fluid.is_compiled_with_cuda()
            else fluid.CPUPlace()
        )
=======
        avg_cost, strategy, input_x, input_y = self.net(train_prog,
                                                        startup_prog)

        with fluid.program_guard(train_prog, startup_prog):
            optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.01)
            optimizer = fleet.distributed_optimizer(optimizer,
                                                    strategy=strategy)
            optimizer.minimize(avg_cost)

        place = fluid.CUDAPlace(
            0) if paddle.fluid.is_compiled_with_cuda() else fluid.CPUPlace()
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(feed_list=[input_x, input_y], place=place)
        exe.run(startup_prog)

        sparsity.prune_model(train_prog)

        data = (np.random.randn(64, 32), np.random.randint(2, size=(64, 1)))
        exe.run(train_prog, feed=feeder.feed([data]))

        for param in train_prog.global_block().all_parameters():
            if ASPHelper._is_supported_layer(train_prog, param.name):
<<<<<<< HEAD
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


class TestFleetWithASPAMPStatic(unittest.TestCase):
=======
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


class TestFleetWithASPAMPStatic(unittest.TestCase):

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def setUp(self):
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36213"
        os.environ["PADDLE_CURRENT_ENDPOINTS"] = "127.0.0.1:36213"
        os.environ["PADDLE_TRAINERS_NUM"] = "1"
        os.environ["PADDLE_TRAINER_ID"] = "0"

    def net(self, main_prog, startup_prog):
        with fluid.program_guard(main_prog, startup_prog):
<<<<<<< HEAD
            input_x = paddle.static.data(
                name="x", shape=[-1, 32], dtype='float32'
            )
=======
            input_x = paddle.static.data(name="x",
                                         shape=[-1, 32],
                                         dtype='float32')
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
            input_y = paddle.static.data(name="y", shape=[-1, 1], dtype='int64')

            fc_1 = fluid.layers.fc(input=input_x, size=64, act='tanh')
            prediction = fluid.layers.fc(input=fc_1, size=2, act='softmax')
            cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
            avg_cost = paddle.mean(x=cost)

            strategy = paddle.distributed.fleet.DistributedStrategy()
            strategy.asp = True
        return avg_cost, strategy, input_x, input_y

    def test_with_asp_and_amp(self):
        fleet.init(is_collective=True)
        train_prog, startup_prog = fluid.Program(), fluid.Program()
<<<<<<< HEAD
        avg_cost, strategy, input_x, input_y = self.net(
            train_prog, startup_prog
        )
=======
        avg_cost, strategy, input_x, input_y = self.net(train_prog,
                                                        startup_prog)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        strategy.amp = True

        with fluid.program_guard(train_prog, startup_prog):
            optimizer = paddle.optimizer.SGD(learning_rate=0.01)
<<<<<<< HEAD
            optimizer = fleet.distributed_optimizer(
                optimizer, strategy=strategy
            )
            optimizer.minimize(avg_cost)

        place = (
            fluid.CUDAPlace(0)
            if paddle.fluid.is_compiled_with_cuda()
            else fluid.CPUPlace()
        )
=======
            optimizer = fleet.distributed_optimizer(optimizer,
                                                    strategy=strategy)
            optimizer.minimize(avg_cost)

        place = fluid.CUDAPlace(
            0) if paddle.fluid.is_compiled_with_cuda() else fluid.CPUPlace()
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(feed_list=[input_x, input_y], place=place)
        exe.run(startup_prog)

        optimizer.amp_init(place)

        sparsity.prune_model(train_prog)

        data = (np.random.randn(64, 32), np.random.randint(2, size=(64, 1)))
        exe.run(train_prog, feed=feeder.feed([data]))

        for param in train_prog.global_block().all_parameters():
            if ASPHelper._is_supported_layer(train_prog, param.name):
<<<<<<< HEAD
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
=======
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
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

    def test_with_asp_and_pure_fp16(self):
        fleet.init(is_collective=True)
        train_prog, startup_prog = fluid.Program(), fluid.Program()
        with paddle.static.amp.fp16_guard():
<<<<<<< HEAD
            avg_cost, strategy, input_x, input_y = self.net(
                train_prog, startup_prog
            )
=======
            avg_cost, strategy, \
                input_x, input_y = self.net(train_prog,
                                            startup_prog)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        strategy.amp = True
        strategy.amp_configs = {'use_pure_fp16': True}

        with fluid.program_guard(train_prog, startup_prog):
            with paddle.static.amp.fp16_guard():
                optimizer = optimizer = paddle.optimizer.Momentum(
<<<<<<< HEAD
                    learning_rate=0.01, multi_precision=True
                )
                optimizer = fleet.distributed_optimizer(
                    optimizer, strategy=strategy
                )
                optimizer.minimize(avg_cost)

        place = (
            fluid.CUDAPlace(0)
            if paddle.fluid.is_compiled_with_cuda()
            else fluid.CPUPlace()
        )
=======
                    learning_rate=0.01, multi_precision=True)
                optimizer = fleet.distributed_optimizer(optimizer,
                                                        strategy=strategy)
                optimizer.minimize(avg_cost)

        place = fluid.CUDAPlace(
            0) if paddle.fluid.is_compiled_with_cuda() else fluid.CPUPlace()
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(feed_list=[input_x, input_y], place=place)
        exe.run(startup_prog)

        optimizer.amp_init(place)

        sparsity.prune_model(train_prog)

        data = (np.random.randn(64, 32), np.random.randint(2, size=(64, 1)))
        exe.run(train_prog, feed=feeder.feed([data]))

        for param in train_prog.global_block().all_parameters():
            if ASPHelper._is_supported_layer(train_prog, param.name):
<<<<<<< HEAD
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
=======
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
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf


if __name__ == "__main__":
    unittest.main()
