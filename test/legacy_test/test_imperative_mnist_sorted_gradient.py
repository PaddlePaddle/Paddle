# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from test_imperative_base import new_program_scope
from test_imperative_mnist import MNIST

import paddle
from paddle import base
from paddle.base import core
from paddle.base.dygraph.base import to_variable


class TestImperativeMnistSortGradient(unittest.TestCase):
    def test_mnist_sort_gradient_float32(self):
        seed = 90
        epoch_num = 1

        with base.dygraph.guard():
            base.default_startup_program().random_seed = seed
            base.default_main_program().random_seed = seed
            base.set_flags({'FLAGS_sort_sum_gradient': True})

            mnist2 = MNIST()
            sgd2 = paddle.optimizer.SGD(
                learning_rate=1e-3, parameters=mnist2.parameters()
            )
            train_reader2 = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=128, drop_last=True
            )

            mnist2.train()
            dy_param_init_value2 = {}
            for epoch in range(epoch_num):
                for batch_id, data in enumerate(train_reader2()):
                    dy_x_data2 = np.array(
                        [x[0].reshape(1, 28, 28) for x in data]
                    ).astype('float32')
                    y_data2 = (
                        np.array([x[1] for x in data])
                        .astype('int64')
                        .reshape(128, 1)
                    )

                    img2 = to_variable(dy_x_data2)
                    label2 = to_variable(y_data2)
                    label2.stop_gradient = True

                    cost2 = mnist2(img2)
                    loss2 = paddle.nn.functional.cross_entropy(
                        cost2, label2, reduction='none', use_softmax=False
                    )
                    avg_loss2 = paddle.mean(loss2)

                    dy_out2 = avg_loss2.numpy()

                    if epoch == 0 and batch_id == 0:
                        for param in mnist2.parameters():
                            dy_param_init_value2[param.name] = param.numpy()

                    avg_loss2.backward()
                    sgd2.minimize(avg_loss2)
                    mnist2.clear_gradients()

                    dy_param_value2 = {}
                    for param in mnist2.parameters():
                        dy_param_value2[param.name] = param.numpy()
                    if batch_id == 20:
                        break

        with new_program_scope():
            base.default_startup_program().random_seed = seed
            base.default_main_program().random_seed = seed

            exe = base.Executor(
                base.CPUPlace()
                if not core.is_compiled_with_cuda()
                else base.CUDAPlace(0)
            )

            mnist = MNIST()
            sgd = paddle.optimizer.SGD(learning_rate=1e-3)
            train_reader = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=128, drop_last=True
            )

            img = paddle.static.data(
                name='pixel', shape=[-1, 1, 28, 28], dtype='float32'
            )
            label = paddle.static.data(
                name='label', shape=[-1, 1], dtype='int64'
            )
            cost = mnist(img)
            loss = paddle.nn.functional.cross_entropy(
                cost, label, reduction='none', use_softmax=False
            )
            avg_loss = paddle.mean(loss)
            sgd.minimize(avg_loss)

            # initialize params and fetch them
            static_param_init_value = {}
            static_param_name_list = []
            for param in mnist.parameters():
                static_param_name_list.append(param.name)

            out = exe.run(
                base.default_startup_program(),
                fetch_list=static_param_name_list,
            )

            for i in range(len(static_param_name_list)):
                static_param_init_value[static_param_name_list[i]] = out[i]

            for epoch in range(epoch_num):
                for batch_id, data in enumerate(train_reader()):
                    static_x_data = np.array(
                        [x[0].reshape(1, 28, 28) for x in data]
                    ).astype('float32')
                    y_data = (
                        np.array([x[1] for x in data])
                        .astype('int64')
                        .reshape([128, 1])
                    )

                    fetch_list = [avg_loss.name]
                    fetch_list.extend(static_param_name_list)
                    out = exe.run(
                        base.default_main_program(),
                        feed={"pixel": static_x_data, "label": y_data},
                        fetch_list=fetch_list,
                    )

                    static_param_value = {}
                    static_out = out[0]
                    for i in range(1, len(out)):
                        static_param_value[static_param_name_list[i - 1]] = out[
                            i
                        ]
                    if batch_id == 20:
                        break

        np.testing.assert_allclose(
            dy_x_data2.all(), static_x_data.all(), rtol=1e-05
        )

        for key, value in static_param_init_value.items():
            np.testing.assert_allclose(
                value, dy_param_init_value2[key], rtol=1e-05
            )

        np.testing.assert_allclose(static_out, dy_out2, rtol=1e-05)

        for key, value in static_param_value.items():
            np.testing.assert_allclose(
                value, dy_param_value2[key], rtol=1e-05, atol=1e-05
            )


if __name__ == '__main__':
    unittest.main()
