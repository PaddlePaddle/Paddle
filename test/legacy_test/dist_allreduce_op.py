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

from functools import reduce

import nets
from test_dist_base import TestDistRunnerBase, runtime_main

import paddle
from paddle import base
from paddle.distributed import fleet

paddle.enable_static()

DTYPE = "float32"
paddle.dataset.mnist.fetch()

# Fix seed for test
paddle.seed(1)


def cnn_model(data):
    conv_pool_1 = nets.simple_img_conv_pool(
        input=data,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu",
        param_attr=base.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=0.01)
        ),
    )
    conv_pool_2 = nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu",
        param_attr=base.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=0.01)
        ),
    )

    SIZE = 10
    input_shape = conv_pool_2.shape
    param_shape = [reduce(lambda a, b: a * b, input_shape[1:], 1), SIZE]
    scale = (2.0 / (param_shape[0] ** 2 * SIZE)) ** 0.5

    predict = paddle.static.nn.fc(
        x=conv_pool_2,
        size=SIZE,
        activation="softmax",
        weight_attr=base.param_attr.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=0.01)
        ),
    )
    return predict


class TestDistMnist2x2(TestDistRunnerBase):
    def get_model(self, batch_size=2, single_device=False):
        # Input data
        images = paddle.static.data(
            name='pixel', shape=[-1, 1, 28, 28], dtype=DTYPE
        )
        label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')

        # Train program
        predict = cnn_model(images)
        cost = paddle.nn.functional.cross_entropy(
            input=predict, label=label, reduction='none', use_softmax=False
        )
        avg_cost = paddle.mean(x=cost)

        # Evaluator
        batch_size_tensor = paddle.tensor.create_tensor(dtype='int64')
        batch_acc = paddle.static.accuracy(
            input=predict, label=label, total=batch_size_tensor
        )

        inference_program = base.default_main_program().clone()

        # Reader
        train_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size
        )
        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size
        )

        # Optimization
        # TODO(typhoonzero): fix distributed adam optimizer
        # opt = paddle.optimizer.Adam(
        #     learning_rate=0.001, beta1=0.9, beta2=0.999)
        opt = paddle.optimizer.Momentum(learning_rate=self.lr, momentum=0.9)
        if single_device:
            opt.minimize(avg_cost)
        else:
            # multi device or distributed multi device
            strategy = fleet.DistributedStrategy()
            strategy.without_graph_optimization = True
            fleet.init(strategy=strategy)
            optimizer = fleet.distributed_optimizer(opt)
            optimizer.minimize(avg_cost)

        return (
            inference_program,
            avg_cost,
            train_reader,
            test_reader,
            batch_acc,
            predict,
        )


if __name__ == "__main__":
    runtime_main(TestDistMnist2x2)
