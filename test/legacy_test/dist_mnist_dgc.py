#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from legacy_test.nets import simple_img_conv_pool
from legacy_test.test_dist_base import (
    TestDistRunnerBase,
    _insert_comm_op,
    runtime_main,
)
from paddle import base

paddle.enable_static()

DTYPE = "float32"
paddle.dataset.mnist.fetch()

# Fix seed for test
paddle.seed(2023)


def cnn_model(data):
    conv_pool_1 = simple_img_conv_pool(
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
    conv_pool_2 = simple_img_conv_pool(
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


class TestDistMnistDGC(TestDistRunnerBase):
    def get_model(self, batch_size=2, use_dgc=False, build_strategy=None):
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
        if not use_dgc:
            opt = paddle.optimizer.Momentum(learning_rate=self.lr, momentum=0.9)
        else:
            opt = paddle.distributed.fleet.meta_optimizers.DGCMomentumOptimizer(
                learning_rate=self.lr,
                momentum=0.9,
                rampup_begin_step=2,
                num_trainers=(
                    build_strategy.num_trainers if build_strategy else None
                ),
            )
        if use_dgc:
            assert (
                build_strategy is not None
            ), "build_strategy can be None with dgc"
            paddle.distributed.collective._init_parallel_env("nccl")
            _insert_comm_op(opt, avg_cost, build_strategy)
        else:
            opt.minimize(avg_cost)

        # Reader
        train_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size
        )
        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size
        )

        return (
            inference_program,
            avg_cost,
            train_reader,
            test_reader,
            batch_acc,
            predict,
        )


if __name__ == "__main__":
    runtime_main(TestDistMnistDGC)
