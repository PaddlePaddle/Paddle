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

from legacy_test import nets
from legacy_test.test_dist_base import TestDistRunnerBase, runtime_main

import paddle
from paddle import base
from paddle.distributed import fleet

paddle.enable_static()

DTYPE = "float32"
paddle.dataset.mnist.fetch()

# Fix seed for test
base.default_startup_program().random_seed = 1
base.default_main_program().random_seed = 1


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
    param_shape = [reduce(lambda a, b: a * b, input_shape[1:], 1)] + [SIZE]
    scale = (2.0 / (param_shape[0] ** 2 * SIZE)) ** 0.5

    with base.device_guard("gpu:1"):
        predict = paddle.static.nn.fc(
            x=conv_pool_2,
            size=SIZE,
            activation="softmax",
            weight_attr=base.param_attr.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.01)
            ),
        )
        # To cover @RENAMED@GRADIENT
        predict2 = paddle.static.nn.fc(
            x=conv_pool_1,
            size=SIZE,
            activation="softmax",
            weight_attr=base.param_attr.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.01)
            ),
        )
        predict += predict2
    return predict


class TestDistMnist2x2(TestDistRunnerBase):
    def get_model(self, batch_size=2, use_dgc=False, dist_strategy=None):
        # Input data
        with base.device_guard("gpu:0"):
            images = paddle.static.data(
                name='pixel', shape=[-1, 1, 28, 28], dtype=DTYPE
            )
            label = paddle.static.data(
                name='label', shape=[-1, 1], dtype='int64'
            )

            if dist_strategy:
                data_loader = base.io.DataLoader.from_generator(
                    feed_list=[images, label],
                    capacity=64,
                    use_double_buffer=False,
                    iterable=False,
                )
            # Train program
            predict = cnn_model(images)
        with base.device_guard("gpu:1"):
            cost = paddle.nn.functional.cross_entropy(
                input=predict, label=label, reduction='none', use_softmax=False
            )
            avg_cost = paddle.mean(x=cost)

        # Evaluator
        with base.device_guard("gpu:1"):
            batch_size_tensor = paddle.tensor.create_tensor(dtype='int64')
            batch_acc = paddle.static.accuracy(
                input=predict, label=label, total=batch_size_tensor
            )

        inference_program = base.default_main_program().clone()
        base_lr = self.lr
        passes = [30, 60, 80, 90]
        steps_per_pass = 10
        bd = [steps_per_pass * p for p in passes]
        lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
        lr_val = paddle.optimizer.lr.PiecewiseDecay(boundaries=bd, values=lr)
        opt = paddle.optimizer.Momentum(
            learning_rate=lr_val,
            momentum=0.9,
            grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0),
        )

        acc_steps = 2  # accumulated steps for pipeline
        if dist_strategy:
            # Reader
            train_reader = paddle.batch(
                paddle.dataset.mnist.test(), batch_size=batch_size
            )
            test_reader = paddle.batch(
                paddle.dataset.mnist.test(), batch_size=batch_size
            )
            fleet.init(is_collective=True)
            strategy = fleet.DistributedStrategy()
            strategy.pipeline = True
            strategy.amp = True
            strategy.pipeline_configs = {
                'micro_batch_size': batch_size,
                'schedule_mode': 'F-then-B',
                'accumulate_steps': acc_steps,
            }
            dist_opt = fleet.distributed_optimizer(
                optimizer=opt, strategy=strategy
            )
            dist_opt.minimize(avg_cost)
        else:
            opt.minimize(avg_cost)
            # Reader
            train_reader = paddle.batch(
                paddle.dataset.mnist.test(), batch_size=batch_size * acc_steps
            )
            test_reader = paddle.batch(
                paddle.dataset.mnist.test(), batch_size=batch_size * acc_steps
            )

        if dist_strategy:
            return (
                inference_program,
                avg_cost,
                train_reader,
                test_reader,
                batch_acc,
                predict,
                data_loader,
            )
        else:
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
