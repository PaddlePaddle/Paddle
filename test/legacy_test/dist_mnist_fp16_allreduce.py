# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from dist_mnist import cnn_model
from test_dist_base import TestDistRunnerBase, _insert_comm_op, runtime_main

import paddle
from paddle import base
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers import (
    FP16AllReduceOptimizer as FP16AllReduce,
)

DTYPE = "float32"
paddle.dataset.mnist.fetch()

# Fix seed for test
base.default_startup_program().random_seed = 1
base.default_main_program().random_seed = 1


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
        # Optimization
        opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9)

        opt = FP16AllReduce(opt)

        if not single_device:
            fleet.init()
            _insert_comm_op(opt, avg_cost)
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
    runtime_main(TestDistMnist2x2)
