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

from dist_mnist import cnn_model
from test_dist_base import TestDistRunnerBase, runtime_main

import paddle
import paddle.fluid as fluid

DTYPE = "float32"


def test_merge_reader(repeat_batch_size=8):
    orig_reader = paddle.dataset.mnist.test()
    record_batch = []
    b = 0
    for d in orig_reader():
        if b >= repeat_batch_size:
            break
        record_batch.append(d)
        b += 1
    while True:
        for d in record_batch:
            yield d


class TestDistMnist2x2(TestDistRunnerBase):
    def get_model(self, batch_size=2):
        # Input data
        images = fluid.layers.data(name='pixel', shape=[1, 28, 28], dtype=DTYPE)
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

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

        inference_program = fluid.default_main_program().clone()
        # Optimization
        opt = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9)

        # Reader
        train_reader = paddle.batch(test_merge_reader, batch_size=batch_size)
        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size
        )
        opt.minimize(avg_cost)
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
