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

import paddle
import paddle.fluid as fluid
from test_dist_base import TestDistRunnerBase, runtime_main
from dist_mnist import cnn_model

DTYPE = "float32"
paddle.dataset.mnist.fetch()

# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1


class TestDistMnist2x2(TestDistRunnerBase):

    def get_model(self, batch_size=2):
        # Input data
        images = fluid.layers.data(name='pixel', shape=[1, 28, 28], dtype=DTYPE)
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

        # Train program
        predict = cnn_model(images)
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        avg_cost = paddle.mean(x=cost)

        # Evaluator
        batch_size_tensor = fluid.layers.create_tensor(dtype='int64')
        batch_acc = fluid.layers.accuracy(input=predict,
                                          label=label,
                                          total=batch_size_tensor)

        inference_program = fluid.default_main_program().clone()
        # Optimization
        opt = fluid.optimizer.MomentumOptimizer(learning_rate=0.001,
                                                momentum=0.9)
        opt = fluid.optimizer.GradientMergeOptimizer(opt, 2)

        # Reader
        train_reader = paddle.batch(paddle.dataset.mnist.test(),
                                    batch_size=batch_size)
        test_reader = paddle.batch(paddle.dataset.mnist.test(),
                                   batch_size=batch_size)
        opt.minimize(avg_cost)
        return inference_program, avg_cost, train_reader, test_reader, batch_acc, predict


if __name__ == "__main__":
    runtime_main(TestDistMnist2x2)
