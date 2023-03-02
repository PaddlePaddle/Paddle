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
from test_dist_base import TestDistRunnerBase, runtime_main

import paddle
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.fluid as fluid
from paddle.distributed.fleet.meta_optimizers import (
    FP16AllReduceOptimizer as FP16AllReduce,
)
from paddle.distributed.fleet.meta_optimizers import (
    RawProgramOptimizer as RawProgram,
)

DTYPE = "float32"
paddle.dataset.mnist.fetch()

# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1


class TestDistMnist2x2(TestDistRunnerBase):
    def get_model(self, batch_size=2, single_device=False, dist_strategy=None):
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

        inference_program = fluid.default_main_program().clone()
        # Optimization
        opt = fluid.optimizer.MomentumOptimizer(
            learning_rate=0.001, momentum=0.9
        )

        if not single_device:
            fleet.init()

        opt = FP16AllReduce(opt)

        if not single_device:
            opt = RawProgram(opt)
            role = role_maker.PaddleCloudRoleMaker(is_collective=True)
            strategy = (
                paddle.distributed.fleet.DistributedStrategy()
                if dist_strategy is None
                else dist_strategy
            )
            opt._set_basic_info(avg_cost, role, opt, strategy)

            # following code is a copy of RawProgramOptimizer.minimize except init_comm_group
            opt.endpoints = opt.role_maker._get_trainer_endpoints()
            opt.current_endpoint = opt.endpoints[opt.role_maker._worker_index()]
            opt.rank = opt.role_maker._worker_index()
            opt.nranks = opt.role_maker._worker_num()
            startup_program = paddle.static.default_startup_program()
            opt.startup_program = startup_program

            block = avg_cost.block
            program = block.program
            opt.main_program = program

            optimize_ops, params_grads = opt.inner_opt.minimize(
                avg_cost, startup_program
            )

            opt.main_program = program
            if opt.nranks > 1:
                opt._transpile_main_program(avg_cost)
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
