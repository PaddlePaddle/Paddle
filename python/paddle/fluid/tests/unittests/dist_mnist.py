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

import warnings
from functools import reduce

from test_dist_base import TestDistRunnerBase, runtime_main

import paddle
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.fluid as fluid
from paddle.distributed.fleet.meta_optimizers import (
    RawProgramOptimizer as RawProgram,
)

paddle.enable_static()

DTYPE = "float32"
paddle.dataset.mnist.fetch()

# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1


def cnn_model(data):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=data,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu",
        param_attr=fluid.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=0.01)
        ),
    )
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu",
        param_attr=fluid.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=0.01)
        ),
    )

    SIZE = 10
    input_shape = conv_pool_2.shape
    param_shape = [reduce(lambda a, b: a * b, input_shape[1:], 1)] + [SIZE]
    scale = (2.0 / (param_shape[0] ** 2 * SIZE)) ** 0.5

    predict = paddle.static.nn.fc(
        x=conv_pool_2,
        size=SIZE,
        activation="softmax",
        weight_attr=fluid.param_attr.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=0.01)
        ),
    )
    return predict


class TestDistMnist2x2(TestDistRunnerBase):
    def get_model(self, batch_size=2, use_dgc=False, dist_strategy=None):
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
        # TODO(typhoonzero): fix distributed adam optimizer
        # opt = fluid.optimizer.AdamOptimizer(
        #     learning_rate=0.001, beta1=0.9, beta2=0.999)
        if not use_dgc:
            opt = fluid.optimizer.Momentum(learning_rate=self.lr, momentum=0.9)
        else:
            opt = paddle.distributed.fleet.meta_optimizers.DGCMomentumOptimizer(
                learning_rate=self.lr,
                momentum=0.9,
                rampup_begin_step=2,
                num_trainers=dist_strategy.build_strategy.num_trainers
                if dist_strategy
                else None,
            )

        # Reader
        train_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size
        )
        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size
        )

        if dist_strategy:
            warnings.warn("Use dist strategy.")
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
