# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys

import paddle
from paddle import static
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.static.cluster import Cluster
from paddle.distributed.auto_parallel.static.cost import CostEstimator
from paddle.distributed.auto_parallel.static.dist_context import (
    get_default_distributed_context,
)

sys.path.append("../../auto_parallel")


def train():
    from auto_parallel_relaunch_model import (
        batch_generator_creator,
        mlp_pretrain_forward,
    )

    dist_strategy = fleet.DistributedStrategy()
    # init parallel optimizer
    dist_strategy.auto_search = True
    fleet.init(is_collective=True, strategy=dist_strategy)
    train_program = static.Program()
    start_program = static.Program()
    loss, train_program, start_program, loader = mlp_pretrain_forward(
        train_program, start_program
    )

    optimizer = paddle.optimizer.Adam(
        learning_rate=0.00001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        grad_clip=None,
    )

    optimizer = fleet.distributed_optimizer(optimizer)
    (
        _,
        _,
        distributed_startup_program,
        distributed_main_program,
    ) = optimizer.minimize(loss, start_program)

    # add cost estimator
    dist_context = get_default_distributed_context()
    cluster = Cluster()
    for op in train_program.global_block().ops:
        dist_op = dist_context.get_dist_op_for_program(op)
        for var_name in op.input_arg_names:
            dims_mapping = dist_op.dist_attr.get_input_dims_mapping(var_name)
            if dims_mapping is None:
                dist_op.dist_attr.set_input_dims_mapping(
                    var_name,
                    [
                        -1
                        for i in range(
                            len(
                                train_program.global_block()
                                .vars[var_name]
                                .shape
                            )
                        )
                    ],
                )
    cluster.gen_default_config_cluster(device_count=2)
    cost_estimator = CostEstimator(train_program, cluster)
    global_cost = cost_estimator.estimate(dist_context)
    max_memory = cost_estimator._estimate_max_memory_by_dist_op(dist_context)
    # test cache
    global_cost = cost_estimator.estimate(dist_context)
    max_memory = cost_estimator._estimate_max_memory_by_dist_op(dist_context)
    assert global_cost.time > 0
    assert max_memory > 0

    places = static.cuda_places()
    loader.set_batch_generator(batch_generator_creator(), places=places)
    exe = paddle.static.Executor(places[0])
    exe.run(distributed_startup_program)

    for data in loader():
        exe.run(distributed_main_program, feed=data)


if __name__ == "__main__":
    train()
