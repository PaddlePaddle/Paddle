#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from legacy_test.test_dist_base import TestDistRunnerBase, runtime_main

import paddle
from paddle import base
from paddle.distributed import fleet

paddle.enable_static()

DTYPE = "float32"
MODEL_PARALLEL_SIZE = 2
IN_SIZE = 2 * MODEL_PARALLEL_SIZE
OUT_SIZE = 2 * MODEL_PARALLEL_SIZE


def get_param_attr(weight, bias):
    weight_attr = paddle.ParamAttr(
        initializer=paddle.nn.initializer.Assign(weight)
    )
    bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(bias))
    return weight_attr, bias_attr


def create_model(data, rank):
    np.random.seed(2021)
    np_weight = np.random.uniform(-1, 1, size=(IN_SIZE, OUT_SIZE)).astype(DTYPE)
    np_bias = np.random.uniform(-1, 1, size=(OUT_SIZE,)).astype(DTYPE)
    if rank is not None:
        start_col = 0 if rank == 0 else OUT_SIZE // 2
        np_weight_part = np_weight[:, start_col : start_col + OUT_SIZE // 2]
        np_bias_part = np_bias[start_col : start_col + OUT_SIZE // 2]

        weight_attr, bias_attr = get_param_attr(np_weight_part, np_bias_part)
        result = paddle.distributed.split(
            data,
            size=(IN_SIZE, OUT_SIZE),
            operation='linear',
            axis=1,
            num_partitions=MODEL_PARALLEL_SIZE,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
        )
    else:
        weight_attr, bias_attr = get_param_attr(np_weight, np_bias)
        result = paddle.static.nn.fc(
            data, size=OUT_SIZE, weight_attr=weight_attr, bias_attr=bias_attr
        )

    predict = paddle.sum(result)
    return predict


class TestModelParallel(TestDistRunnerBase):
    def get_model(self, batch_size=2, use_dgc=False, dist_strategy=None):
        # Input data
        data_in = paddle.static.data(
            name='data_in', shape=[batch_size, IN_SIZE], dtype=DTYPE
        )

        if dist_strategy:
            data_loader = base.io.DataLoader.from_generator(
                feed_list=[data_in],
                capacity=64,
                use_double_buffer=False,
                iterable=False,
            )

        if dist_strategy:
            fleet.init(is_collective=True)
            strategy = fleet.DistributedStrategy()
            strategy.tensor_parallel = True
            strategy.tensor_parallel_configs = {'tensor_parallel_degree': 2}

        rank = fleet.worker_index() if dist_strategy else None
        avg_cost = create_model(data_in, rank)
        opt = paddle.optimizer.SGD(0.1)

        if dist_strategy:
            dist_opt = fleet.distributed_optimizer(
                optimizer=opt, strategy=strategy
            )
            dist_opt.minimize(avg_cost)
        else:
            opt.minimize(avg_cost)

        def gen_data():
            np.random.seed(2021)
            while True:
                data = [np.random.random([IN_SIZE]).astype(DTYPE)]
                yield data

        train_reader = paddle.batch(gen_data, batch_size=batch_size)

        if dist_strategy:
            return None, avg_cost, train_reader, None, None, None, data_loader
        else:
            return None, avg_cost, train_reader, None, None, None


if __name__ == "__main__":
    runtime_main(TestModelParallel)
