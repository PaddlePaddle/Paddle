#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy as np

import paddle
import paddle.fluid as fluid
from test_dist_base import TestDistRunnerBase, runtime_main
import paddle.distributed.fleet as fleet
from paddle.incubate.nn import FusedMultiHeadAttention

paddle.enable_static()


def get_param_attr(weight, bias):
    weight_attr = paddle.ParamAttr(
        initializer=fluid.initializer.NumpyArrayInitializer(weight))
    bias_attr = paddle.ParamAttr(
        initializer=fluid.initializer.NumpyArrayInitializer(bias))
    return weight_attr, bias_attr


DTYPE = "float32"
MODEL_PARALLEL_SIZE = 2
n_head = 2 * MODEL_PARALLEL_SIZE
d_key = 2
hidden = n_head * d_key


def create_model(data, rank):
    np.random.seed(2021)
    pre_ln_w = np.random.uniform(-1, 1, size=(hidden, )).astype(DTYPE)
    pre_ln_b = np.random.uniform(-1, 1, size=(hidden, )).astype(DTYPE)
    qkv_w = np.random.uniform(-1, 1,
                              size=(3, n_head, d_key, hidden)).astype(DTYPE)
    qkv_b = np.random.uniform(-1, 1, size=(3, n_head, d_key)).astype(DTYPE)
    linear_w = np.random.uniform(-1, 1,
                                 size=(n_head * d_key, hidden)).astype(DTYPE)
    linear_b = np.random.uniform(-1, 1, size=(hidden, )).astype(DTYPE)

    data.stop_gradient = False
    if rank is not None:
        start = 0 if rank == 0 else n_head // MODEL_PARALLEL_SIZE
        end = start + n_head // MODEL_PARALLEL_SIZE
        col_qkv_w = qkv_w[:, start:end, :, :]
        col_qkv_b = qkv_b[:, start:end, :]
        row_linear_w = linear_w[(start * d_key):(end * d_key), :]

        pre_ln_w_attr, pre_ln_b_attr = get_param_attr(pre_ln_w, pre_ln_b)
        qkv_w_attr, qkv_b_attr = get_param_attr(col_qkv_w, col_qkv_b)
        linear_w_attr, linear_b_attr = get_param_attr(row_linear_w, linear_b)

        attn = FusedMultiHeadAttention(hidden,
                                       n_head,
                                       dropout_rate=0.0,
                                       attn_dropout_rate=0.0,
                                       normalize_before=False,
                                       qkv_weight_attr=qkv_w_attr,
                                       qkv_bias_attr=qkv_b_attr,
                                       linear_weight_attr=linear_w_attr,
                                       linear_bias_attr=linear_b_attr,
                                       pre_ln_scale_attr=pre_ln_w_attr,
                                       pre_ln_bias_attr=pre_ln_b_attr,
                                       ln_scale_attr=pre_ln_w_attr,
                                       ln_bias_attr=pre_ln_b_attr,
                                       nranks=MODEL_PARALLEL_SIZE,
                                       ring_id=0)
        result = attn(data)
    else:
        pre_ln_w_attr, pre_ln_b_attr = get_param_attr(pre_ln_w, pre_ln_b)
        qkv_w_attr, qkv_b_attr = get_param_attr(qkv_w, qkv_b)
        linear_w_attr, linear_b_attr = get_param_attr(linear_w, linear_b)

        attn = FusedMultiHeadAttention(hidden,
                                       n_head,
                                       dropout_rate=0.0,
                                       attn_dropout_rate=0.0,
                                       normalize_before=False,
                                       qkv_weight_attr=qkv_w_attr,
                                       qkv_bias_attr=qkv_b_attr,
                                       linear_weight_attr=linear_w_attr,
                                       linear_bias_attr=linear_b_attr,
                                       pre_ln_scale_attr=pre_ln_w_attr,
                                       pre_ln_bias_attr=pre_ln_b_attr,
                                       ln_scale_attr=pre_ln_w_attr,
                                       ln_bias_attr=pre_ln_b_attr)
        result = attn(data)

    predict = paddle.sum(result)
    return predict


class TestModelParallel(TestDistRunnerBase):

    def get_model(self, batch_size=2, use_dgc=False, dist_strategy=None):
        # Input data
        seq_len = 2
        data_in = fluid.data(name='data_in',
                             shape=[batch_size, seq_len, hidden],
                             dtype=DTYPE)

        if dist_strategy:
            data_loader = fluid.io.DataLoader.from_generator(
                feed_list=[data_in],
                capacity=64,
                use_double_buffer=False,
                iterable=False)

        if dist_strategy:
            fleet.init(is_collective=True)
            strategy = fleet.DistributedStrategy()
            strategy.tensor_parallel = True
            strategy.tensor_parallel_configs = {'tensor_parallel_degree': 2}

        rank = fleet.worker_index() if dist_strategy else None
        avg_cost = create_model(data_in, rank)
        opt = fluid.optimizer.SGD(0.1)

        if dist_strategy:
            dist_opt = fleet.distributed_optimizer(optimizer=opt,
                                                   strategy=strategy)
            dist_opt.minimize(avg_cost)
        else:
            opt.minimize(avg_cost)

        def gen_data():
            np.random.seed(2021)
            while True:
                data = [np.random.random([seq_len, hidden]).astype(DTYPE)]
                yield data

        train_reader = paddle.batch(gen_data, batch_size=batch_size)

        if dist_strategy:
            return None, avg_cost, train_reader, None, None, None, data_loader
        else:
            return None, avg_cost, train_reader, None, None, None


if __name__ == "__main__":
    runtime_main(TestModelParallel)
