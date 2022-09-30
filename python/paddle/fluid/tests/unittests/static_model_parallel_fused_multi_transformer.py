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

import numpy as np

import paddle
import paddle.fluid as fluid
from test_dist_base import TestDistRunnerBase, runtime_main
from paddle.incubate.nn import FusedMultiTransformer
import paddle.distributed.fleet as fleet

from paddle.fluid.data_feeder import check_variable_and_dtype, check_dtype
from paddle.fluid.dygraph.layers import Layer
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid import core
from paddle.nn.initializer import Constant

paddle.enable_static()


def get_param_attr(weight, bias):
    weight_attr = paddle.ParamAttr(
        initializer=fluid.initializer.NumpyArrayInitializer(weight))
    bias_attr = paddle.ParamAttr(
        initializer=fluid.initializer.NumpyArrayInitializer(bias))
    return weight_attr, bias_attr


DTYPE = "float32"
MODEL_PARALLEL_SIZE = 2
num_head = 2 * MODEL_PARALLEL_SIZE
dim_head = 4
hidden = num_head * dim_head
dim_ffn = 4 * hidden


def create_model(data, rank):
    np.random.seed(2021)
    ln_w = np.random.uniform(-1, 1, size=(hidden, )).astype(DTYPE)
    ln_b = np.random.uniform(-1, 1, size=(hidden, )).astype(DTYPE)
    qkv_w = np.random.uniform(-1, 1, size=(3, num_head, dim_head,
                                           hidden)).astype(DTYPE)
    qkv_b = np.random.uniform(-1, 1, size=(3, num_head, dim_head)).astype(DTYPE)
    linear_w = np.random.uniform(-1, 1, size=(num_head * dim_head,
                                              hidden)).astype(DTYPE)
    linear_b = np.random.uniform(-1, 1, size=(hidden, )).astype(DTYPE)

    ffn_ln_w = np.random.uniform(-1, 1, size=(hidden, )).astype(DTYPE)
    ffn_ln_b = np.random.uniform(-1, 1, size=(hidden, )).astype(DTYPE)
    ffn1_w = np.random.uniform(-1, 1, size=(hidden, dim_ffn)).astype(DTYPE)
    ffn1_b = np.random.uniform(-1, 1, size=(dim_ffn, )).astype(DTYPE)
    ffn2_w = np.random.uniform(-1, 1, size=(dim_ffn, hidden)).astype(DTYPE)
    ffn2_b = np.random.uniform(-1, 1, size=(hidden, )).astype(DTYPE)

    if rank is not None:
        start = 0 if rank == 0 else (num_head // MODEL_PARALLEL_SIZE)
        end = start + (num_head // MODEL_PARALLEL_SIZE)
        col_qkv_w = qkv_w[:, start:end, :, :]
        col_qkv_b = qkv_b[:, start:end, :]
        row_linear_w = linear_w[(start * dim_head):(end * dim_head), :]

        ln_w_attr, ln_b_attr = get_param_attr(ln_w, ln_b)
        qkv_w_attr, qkv_b_attr = get_param_attr(col_qkv_w, col_qkv_b)
        linear_w_attr, linear_b_attr = get_param_attr(row_linear_w, linear_b)

        start = 0 if rank == 0 else (dim_ffn // MODEL_PARALLEL_SIZE)
        end = start + (dim_ffn // MODEL_PARALLEL_SIZE)
        col_ffn1_w = ffn1_w[:, start:end]
        col_ffn1_b = ffn1_b[start:end]
        row_ffn2_w = ffn2_w[start:end, :]

        ffn_ln_w_attr, ffn_ln_b_attr = get_param_attr(ffn_ln_w, ffn_ln_b)
        ffn1_w_attr, ffn1_b_attr = get_param_attr(col_ffn1_w, col_ffn1_b)
        ffn2_w_attr, ffn2_b_attr = get_param_attr(row_ffn2_w, ffn2_b)

        multi_transformer = FusedMultiTransformer(
            hidden,
            num_head,
            dim_ffn,
            dropout_rate=0.0,
            activation="gelu",
            normalize_before=True,
            ln_scale_attrs=[ln_w_attr],
            ln_bias_attrs=[ln_b_attr],
            qkv_weight_attrs=[qkv_w_attr],
            qkv_bias_attrs=[qkv_b_attr],
            linear_weight_attrs=[linear_w_attr],
            linear_bias_attrs=[linear_b_attr],
            ffn_ln_scale_attrs=[ffn_ln_w_attr],
            ffn_ln_bias_attrs=[ffn_ln_b_attr],
            ffn1_weight_attrs=[ffn1_w_attr],
            ffn1_bias_attrs=[ffn1_b_attr],
            ffn2_weight_attrs=[ffn2_w_attr],
            ffn2_bias_attrs=[ffn2_b_attr],
            nranks=MODEL_PARALLEL_SIZE,
            ring_id=0)
        result = multi_transformer(data)
    else:
        ln_w_attr, ln_b_attr = get_param_attr(ln_w, ln_b)
        qkv_w_attr, qkv_b_attr = get_param_attr(qkv_w, qkv_b)
        linear_w_attr, linear_b_attr = get_param_attr(linear_w, linear_b)

        ffn_ln_w_attr, ffn_ln_b_attr = get_param_attr(ffn_ln_w, ffn_ln_b)
        ffn1_w_attr, ffn1_b_attr = get_param_attr(ffn1_w, ffn1_b)
        ffn2_w_attr, ffn2_b_attr = get_param_attr(ffn2_w, ffn2_b)

        multi_transformer = FusedMultiTransformer(
            hidden,
            num_head,
            dim_ffn,
            dropout_rate=0.0,
            activation="gelu",
            normalize_before=True,
            ln_scale_attrs=[ln_w_attr],
            ln_bias_attrs=[ln_b_attr],
            qkv_weight_attrs=[qkv_w_attr],
            qkv_bias_attrs=[qkv_b_attr],
            linear_weight_attrs=[linear_w_attr],
            linear_bias_attrs=[linear_b_attr],
            ffn_ln_scale_attrs=[ffn_ln_w_attr],
            ffn_ln_bias_attrs=[ffn_ln_b_attr],
            ffn1_weight_attrs=[ffn1_w_attr],
            ffn1_bias_attrs=[ffn1_b_attr],
            ffn2_weight_attrs=[ffn2_w_attr],
            ffn2_bias_attrs=[ffn2_b_attr])
        result = multi_transformer(data)

    # fused_multi_transformer have no backward
    result.stop_gradient = True
    predict = paddle.mean(result)
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
