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
import paddle.distributed.fleet as fleet
from paddle.incubate.nn import FusedFeedForward

paddle.enable_static()

DTYPE = "float32"
MODEL_PARALLEL_SIZE = 2
IN_SIZE = 2 * MODEL_PARALLEL_SIZE
OUT_SIZE = 2 * MODEL_PARALLEL_SIZE


def get_param_attr(weight, bias):
    weight_attr = paddle.ParamAttr(
        initializer=fluid.initializer.NumpyArrayInitializer(weight))
    bias_attr = paddle.ParamAttr(
        initializer=fluid.initializer.NumpyArrayInitializer(bias))
    return weight_attr, bias_attr


def create_model(data, rank):
    np.random.seed(2021)
    ln_w = np.random.uniform(-1, 1, size=(IN_SIZE, )).astype(DTYPE)
    ln_b = np.random.uniform(-1, 1, size=(IN_SIZE, )).astype(DTYPE)
    w0 = np.random.uniform(-1, 1, size=(IN_SIZE, OUT_SIZE)).astype(DTYPE)
    b0 = np.random.uniform(-1, 1, size=(OUT_SIZE, )).astype(DTYPE)
    w1 = np.random.uniform(-1, 1, size=(OUT_SIZE, IN_SIZE)).astype(DTYPE)
    b1 = np.random.uniform(-1, 1, size=(IN_SIZE, )).astype(DTYPE)
    data.stop_gradient = False
    if rank is not None:
        start = 0 if rank == 0 else OUT_SIZE // MODEL_PARALLEL_SIZE
        end = start + OUT_SIZE // MODEL_PARALLEL_SIZE
        col_w0 = w0[:, start:end]
        col_b0 = b0[start:end]
        row_w1 = w1[start:end, :]

        ln_w_attr, ln_b_attr = get_param_attr(ln_w, ln_b)
        w0_attr, b0_attr = get_param_attr(col_w0, col_b0)
        w1_attr, b1_attr = get_param_attr(row_w1, b1)

        ffn = FusedFeedForward(IN_SIZE,
                               OUT_SIZE,
                               dropout_rate=0.0,
                               activation='gelu',
                               normalize_before=True,
                               linear1_weight_attr=w0_attr,
                               linear1_bias_attr=b0_attr,
                               linear2_weight_attr=w1_attr,
                               linear2_bias_attr=b1_attr,
                               ln1_scale_attr=ln_w_attr,
                               ln1_bias_attr=ln_b_attr,
                               nranks=MODEL_PARALLEL_SIZE,
                               ring_id=0)
        #ffn.eval()
        result = ffn(data)
    else:
        ln_w_attr, ln_b_attr = get_param_attr(ln_w, ln_b)
        w0_attr, b0_attr = get_param_attr(w0, b0)
        w1_attr, b1_attr = get_param_attr(w1, b1)

        ffn = FusedFeedForward(IN_SIZE,
                               OUT_SIZE,
                               dropout_rate=0.0,
                               activation='gelu',
                               normalize_before=True,
                               linear1_weight_attr=w0_attr,
                               linear1_bias_attr=b0_attr,
                               linear2_weight_attr=w1_attr,
                               linear2_bias_attr=b1_attr,
                               ln1_scale_attr=ln_w_attr,
                               ln1_bias_attr=ln_b_attr)
        #ffn.eval()
        result = ffn(data)

    predict = paddle.sum(result)
    return predict


class TestModelParallel(TestDistRunnerBase):

    def get_model(self, batch_size=2, use_dgc=False, dist_strategy=None):
        # Input data
        seq_len = 2
        data_in = fluid.data(name='data_in',
                             shape=[batch_size, seq_len, IN_SIZE],
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
                data = [np.random.random([seq_len, IN_SIZE]).astype(DTYPE)]
                yield data

        train_reader = paddle.batch(gen_data, batch_size=batch_size)

        if dist_strategy:
            return None, avg_cost, train_reader, None, None, None, data_loader
        else:
            return None, avg_cost, train_reader, None, None, None


if __name__ == "__main__":
    runtime_main(TestModelParallel)
