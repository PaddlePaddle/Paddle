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
import paddle.incubate.nn.functional as incubate_f

from paddle.fluid.data_feeder import check_variable_and_dtype, check_dtype
from paddle.fluid.dygraph.layers import Layer
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid import core
from paddle.nn.initializer import Constant

paddle.enable_static()


def _set_var_distributed(var):
    if var is None:
        return

    var.is_distributed = True

    # NOTE: use current_block and find_var_recursive to support while_loop
    startup_block = paddle.static.default_startup_program().current_block()
    main_block = paddle.static.default_main_program().current_block()
    startup_block._find_var_recursive(var.name).is_distributed = True
    main_block._find_var_recursive(var.name).is_distributed = True


class ParallelFusedMultiHeadAttention(Layer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout_rate=0.5,
                 attn_dropout_rate=0.5,
                 kdim=None,
                 vdim=None,
                 normalize_before=False,
                 need_weights=False,
                 qkv_weight_attr=None,
                 qkv_bias_attr=None,
                 linear_weight_attr=None,
                 linear_bias_attr=None,
                 pre_ln_scale_attr=None,
                 pre_ln_bias_attr=None,
                 ln_scale_attr=None,
                 ln_bias_attr=None,
                 epsilon=1e-5,
                 nranks=1,
                 ring_id=-1,
                 name=None):
        super(ParallelFusedMultiHeadAttention, self).__init__()

        assert embed_dim > 0, ("Expected embed_dim to be greater than 0, "
                               "but recieved {}".format(embed_dim))
        assert num_heads > 0, ("Expected nhead to be greater than 0, "
                               "but recieved {}".format(num_heads))

        self.normalize_before = normalize_before
        self._dtype = self._helper.get_default_dtype()
        self._epsilon = epsilon
        self._ring_id = ring_id

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kdim = kdim
        self.vdim = vdim
        self.need_weights = need_weights
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        assert need_weights == False, "Only support need_weight is False now."

        # tensor model parallel
        assert num_heads % nranks == 0
        num_heads = num_heads // nranks

        self.qkv_weight = self.create_parameter(
            shape=[3, num_heads, self.head_dim, embed_dim],
            attr=qkv_weight_attr,
            dtype=self._dtype,
            is_bias=False)
        self.qkv_bias = self.create_parameter(
            shape=[3, num_heads, self.head_dim],
            attr=qkv_bias_attr,
            dtype=self._dtype,
            is_bias=True)
        self.linear_weight = self.create_parameter(
            shape=[num_heads * self.head_dim, embed_dim],
            attr=linear_weight_attr,
            dtype=self._dtype,
            is_bias=False)
        self.linear_bias = self.create_parameter(
            shape=[embed_dim],
            attr=linear_bias_attr,
            dtype=self._dtype,
            is_bias=True)

        # tensor model parallel
        if nranks > 1:
            assert ring_id != -1
            # column parallel
            _set_var_distributed(self.qkv_weight)
            _set_var_distributed(self.qkv_bias)
            # row parallel
            _set_var_distributed(self.linear_weight)

        if normalize_before:
            self.pre_ln_scale = self.create_parameter(
                attr=pre_ln_scale_attr,
                shape=[embed_dim],
                default_initializer=Constant(value=1.0))
            self.pre_ln_bias = self.create_parameter(
                attr=pre_ln_bias_attr, shape=[embed_dim], is_bias=True)
            self.ln_scale = None
            self.ln_bias = None
        else:
            self.pre_ln_scale = None
            self.pre_ln_bias = None
            self.ln_scale = self.create_parameter(
                attr=ln_scale_attr,
                shape=[embed_dim],
                default_initializer=Constant(value=1.0))
            self.ln_bias = self.create_parameter(
                attr=ln_bias_attr, shape=[embed_dim], is_bias=True)

        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate

        self.name = name

    def forward(self, query, key=None, value=None, attn_mask=None, cache=None):
        out = incubate_f.fused_multi_head_attention(
            x=query,
            qkv_weight=self.qkv_weight,
            linear_weight=self.linear_weight,
            pre_layer_norm=self.normalize_before,
            pre_ln_scale=self.pre_ln_scale,
            pre_ln_bias=self.pre_ln_bias,
            ln_scale=self.ln_scale,
            ln_bias=self.ln_bias,
            pre_ln_epsilon=self._epsilon,
            qkv_bias=self.qkv_bias,
            linear_bias=self.linear_bias,
            attn_mask=attn_mask,
            dropout_rate=self.dropout_rate,
            attn_dropout_rate=self.attn_dropout_rate,
            ln_epsilon=self._epsilon,
            training=self.training,
            ring_id=self._ring_id,
            name=self.name)
        return out


def get_param_attr(weight, bias):
    weight_attr = paddle.ParamAttr(
        initializer=fluid.initializer.NumpyArrayInitializer(weight))
    bias_attr = paddle.ParamAttr(
        initializer=fluid.initializer.NumpyArrayInitializer(bias))
    return weight_attr, bias_attr


DTYPE = "float32"
MODEL_PARALLEL_SIZE = 2
n_head = 2 * MODEL_PARALLEL_SIZE
d_key = 4
hidden = n_head * d_key


def create_model(data, rank):
    np.random.seed(2021)
    pre_ln_w = np.random.uniform(-1, 1, size=(hidden, )).astype(DTYPE)
    pre_ln_b = np.random.uniform(-1, 1, size=(hidden, )).astype(DTYPE)
    qkv_w = np.random.uniform(
        -1, 1, size=(3, n_head, d_key, hidden)).astype(DTYPE)
    qkv_b = np.random.uniform(-1, 1, size=(3, n_head, d_key)).astype(DTYPE)
    linear_w = np.random.uniform(
        -1, 1, size=(n_head * d_key, hidden)).astype(DTYPE)
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

        attn = ParallelFusedMultiHeadAttention(
            hidden,
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

        attn = ParallelFusedMultiHeadAttention(
            hidden,
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
        data_in = fluid.data(
            name='data_in', shape=[batch_size, seq_len, hidden], dtype=DTYPE)

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
            dist_opt = fleet.distributed_optimizer(
                optimizer=opt, strategy=strategy)
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
