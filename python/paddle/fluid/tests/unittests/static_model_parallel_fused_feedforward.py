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

from paddle.fluid.data_feeder import check_variable_and_dtype, check_dtype
from paddle.fluid.dygraph.layers import Layer
from paddle.fluid.layer_helper import LayerHelper
from paddle.nn.initializer import Constant

paddle.enable_static()

DTYPE = "float32"
MODEL_PARALLEL_SIZE = 2
IN_SIZE = 2 * MODEL_PARALLEL_SIZE
OUT_SIZE = 2 * MODEL_PARALLEL_SIZE


def fused_feedforward(x,
                      linear1_weight,
                      linear2_weight,
                      linear1_bias=None,
                      linear2_bias=None,
                      ln1_scale=None,
                      ln1_bias=None,
                      ln2_scale=None,
                      ln2_bias=None,
                      dropout1_rate=0.5,
                      dropout2_rate=0.5,
                      activation="relu",
                      ln1_epsilon=1e-5,
                      ln2_epsilon=1e-5,
                      pre_layer_norm=False,
                      training=True,
                      mode='upscale_in_train',
                      ring_id=-1,
                      name=None):
    seed = None
    if mode not in ('downscale_in_infer', 'upscale_in_train'):
        raise ValueError(
            "mode argument should be 'downscale_in_infer' or 'upscale_in_train'")
    mode = 'downgrade_in_infer' if mode == 'downscale_in_infer' else mode  #semantic transfer

    helper = LayerHelper("fused_feedforward")
    dtype = x.dtype
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                             'fused_feedforward')
    check_dtype(dtype, 'dtype', ['float16', 'float32', 'float64'],
                'fused_feedforward')

    out = helper.create_variable_for_type_inference(x.dtype)
    dropout1_mask = helper.create_variable_for_type_inference(
        'uint8', stop_gradient=True)
    dropout2_mask = helper.create_variable_for_type_inference(
        'uint8', stop_gradient=True)
    ln1_mean = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    ln1_variance = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    ln2_mean = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    ln2_variance = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    linear1_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    ln1_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    dropout1_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    dropout2_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)

    if (seed is None or seed == 0) and helper.main_program.random_seed != 0:
        seed = helper.main_program.random_seed

    helper.append_op(
        type='fused_feedforward',
        inputs={
            'X': x,
            'Linear1Weight': linear1_weight,
            'Linear1Bias': linear1_bias,
            'Linear2Weight': linear2_weight,
            'Linear2Bias': linear2_bias,
            'Ln1Scale': ln1_scale,
            'Ln1Bias': ln1_bias,
            'Ln2Scale': ln2_scale,
            'Ln2Bias': ln2_bias,
        },
        outputs={
            'Out': out,
            'Dropout1Mask': dropout1_mask,
            'Dropout2Mask': dropout2_mask,
            'Ln1Mean': ln1_mean,
            'Ln1Variance': ln1_variance,
            'Ln2Mean': ln2_mean,
            'Ln2Variance': ln2_variance,
            'Linear1Out': linear1_out,
            'Ln1Out': ln1_out,
            'Dropout1Out': dropout1_out,
            'Dropout2Out': dropout2_out,
        },
        attrs={
            'dropout1_rate': dropout1_rate,
            'dropout2_rate': dropout2_rate,
            'act_method': activation,
            'pre_layer_norm': pre_layer_norm,
            'ln1_epsilon': ln1_epsilon,
            'ln2_epsilon': ln2_epsilon,
            'dropout1_is_test': not training,
            'dropout2_is_test': not training,
            'dropout1_fix_seed': seed is not None,
            'dropout2_fix_seed': seed is not None,
            'dropout1_seed': seed if seed is not None else 0,
            'dropout2_seed': seed if seed is not None else 0,
            'dropout1_implementation': mode,
            'dropout2_implementation': mode,
            'ring_id': ring_id,
        })
    return out


def _set_var_distributed(var):
    if var is None:
        return

    var.is_distributed = True

    # NOTE: use current_block and find_var_recursive to support while_loop
    startup_block = paddle.static.default_startup_program().current_block()
    main_block = paddle.static.default_main_program().current_block()
    startup_block._find_var_recursive(var.name).is_distributed = True
    main_block._find_var_recursive(var.name).is_distributed = True


class ParallelFusedFeedForward(Layer):
    def __init__(self,
                 d_model,
                 dim_feedforward,
                 dropout_rate=0.1,
                 epsilon=1e-05,
                 activation="relu",
                 act_dropout_rate=None,
                 normalize_before=False,
                 linear1_weight_attr=None,
                 linear1_bias_attr=None,
                 linear2_weight_attr=None,
                 linear2_bias_attr=None,
                 ln1_scale_attr=None,
                 ln1_bias_attr=None,
                 ln2_scale_attr=None,
                 ln2_bias_attr=None,
                 nranks=1,
                 ring_id=-1,
                 name=None):
        super(ParallelFusedFeedForward, self).__init__()
        assert d_model > 0, (
            "Expected d_model to be greater than 0, but recieved {}".format(
                d_model))
        assert dim_feedforward > 0, (
            "Expected dim_feedforward to be greater than 0, but recieved {}".
            format(dim_feedforward))

        self._dtype = self._helper.get_default_dtype()
        self._d_model = d_model

        assert dim_feedforward % nranks == 0
        dim_feedforward = dim_feedforward // nranks
        self._dim_feedforward = dim_feedforward
        self._dropout_rate = dropout_rate
        self._act_dropout_rate = dropout_rate if act_dropout_rate is None else act_dropout_rate
        self._act_method = activation
        self._normalize_before = normalize_before
        self._epsilon = epsilon
        self._ring_id = ring_id

        self._linear1_weight = self.create_parameter(
            shape=[d_model, dim_feedforward],
            attr=linear1_weight_attr,
            dtype=self._dtype,
            is_bias=False)
        self._linear1_bias = self.create_parameter(
            shape=[dim_feedforward],
            attr=linear1_bias_attr,
            dtype=self._dtype,
            is_bias=True)

        self._linear2_weight = self.create_parameter(
            shape=[dim_feedforward, d_model],
            attr=linear2_weight_attr,
            dtype=self._dtype,
            is_bias=False)

        self._linear2_bias = self.create_parameter(
            shape=[d_model],
            attr=linear2_bias_attr,
            dtype=self._dtype,
            is_bias=True)

        if nranks > 1:
            assert ring_id != -1
            # column parallel
            _set_var_distributed(self._linear1_weight)
            _set_var_distributed(self._linear1_bias)
            _set_var_distributed(self._linear2_weight)

        if normalize_before:
            self._ln1_scale = self.create_parameter(
                shape=[d_model],
                attr=ln1_scale_attr,
                is_bias=False,
                default_initializer=Constant(1.0))
            self._ln1_bias = self.create_parameter(
                shape=[d_model], attr=ln1_bias_attr, is_bias=True)
            self._ln2_scale = None
            self._ln2_bias = None
        else:
            self._ln1_bias = None
            self._ln2_bias = None
            self._ln2_scale = self.create_parameter(
                shape=[d_model],
                attr=ln2_scale_attr,
                is_bias=False,
                default_initializer=Constant(1.0))
            self._ln2_bias = self.create_parameter(
                shape=[d_model], attr=ln2_bias_attr, is_bias=True)

        self.name = name

    def forward(self, src, cache=None):
        out = fused_feedforward(
            src,
            self._linear1_weight,
            self._linear2_weight,
            self._linear1_bias,
            self._linear2_bias,
            self._ln1_scale,
            self._ln1_bias,
            self._ln2_scale,
            self._ln2_bias,
            dropout1_rate=self._act_dropout_rate,
            dropout2_rate=self._dropout_rate,
            activation=self._act_method,
            ln1_epsilon=self._epsilon,
            ln2_epsilon=self._epsilon,
            pre_layer_norm=self._normalize_before,
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

        ffn = ParallelFusedFeedForward(
            IN_SIZE,
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

        ffn = ParallelFusedFeedForward(
            IN_SIZE,
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
        data_in = fluid.data(
            name='data_in', shape=[batch_size, seq_len, IN_SIZE], dtype=DTYPE)

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
                data = [np.random.random([seq_len, IN_SIZE]).astype(DTYPE)]
                yield data

        train_reader = paddle.batch(gen_data, batch_size=batch_size)

        if dist_strategy:
            return None, avg_cost, train_reader, None, None, None, data_loader
        else:
            return None, avg_cost, train_reader, None, None, None


if __name__ == "__main__":
    runtime_main(TestModelParallel)
