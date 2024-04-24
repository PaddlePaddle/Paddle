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

import logging
import sys
import time

import numpy as np

import paddle
from paddle import static

from ..log_helper import get_logger
from .utils import (
    _channelwise_quant_axis1_ops,
    bias_correction_w,
    calculate_quant_cos_error,
    dequant_tensor,
    load_variable_data,
    quant_tensor,
    set_variable_data,
    stable_sigmoid,
)

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)

GAMMA = -0.1
ZETA = 1.1


def compute_soft_rounding(alpha_v):
    return paddle.clip(
        paddle.nn.functional.sigmoid(alpha_v) * (ZETA - GAMMA) + GAMMA,
        min=0,
        max=1,
    )


def compute_soft_rounding_np(alpha_v):
    return np.clip(
        stable_sigmoid(alpha_v) * (ZETA - GAMMA) + GAMMA, a_min=0, a_max=1
    )


class AdaRoundLoss:
    def __init__(self, reg_param=0.01, default_beta_range=(20, 2)):
        self.default_reg_param = reg_param
        self.default_beta_range = default_beta_range

    def compute_recon_loss(self, ada_quantized_output, orig_output):
        square_cost = paddle.nn.functional.square_error_cost(
            ada_quantized_output, orig_output
        )
        recon_loss = paddle.mean(paddle.sum(square_cost, axis=-1))
        return recon_loss

    def compute_round_loss(self, alpha_v, warm_start, beta):
        def round_loss_fn():
            # compute rectified sigmoid of parameter 'alpha' which maps it between zero and one
            h_v = compute_soft_rounding(alpha_v)

            # calculate regularization term - which ensures parameter to converge to exactly zeros and ones
            # at the end of optimization
            reg_term = paddle.sum(
                -paddle.pow(paddle.abs(2 * h_v - 1), beta) + 1
            )

            # calculate the rounding loss
            round_loss = self.default_reg_param * reg_term

            return round_loss

        round_loss = static.nn.cond(
            warm_start,
            lambda: paddle.full(shape=[1], dtype='float32', fill_value=0.0),
            round_loss_fn,
        )

        return round_loss

    def compute_beta(self, max_iter, cur_iter, warm_start):
        #  Start and stop beta for annealing of rounding loss (start_beta, end_beta)
        start_beta, end_beta = self.default_beta_range

        # iteration at end of warm start period, which is 20% of max iterations
        warm_start_end_iter = warm_start * max_iter

        # compute relative iteration of current iteration
        rel_iter = (cur_iter - warm_start_end_iter) / (
            max_iter - warm_start_end_iter
        )
        beta = end_beta + 0.5 * (start_beta - end_beta) * (
            1 + np.cos(rel_iter * np.pi)
        )

        return beta


class AdaRound:
    def __init__(
        self,
        scale,
        weight_tensor,
        scope=None,
        weight_var_name=None,
        weight_op_type=None,
        is_train=True,
        num_iterations=1000,
    ):
        self.is_train = is_train
        self.num_iterations = num_iterations
        self.warm_start = 0.1
        self.weight_bits = 8
        self.offset = 0.0  # zero-point offset
        self.adaround_loss = AdaRoundLoss()
        self.ori_weight_tensor = weight_tensor
        self.scale = scale
        self.scope = scope
        self.quant_axis = 0
        if weight_op_type in _channelwise_quant_axis1_ops:
            self.quant_axis = 1
        self.weight_var_name = weight_var_name
        self.alpha_name = weight_var_name + ".alpha"
        self.initialize_alpha(weight_tensor.copy(), scale, weight_var_name)

    def initialize_alpha(self, tensor, scale, var_name):
        """
        Initializes alpha parameter, same shape as the weight tensor
        """
        tensor_scale = quant_tensor(tensor, scale, quant_axis=self.quant_axis)
        tensor_floor = np.floor(tensor_scale)
        tensor = tensor_scale - tensor_floor
        alpha = -np.log((ZETA - GAMMA) / (tensor - GAMMA) - 1)
        self.alpha_v = paddle.create_parameter(
            shape=alpha.shape,
            dtype="float32",
            name=var_name + ".alpha",
            default_initializer=paddle.nn.initializer.Assign(alpha),
        )

    def _calculate_output_with_adarounded_weights(
        self, program, place, exe, data, fp32_fetch_list, weight_tensor_dequant
    ):
        set_variable_data(
            self.scope, place, self.weight_var_name, weight_tensor_dequant
        )

        adaround_out_tensor = exe.run(
            program=program,
            feed=data,
            fetch_list=[fp32_fetch_list],
            return_numpy=True,
            scope=self.scope,
        )
        return adaround_out_tensor

    def _calculate_quant_weight(self):
        np_alpha = load_variable_data(self.scope, self.alpha_name)
        h_alpha = compute_soft_rounding_np(np_alpha)

        # Scale the tensor
        tensor_scale = quant_tensor(
            self.ori_weight_tensor.copy(),
            self.scale,
            quant_axis=self.quant_axis,
        )

        weight_tensor = np.floor(tensor_scale)

        # Adaround the tensor
        weight_tensor_quant = np.add(weight_tensor, h_alpha)
        return weight_tensor_quant

    def _calculate_adarounded_weights(self):
        weight_tensor_quant = self._calculate_quant_weight()

        # Dequantize the tensor
        weight_tensor_dequant = dequant_tensor(
            weight_tensor_quant + self.offset,
            self.scale,
            quant_axis=self.quant_axis,
        )
        return weight_tensor_dequant

    def update_final_weights(self):
        weight_tensor_quant = self._calculate_quant_weight()
        return weight_tensor_quant

    def get_loss(self, beta, warm_start, adaround_out_tensor, orig_out_tensor):
        round_loss = self.adaround_loss.compute_round_loss(
            self.alpha_v, warm_start, beta
        )
        recon_loss = self.adaround_loss.compute_recon_loss(
            adaround_out_tensor, orig_out_tensor
        )
        loss = round_loss + recon_loss
        losses = {
            'loss': loss,
            'round_loss': round_loss,
            'recon_loss': recon_loss,
        }
        return losses

    def update_beta_warm(self, cur_iteration):
        warm_start = cur_iteration < self.num_iterations * self.warm_start
        beta = self.adaround_loss.compute_beta(
            self.num_iterations, cur_iteration, self.warm_start
        )
        return beta, warm_start


def run_adaround(
    data_loader,
    fp32_program,
    fetch_list,
    exe,
    scope,
    place,
    quantized_op_pairs,
    weight_op_pairs,
    scale_dict,
    num_iterations=1000,
    lr=0.001,
    bias_correction=False,
    fast_mode=True,
):
    fetch_op_name = fetch_list[0].name
    final_weight_tensor_quant_dict = {}
    for weight_var_name, quant_op_out_name in quantized_op_pairs.items():
        _logger.info(f'Start adaround op: {weight_var_name}')
        weight_op_type = weight_op_pairs[weight_var_name]
        # get scale and weight tensor
        weight_var_tensor = load_variable_data(scope, weight_var_name)
        scale = scale_dict[weight_var_name]
        fp32_fetch_list = None
        for _op in fp32_program.global_block().ops:
            if _op.type == "fetch":
                _op._rename_input(fetch_op_name, quant_op_out_name)
                fp32_fetch_list = fp32_program.global_block().var(
                    quant_op_out_name
                )
                fetch_op_name = quant_op_out_name

        # build adaround program
        startup_program = static.Program()
        train_program = static.Program()
        with static.program_guard(train_program, startup_program):
            with paddle.utils.unique_name.guard():
                # initialize adaround
                adaround = AdaRound(
                    scale,
                    weight_var_tensor,
                    scope=scope,
                    weight_var_name=weight_var_name,
                    weight_op_type=weight_op_type,
                    num_iterations=num_iterations,
                )
                orig_out_tensor = static.data(
                    name='orig_out_tensor',
                    shape=(-1,) + fp32_fetch_list.shape,
                    dtype='float32',
                )
                adaround_out_tensor = static.data(
                    name='adaround_out_tensor',
                    shape=(-1,) + fp32_fetch_list.shape,
                    dtype='float32',
                )
                beta_tensor = static.data(
                    name='beta', shape=[-1, 1], dtype='float32'
                )
                warm_start_tensor = static.data(
                    name='warm_start', shape=[-1, 1], dtype='bool'
                )

                train_fetches_loss = adaround.get_loss(
                    beta_tensor,
                    warm_start_tensor,
                    adaround_out_tensor,
                    orig_out_tensor,
                )
                optimizer = paddle.optimizer.Adam(learning_rate=lr)
                loss = train_fetches_loss['loss']
                optimizer.minimize(loss)
        exe.run(startup_program)

        start_time = time.time()
        prev_start_time = start_time
        for i, data in enumerate(data_loader()):
            prev_start_time = start_time
            start_time = time.time()
            # run fp32 model
            np_orig_out_tensor = exe.run(
                program=fp32_program,
                feed=data,
                fetch_list=[fp32_fetch_list],
                return_numpy=True,
                scope=scope,
            )

            adaround_weight_tensor_dequant = (
                adaround._calculate_adarounded_weights()
            )
            np_adaround_out_tensor = (
                adaround._calculate_output_with_adarounded_weights(
                    fp32_program,
                    place,
                    exe,
                    data,
                    fp32_fetch_list,
                    adaround_weight_tensor_dequant,
                )
            )

            # If the cosine distance of the two tensor is small, skip training
            cos_error = calculate_quant_cos_error(
                np_orig_out_tensor[0], np_adaround_out_tensor[0]
            )
            if fast_mode and cos_error > 0.99:
                _logger.info("The cosine error is small, skip training.")
                break
            beta, warm_start = adaround.update_beta_warm(i)
            feed_dict = {
                'orig_out_tensor': np_orig_out_tensor[0],
                'adaround_out_tensor': np_adaround_out_tensor[0],
                'beta': beta,
                'warm_start': warm_start,
            }
            out = exe.run(
                train_program,
                feed=feed_dict,
                fetch_list=[v.name for v in train_fetches_loss.values()],
                return_numpy=True,
            )
            _logger.info(
                f"Iter {i:d}, lr {lr:.5f}, loss {np.mean(out[0]):.5f}, loss_round {np.mean(out[1]):.5f}, loss_recon {np.mean(out[2]):.5f}, time {start_time - prev_start_time:.5f}s"
            )
            sys.stdout.flush()
            if i == num_iterations:
                break
        final_weight_tensor_quant_dict[
            weight_var_name
        ] = adaround.update_final_weights()

        if bias_correction:
            final_weight_tensor_quant_dict[weight_var_name] = bias_correction_w(
                weight_var_tensor,
                final_weight_tensor_quant_dict[weight_var_name],
                scale,
                adaround.quant_axis,
                weight_bits=adaround.weight_bits,
            )

        del adaround

    # update adarounded calibrated weights
    for weight_var_name in quantized_op_pairs.keys():
        set_variable_data(
            scope,
            place,
            weight_var_name,
            final_weight_tensor_quant_dict[weight_var_name],
        )
