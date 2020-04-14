#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Optimization and learning rate scheduling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle.fluid as fluid
from utils.fp16 import create_master_params_grads, master_param_to_train_param, apply_dynamic_loss_scaling


def linear_warmup_decay(learning_rate, warmup_steps, num_train_steps):
    """ Applies linear warmup of learning rate from 0 and decay to 0."""
    with fluid.default_main_program()._lr_schedule_guard():
        lr = fluid.layers.tensor.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=True,
            name="scheduled_learning_rate")

        global_step = fluid.layers.learning_rate_scheduler._decay_step_counter(
        )

        with fluid.layers.control_flow.Switch() as switch:
            with switch.case(global_step < warmup_steps):
                warmup_lr = learning_rate * (global_step / warmup_steps)
                fluid.layers.tensor.assign(warmup_lr, lr)
            with switch.default():
                decayed_lr = fluid.layers.learning_rate_scheduler.polynomial_decay(
                    learning_rate=learning_rate,
                    decay_steps=num_train_steps,
                    end_learning_rate=0.0,
                    power=1.0,
                    cycle=False)
                fluid.layers.tensor.assign(decayed_lr, lr)

        return lr


def optimization(loss,
                 warmup_steps,
                 num_train_steps,
                 learning_rate,
                 train_program,
                 startup_prog,
                 weight_decay,
                 scheduler='linear_warmup_decay',
                 use_fp16=False,
                 use_dynamic_loss_scaling=False,
                 init_loss_scaling=1.0,
                 incr_every_n_steps=1000,
                 decr_every_n_nan_or_inf=2,
                 incr_ratio=2.0,
                 decr_ratio=0.8):

    scheduled_lr, loss_scaling = None, None
    if scheduler == 'noam_decay':
        if warmup_steps > 0:
            scheduled_lr = fluid.layers.learning_rate_scheduler\
             .noam_decay(1/(warmup_steps *(learning_rate ** 2)),
           warmup_steps)
        else:
            print(
                "WARNING: noam decay of learning rate should have postive warmup "
                "steps but given {}, using constant learning rate instead!"
                .format(warmup_steps))
            scheduled_lr = fluid.layers.create_global_var(
                name=fluid.unique_name.generate("learning_rate"),
                shape=[1],
                value=learning_rate,
                dtype='float32',
                persistable=True)
    elif scheduler == 'linear_warmup_decay':
        if warmup_steps > 0:
            scheduled_lr = linear_warmup_decay(learning_rate, warmup_steps,
                                               num_train_steps)
        else:
            print(
                "WARNING: linear warmup decay of learning rate should have "
                "postive warmup steps but given {}, use constant learning rate "
                "instead!".format(warmup_steps))
            scheduled_lr = fluid.layers.create_global_var(
                name=fluid.unique_name.generate("learning_rate"),
                shape=[1],
                value=learning_rate,
                dtype='float32',
                persistable=True)
    else:
        raise ValueError("Unkown learning rate scheduler, should be "
                         "'noam_decay' or 'linear_warmup_decay'")

    optimizer = fluid.optimizer.Adam(learning_rate=scheduled_lr)
    fluid.clip.set_gradient_clip(
        clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0))

    def exclude_from_weight_decay(param):
        name = param.name.rstrip(".master")
        if name.find("layer_norm") > -1:
            return True
        bias_suffix = ["_bias", "_b", ".b_0"]
        for suffix in bias_suffix:
            if name.endswith(suffix):
                return True
        return False

    param_list = dict()

    if use_fp16:
        loss_scaling = fluid.layers.create_global_var(
            name=fluid.unique_name.generate("loss_scaling"),
            shape=[1],
            value=init_loss_scaling,
            dtype='float32',
            persistable=True)
        loss *= loss_scaling

        param_grads = optimizer.backward(loss)
        master_param_grads = create_master_params_grads(
            param_grads, train_program, startup_prog, loss_scaling)

        if weight_decay > 0:
            for param, _ in master_param_grads:
                param_list[param.name] = param * 1.0
                param_list[param.name].stop_gradient = True

        if use_dynamic_loss_scaling:
            apply_dynamic_loss_scaling(
                loss_scaling, master_param_grads, incr_every_n_steps,
                decr_every_n_nan_or_inf, incr_ratio, decr_ratio)

        optimizer.apply_gradients(master_param_grads)

        if weight_decay > 0:
            for param, grad in master_param_grads:
                if exclude_from_weight_decay(param):
                    continue
                with param.block.program._optimized_guard(
                    [param, grad]), fluid.framework.name_scope("weight_decay"):
                    updated_param = param - param_list[
                        param.name] * weight_decay * scheduled_lr
                    fluid.layers.assign(output=param, input=updated_param)

        master_param_to_train_param(master_param_grads, param_grads,
                                    train_program)

    else:
        if weight_decay > 0:
            for param in train_program.all_parameters():
                param_list[param.name] = param * 1.0
                param_list[param.name].stop_gradient = True

        _, param_grads = optimizer.minimize(loss)

        if weight_decay > 0:
            for param, grad in param_grads:
                if exclude_from_weight_decay(param):
                    continue
                with param.block.program._optimized_guard(
                    [param, grad]), fluid.framework.name_scope("weight_decay"):
                    updated_param = param - param_list[
                        param.name] * weight_decay * scheduled_lr
                    fluid.layers.assign(output=param, input=updated_param)

    return scheduled_lr, loss_scaling
