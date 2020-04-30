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


class StOptimizer(fluid.optimizer.Optimizer):
    def __init__(self,
                 warmup_steps,
                 num_train_steps,
                 learning_rate,
                 weight_decay,
                 scheduler='linear_warmup_decay'):
        super(StOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=None,
            regularization=None,
            grad_clip=None,
            name=None)
        self.warmup_steps = warmup_steps
        self.num_train_steps = num_train_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler = scheduler

    def minimize(self, loss):

        train_program = fluid.default_main_program()
        startup_program = fluid.default_startup_program()

        if self.scheduler == 'noam_decay':
            if self.warmup_steps > 0:
                scheduled_lr = fluid.layers.learning_rate_scheduler\
                 .noam_decay(1/(self.warmup_steps *(self.learning_rate ** 2)),
                self.warmup_steps)
            else:
                print(
                    "WARNING: noam decay of learning rate should have postive warmup "
                    "steps but given {}, using constant learning rate instead!"
                    .format(self.warmup_steps))
                scheduled_lr = fluid.layers.create_global_var(
                    name=fluid.unique_name.generate("learning_rate"),
                    shape=[1],
                    value=self.learning_rate,
                    dtype='float32',
                    persistable=True)
        elif self.scheduler == 'linear_warmup_decay':
            if self.warmup_steps > 0:
                scheduled_lr = linear_warmup_decay(self.learning_rate,
                                                   self.warmup_steps,
                                                   self.num_train_steps)
            else:
                print(
                    "WARNING: linear warmup decay of learning rate should have "
                    "postive warmup steps but given {}, use constant learning rate "
                    "instead!".format(self.warmup_steps))
                scheduled_lr = fluid.layers.create_global_var(
                    name=fluid.unique_name.generate("learning_rate"),
                    shape=[1],
                    value=self.learning_rate,
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

        if self.weight_decay > 0:
            for param in train_program.all_parameters():
                param_list[param.name] = param * 1.0
                param_list[param.name].stop_gradient = True

        _, param_grads = optimizer.minimize(loss)

        if self.weight_decay > 0:
            for param, grad in param_grads:
                if exclude_from_weight_decay(param):
                    continue
                with param.block.program._optimized_guard(
                    [param, grad]), fluid.framework.name_scope("weight_decay"):
                    updated_param = param - param_list[
                        param.name] * self.weight_decay * scheduled_lr
                    fluid.layers.assign(output=param, input=updated_param)
