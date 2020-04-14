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

from paddle.fluid.dygraph.learning_rate_scheduler import LearningRateDecay


class ConstantLR(LearningRateDecay):
    def __init__(self, learning_rate, begin=0, step=1, dtype='float32'):
        super(ConstantLR, self).__init__(begin, step, dtype)
        self.learning_rate = learning_rate

    def step(self):
        return self.learning_rate


class LinearDecay(LearningRateDecay):
    def __init__(self,
                 learning_rate,
                 warmup_steps,
                 decay_steps,
                 end_learning_rate=0.0001,
                 power=1.0,
                 cycle=False,
                 begin=0,
                 step=1,
                 dtype='float32'):
        super(LinearDecay, self).__init__(begin, step, dtype)
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle

    def step(self):
        if self.step_num < self.warmup_steps:
            decayed_lr = self.learning_rate * (self.step_num /
                                               self.warmup_steps)
            decayed_lr = self.create_lr_var(decayed_lr)
        else:
            tmp_step_num = self.step_num
            tmp_decay_steps = self.decay_steps
            if self.cycle:
                div_res = fluid.layers.ceil(
                    self.create_lr_var(tmp_step_num / float(self.decay_steps)))
                if tmp_step_num == 0:
                    div_res = self.create_lr_var(1.0)
                tmp_decay_steps = self.decay_steps * div_res
            else:
                tmp_step_num = self.create_lr_var(
                    tmp_step_num
                    if tmp_step_num < self.decay_steps else self.decay_steps)
                decayed_lr = (self.learning_rate - self.end_learning_rate) * \
                    ((1 - tmp_step_num / tmp_decay_steps) ** self.power) + self.end_learning_rate

        return decayed_lr


class Optimizer(object):
    def __init__(self,
                 warmup_steps,
                 num_train_steps,
                 learning_rate,
                 model_cls,
                 weight_decay,
                 scheduler='linear_warmup_decay',
                 loss_scaling=1.0,
                 parameter_list=None):
        self.warmup_steps = warmup_steps
        self.num_train_steps = num_train_steps
        self.learning_rate = learning_rate
        self.model_cls = model_cls
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.loss_scaling = loss_scaling
        self.parameter_list = parameter_list

        self.scheduled_lr = 0.0
        self.optimizer = self.lr_schedule()

    def lr_schedule(self):
        if self.warmup_steps > 0:
            if self.scheduler == 'noam_decay':
                self.scheduled_lr = fluid.dygraph.NoamDecay(1 / (
                    self.warmup_steps * (self.learning_rate**2)),
                                                            self.warmup_steps)
            elif self.scheduler == 'linear_warmup_decay':
                self.scheduled_lr = LinearDecay(self.learning_rate,
                                                self.warmup_steps,
                                                self.num_train_steps, 0.0)
            else:
                raise ValueError("Unkown learning rate scheduler, should be "
                                 "'noam_decay' or 'linear_warmup_decay'")
            optimizer = fluid.optimizer.Adam(
                learning_rate=self.scheduled_lr,
                parameter_list=self.parameter_list)
        else:
            self.scheduled_lr = ConstantLR(self.learning_rate)
            optimizer = fluid.optimizer.Adam(
                learning_rate=self.scheduled_lr,
                parameter_list=self.parameter_list)

        return optimizer

    def exclude_from_weight_decay(self, name):
        if name.find("layer_norm") > -1:
            return True
        bias_suffix = ["_bias", "_b", ".b_0"]
        for suffix in bias_suffix:
            if name.endswith(suffix):
                return True
        return False

    def minimize(self, loss, use_data_parallel=False, model=None):
        param_list = dict()

        clip_norm_thres = 1.0
        #grad_clip = fluid.clip.GradientClipByGlobalNorm(clip_norm_thres)

        if use_data_parallel:
            loss = model.scale_loss(loss)

        loss.backward()

        if self.weight_decay > 0:
            for param in self.model_cls.parameters():
                param_list[param.name] = param * 1.0
                param_list[param.name].stop_gradient = True

        if use_data_parallel:
            assert model is not None
            model.apply_collective_grads()

        #_, param_grads = self.optimizer.minimize(loss, grad_clip=grad_clip)
        _, param_grads = self.optimizer.minimize(loss)

        if self.weight_decay > 0:
            for param, grad in param_grads:
                if self.exclude_from_weight_decay(param.name):
                    continue
                if isinstance(self.scheduled_lr.step(), float):
                    updated_param = param.numpy() - param_list[
                        param.name].numpy(
                        ) * self.weight_decay * self.scheduled_lr.step()
                else:
                    updated_param = param.numpy(
                    ) - param_list[param.name].numpy(
                    ) * self.weight_decay * self.scheduled_lr.step().numpy()
                updated_param_var = fluid.dygraph.to_variable(updated_param)
                param = updated_param_var
                #param = fluid.layers.reshape(x=updated_param_var, shape=list(updated_param_var.shape))
