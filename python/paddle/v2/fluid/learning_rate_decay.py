# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

import layers
from framework import Variable

__all__ = ['exponential_decay', ]
"""
When training a model, it's often useful to decay the
learning rate during training process, this is called
learning_rate_decay. There are many strategies to do
this, this module will provide some classic method.
User can also implement their own learning_rate_decay
strategy according to this module.
"""


def exponential_decay(learning_rate,
                      global_step,
                      decay_steps,
                      decay_rate,
                      staircase=False):
    """Applies exponential decay to the learning rate.

    ```python
    decayed_learning_rate = learning_rate *
            decay_rate ^ (global_step / decay_steps)
    ```
    Args:
        learning_rate: A scalar float32 value or a Variable. This
          will be the initial learning rate during training
        global_step: A Variable that record the training step.
        decay_steps: A Python `int32` number.
        decay_rate: A Python `float` number.
        staircase: Boolean. If set true, decay the learning rate every decay_steps.

    Returns:
        The decayed learning rate
    """
    if global_step is None or not isinstance(global_step, Variable):
        raise ValueError("global_step is required for exponential_decay.")

    lr = learning_rate
    if isinstance(lr, float):
        lr = layers.create_global_var(
            shape=[1], value=learning_rate, dtype='float32')
    elif isinstance(lr, Variable):
        lr = learning_rate
    else:
        raise ValueError("unsupported learning rate type : %s",
                         type(learning_rate))

    decay_steps_var = layers.fill_constant(
        shape=[1], dtype='float32', value=float(decay_steps))
    decay_rate_var = layers.fill_constant(
        shape=[1], dtype='float32', value=float(decay_rate))

    # update learning_rate
    div_res = global_step / decay_steps_var
    if staircase:
        div_res = layers.floor(x=div_res)
    return lr * (decay_rate_var**div_res)


def inverse_time_decay(learning_rate,
                       global_step,
                       decay_steps,
                       decay_rate,
                       staircase=False):
    """Applies inverse time decay to the initial learning rate.

    ```python
    if staircase:
      decayed_learning_rate = learning_rate / (1 + decay_rate * floor(global_step / decay_step))
    else
      decayed_learning_rate = learning_rate / (1 + decay_rate * global_step / decay_step)
    ```
    Args:
        learning_rate: A scalar float32 value or a Variable. This
          will be the initial learning rate during training
        global_step: A Variable that record the training step.
        decay_steps: A Python `int32` number.
        decay_rate: A Python `float` number.
        staircase: Boolean. If set true, decay the learning rate every decay_steps.

    Returns:
        The decayed learning rate
    """
    if global_step is None or not isinstance(global_step, Variable):
        raise ValueError("global_step is required for exponential_decay.")

    lr = learning_rate
    if isinstance(lr, float):
        lr = layers.create_global_var(
            shape=[1], value=learning_rate, dtype='float32')
    elif isinstance(lr, Variable):
        lr = learning_rate
    else:
        raise ValueError("unsupported learning rate type : %s",
                         type(learning_rate))

    decay_steps_var = layers.fill_constant(
        shape=[1], dtype='float32', value=float(decay_steps))
    decay_rate_var = layers.fill_constant(
        shape=[1], dtype='float32', value=float(decay_rate))
    div_res = global_step / decay_steps_var
    if staircase:
        div_res = layers.floor(x=div_res)
    const_one = layers.fill_constant(shape=[1], dtype='float32', value=float(1))

    return lr / (const_one + decay_rate_var * div_res)
