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

import control_flow
import nn
import ops
import tensor
from ..initializer import init_on_cpu

__all__ = [
    'exponential_decay', 'natural_exp_decay', 'inverse_time_decay',
    'polynomial_decay', 'piecewise_decay', 'noam_decay'
]
"""
When training a model, it's often useful to decay the
learning rate during training process, this is called
learning_rate_decay. There are many strategies to do
this, this module will provide some classical method.
User can also implement their own learning_rate_decay
strategy according to this module.
"""


def _decay_step_counter(begin=0):
    # the first global step is zero in learning rate decay
    global_step = nn.autoincreased_step_counter(
        counter_name='@LR_DECAY_COUNTER@', begin=begin, step=1)
    global_step = tensor.cast(global_step, 'float32')
    return global_step


def noam_decay(d_model, warmup_steps):
    """Apply decay to learning rate.
    ```python
    lr_value = np.power(d_model, -0.5) * np.min([
            np.power(current_steps, -0.5),
            np.power(warmup_steps, -1.5) * current_steps
        ])
    ```

    Args:
        d_model(Variable): The dimensionality of input and output of model.
            Reference: attention is all you need
                https://arxiv.org/pdf/1706.03762.pdf
        warmup_steps(Variable): A super parameter.

    Returns:
        The decayed learning rate.
    """
    global_step = _decay_step_counter(1)
    with init_on_cpu():
        a = global_step**-0.5
        b = (warmup_steps**-1.5) * global_step
        lr_value = (d_model**-0.5) * ops.elementwise_min(a, b)

    return lr_value


def exponential_decay(learning_rate, decay_steps, decay_rate, staircase=False):
    """Applies exponential decay to the learning rate.

    ```python
    decayed_learning_rate = learning_rate *
            decay_rate ^ (global_step / decay_steps)
    ```
    Args:
        learning_rate: A scalar float32 value or a Variable. This
          will be the initial learning rate during training
        decay_steps: A Python `int32` number.
        decay_rate: A Python `float` number.
        staircase: Boolean. If set true, decay the learning rate every decay_steps.

    Returns:
        The decayed learning rate
    """
    global_step = _decay_step_counter()

    with init_on_cpu():
        # update learning_rate
        div_res = global_step / decay_steps
        if staircase:
            div_res = ops.floor(div_res)
        decayed_lr = learning_rate * (decay_rate**div_res)

    return decayed_lr


def natural_exp_decay(learning_rate, decay_steps, decay_rate, staircase=False):
    """Applies natural exponential decay to the initial learning rate.

    >>> if not staircase:
    >>>     decayed_learning_rate = learning_rate * exp(- decay_rate * (global_step / decay_steps))
    >>> else:
    >>>     decayed_learning_rate = learning_rate * exp(- decay_rate * (global_step / decay_steps))

    Args:
        learning_rate: A scalar float32 value or a Variable. This
          will be the initial learning rate during training
        decay_steps: A Python `int32` number.
        decay_rate: A Python `float` number.
        staircase: Boolean. If set true, decay the learning rate every decay_steps.

    Returns:
        The decayed learning rate
    """
    global_step = _decay_step_counter()

    with init_on_cpu():
        div_res = global_step / decay_steps
        if staircase:
            div_res = ops.floor(div_res)
        decayed_lr = learning_rate * ops.exp(-1 * decay_rate * div_res)

    return decayed_lr


def inverse_time_decay(learning_rate, decay_steps, decay_rate, staircase=False):
    """Applies inverse time decay to the initial learning rate.

    >>> if staircase:
    >>>     decayed_learning_rate = learning_rate / (1 + decay_rate * floor(global_step / decay_step))
    >>> else:
    >>>     decayed_learning_rate = learning_rate / (1 + decay_rate * global_step / decay_step)

    Args:
        learning_rate: A scalar float32 value or a Variable. This
          will be the initial learning rate during training.
        decay_steps: A Python `int32` number.
        decay_rate: A Python `float` number.
        staircase: Boolean. If set true, decay the learning rate every decay_steps.

    Returns:
        The decayed learning rate
    """
    global_step = _decay_step_counter()

    with init_on_cpu():
        div_res = global_step / decay_steps
        if staircase:
            div_res = ops.floor(div_res)

        decayed_lr = learning_rate / (1 + decay_rate * div_res)

    return decayed_lr


def polynomial_decay(learning_rate,
                     decay_steps,
                     end_learning_rate=0.0001,
                     power=1.0,
                     cycle=False):
    """Applies polynomial decay to the initial learning rate.

    >>> if cycle:
    >>>     decay_steps = decay_steps * ceil(global_step / decay_steps)
    >>> else:
    >>>     global_step = min(global_step, decay_steps)
    >>> decayed_learning_rate = (learning_rate - end_learning_rate) *
    >>>                   (1 - global_step / decay_steps) ^ power +
    >>>                   end_learning_rate
    Args:
        learning_rate: A scalar float32 value or a Variable. This
          will be the initial learning rate during training
        decay_steps: A Python `int32` number.
        end_learning_rate: A Python `float` number.
        power: A Python `float` number
        cycle: Boolean. If set true, decay the learning rate every decay_steps.

    Returns:
        The decayed learning rate
    """
    global_step = _decay_step_counter()

    with init_on_cpu():
        if cycle:
            div_res = ops.ceil(global_step / decay_steps)
            zero_var = tensor.fill_constant(
                shape=[1], dtype='float32', value=0.0)
            one_var = tensor.fill_constant(
                shape=[1], dtype='float32', value=1.0)

            with control_flow.Switch() as switch:
                with switch.case(global_step == zero_var):
                    tensor.assign(input=one_var, output=div_res)
            decay_steps = decay_steps * div_res
        else:
            decay_steps_var = tensor.fill_constant(
                shape=[1], dtype='float32', value=float(decay_steps))
            global_step = ops.elementwise_min(x=global_step, y=decay_steps_var)

        decayed_lr = (learning_rate - end_learning_rate) * \
                     ((1 - global_step / decay_steps) ** power) + end_learning_rate
    return decayed_lr


def piecewise_decay(boundaries, values):
    """Applies piecewise decay to the initial learning rate.

    >>> boundaries = [10000, 20000]
    >>> values = [1.0, 0.5, 0.1]
    >>>
    >>> if step < 10000:
    >>>     learning_rate = 1.0
    >>> elif 10000 <= step < 20000:
    >>>     learning_rate = 0.5
    >>> else:
    >>>     learning_rate = 0.1
    """

    if len(values) - len(boundaries) != 1:
        raise ValueError("len(values) - len(boundaries) should be 1")

    global_step = _decay_step_counter()

    with init_on_cpu():
        lr = tensor.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=True,
            name="learning_rate")

        with control_flow.Switch() as switch:
            for i in range(len(boundaries)):
                boundary_val = tensor.fill_constant(
                    shape=[1], dtype='float32', value=float(boundaries[i]))
                value_var = tensor.fill_constant(
                    shape=[1], dtype='float32', value=float(values[i]))
                with switch.case(global_step < boundary_val):
                    tensor.assign(value_var, lr)
            last_value_var = tensor.fill_constant(
                shape=[1],
                dtype='float32',
                value=float(values[len(values) - 1]))
            with switch.default():
                tensor.assign(last_value_var, lr)

    return lr
