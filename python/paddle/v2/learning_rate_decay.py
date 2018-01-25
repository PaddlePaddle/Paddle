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

import paddle.v2.fluid as fluid
import paddle.v2.fluid.layers as layers


def exponential_decay(learning_rate, global_step, decay_steps, decay_rate):
    """
    ```python
    decayed_learning_rate = learning_rate *
            decay_rate ^ (global_step / decay_steps)
    ```
    """
    if global_step is None or not isinstance(global_step, fluid.Variable):
        raise ValueError("global_step is required for exponential_decay.")

    decay_steps_var = layers.fill_constant(
        shape=[1], dtype='int64', value=decay_steps)
    decay_rate_var = layers.fill_constant(
        shape=[1], dtype='float64', value=decay_rate)

    # update learning_rate
    div_res = layers.elementwise_div(X=global_step, Y=decay_steps_var)
    pow_res = fluid.layers.pow(X=decay_rate_var, Y=div_res)
    layers.assign(learning_rate, learning_rate * pow_res)
