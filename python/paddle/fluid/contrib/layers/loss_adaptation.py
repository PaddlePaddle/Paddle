# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""
Contrib layers just related to loss.
"""

from __future__ import print_function

import warnings
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.initializer import Normal, Constant
from paddle.fluid.framework import Variable
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid import layers
import paddle

__all__ = ['loss_adaptation']


def loss_adaptation(input, last_shared_layer):
    """
    multi-task loss weighted layer

    This function help balance eacn loss of multi-task.

    Args:
        input(list[Variable]): A list of Variable, This
                         Variable indicates the loss of each task.
        last_shared_layer(Variable): A Variable indicating the last shared layer.
    Returns:
        loss(Variable): the final loss of tasks.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            data = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
            loss1 = fluid.data(name='loss1', shape=[None, 1], dtype='float32')
            loss2 = fluid.data(name='loss2', shape=[None, 1], dtype='float32')
            loss3 = fluid.data(name='loss3', shape=[None, 1], dtype='float32')
            fc = fluid.layers.fc(input=data, size=1000, act="tanh")
            loss_out = fluid.contrib.layers.loss_adaptation(input=[loss1, loss2, loss3], last_shared_layer=fc)
    """
    assert len(input) > 0
    helper = LayerHelper("loss_adaptation", **locals())

    #get grad
    grad_list = []
    for per_input in input:
        var = paddle.static.gradients(per_input, last_shared_layer)
        grad_list.append(var)

    loss_w = []
    #comput loss weight 
    for grad in grad_list:
        limit_v = paddle.full_like(grad[0], fill_value=0.000001)
        w_ = layers.reciprocal(paddle.add(grad[0], limit_v))
        loss_w.append(w_)

    #compute the final loss
    loss = []
    for i in range(len(loss_w)):
        loss_ = paddle.multiply(paddle.mean(loss_w[i]), paddle.mean(input[i]))
        loss.append(loss_)
    loss_concat = paddle.concat(x=loss, axis=0)
    return paddle.sum(loss_concat, axis=0)
