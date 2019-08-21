#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from .layer_helper import LayerHelper

__all__ = ['huber_loss']


def huber_loss(input, label, delta):
    """
    Huber loss is a loss function used in robust.
    Huber loss can evaluate the fitness of input to label.
    Different from MSE loss, Huber loss is more robust for outliers.

    When the difference between input and label is large than delta
    .. math::

        huber\_loss = delta * (label - input) - 0.5 * delta * delta

    When the difference between input and label is less than delta
    .. math::

        huber\_loss = 0.5 * (label - input) * (label - input)


    Args:
        input (Variable): This input is a probability computed by the previous operator.
                          The first dimension is batch size, and the last dimension is 1.
        label (Variable): The groud truth whose first dimension is batch size
                          and last dimension is 1.
        delta (float): The parameter of huber loss, which controls
                       the range of outliers

    Returns:
        huber\_loss (Variable): The huber loss with shape [batch_size, 1].

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            x = fluid.layers.data(name='x', shape=[13], dtype='float32')
            predict = fluid.layers.fc(input=x, size=1)
            label = fluid.layers.data(
                name='label', shape=[1], dtype='float32')
            loss = fluid.layers.huber_loss(
                input=predict, label=label, delta=1.0)

    """
    helper = LayerHelper('huber_loss', **locals())
    residual = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    helper.append_op(
        type='huber_loss_v2',
        inputs={'X': input,
                'Y': label},
        outputs={'Out': out,
                 'Residual': residual},
        attrs={'delta': delta})
    return out
