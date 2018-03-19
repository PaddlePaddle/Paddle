#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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


def lrn(input, n=5, k=1.0, alpha=1e-4, beta=0.75, name=None):
    """
    **Local Response Normalization Operator**

    Refer to `ImageNet Classification with Deep Convolutional Neural Networks 
    <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_

    The formula is as follows:

    .. math::

        Output(i, x, y) = Input(i, x, y) / \left(
        k + \alpha \sum\limits^{\min(C, c + n/2)}_{j = \max(0, c - n/2)}
        (Input(j, x, y))^2 \right)^{\beta}

    In the above equation:

    * :math:`n`: The number of channels to sum over.
    * :math:`k`: The offset (usually positive to avoid dividing by 0).
    * :math:`alpha`: The scaling parameter.
    * :math:`beta`: The exponent.

    Args:
        input (Variable): The input tensor of this layer. The dimension of input tensor must be 4 and it's order should be 'NCHW'.
        n (int, default 5): The number of channels to sum over.
        k (float, default 1.0): An offset (usually positive to avoid dividing by 0).
        alpha (float, default 1e-4): The scaling parameter.
        beta (float, default 0.75): The exponent.
        name (str, default None): A name for this operation.

    Raises:
        ValueError: If rank of the input tensor is not 4.

    Returns:
        A tensor variable storing the transformation result.

    Examples:
        .. code-block:: python

          data = fluid.layers.data(name="data", shape=[3, 112, 112], dtype="float32")
          lrn = fluid.layers.lrn(input=data)
    """
