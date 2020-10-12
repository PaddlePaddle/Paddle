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

import paddle
from paddle.fluid.framework import static_only

__all__ = ['fc']


@static_only
def fc(x,
       size,
       num_flatten_dims=1,
       weight_attr=None,
       bias_attr=None,
       activation=None,
       name=None):
    """

    Fully-Connected layer can take a tensor or a list of tensor as its inputs.
    It creates a 2-D weight tensor for each input tensor, which represents its
    weight matrix from each input unit to each output unit. The fully connected
    layer multiplies each input tensor with its corresponding weight to produce
    an output tensor with shape :math:`[batch\_size, *, size]` , where :math:`*`
    means any number of additional dimensions. If a list of tensor is given,
    the results of multiple output tensors with shape :math:`[batch\_size, *, size]`
    will be summed up. If :attr:`bias_attr` is not False, a 1-D bias tensor will
    be created and added to the output. Finally, if :attr:`activation` is not None,
    it will be applied to the output as well.

    For a single input tensor :math:`X` , the equation is:

    .. math::

        Out = Act({XW + b})

    For a list of input tensor, the equation is:

    .. math::

        Out = Act({\sum_{i=0}^{N-1}X_iW_i + b})

    where:

    * :math:`N`: The number of the input tensors. :math:`N` equals to :math:`len(X)` if :math:`X` is list of tensor.
    * :math:`X_i`: The i-th input tensor.
    * :math:`W_i`: The i-th weight matrix corresponding i-th input tensor.
    * :math:`b`: The bias created by this layer (if needed).
    * :math:`Act`: The activation function.
    * :math:`Out`: The output tensor.

    .. code-block:: text

        # Case 1, input is a single tensor:
        x.data = [[[0.1, 0.2],
                   [0.3, 0.4]]]
        x.shape = (1, 2, 2) # 1 is batch_size

        out = paddle.static.nn.fc(x=x, size=1, num_flatten_dims=2)

        # Get the output:
        out.data = [[0.83234344], [0.34936576]]
        out.shape = (1, 2, 1)

        # Case 2, input is a list of tensor:
        x0.data = [[[0.1, 0.2],
                    [0.3, 0.4]]]
        x0.shape = (1, 2, 2) # 1 is batch_size

        x1.data = [[[0.1, 0.2, 0.3]]]
        x1.shape = (1, 1, 3)

        out = paddle.static.nn.fc(x=[x0, x1], size=2)

        # Get the output:
        out.data = [[0.18669507, 0.1893476]]
        out.shape = (1, 2)

    Args:
        x (Tensor|list of Tensor): A tensor or a list of tensor. The number of dimensions
            of each tensor is at least 2. The data type should be float16, float32 or float64.
        size (int): The number of output units in this layer, which also means the feature
            size of output tensor.
        num_flatten_dims (int, optional): The fc layer can accept an input tensor with more than
            two dimensions. If this happens, the multi-dimensional tensor will first be flattened
            into a 2-D matrix. The parameter :attr:`num_flatten_dims` determines how the input
            tensor is flattened: the first :math:`num\_flatten\_dims` (inclusive, index starts from 1)
            dimensions will be flatten to form the first dimension of the final matrix (height of
            the matrix), and the rest :math:`rank(x) - num\_flatten\_dims` dimensions are
            flattened to form the second dimension of the final matrix (width of the matrix).
            For example, assuming that :attr:`x` is a 5-dimensional tensor with a shape
            :math:`[2, 3, 4, 5, 6]` , and :attr:`num_flatten_dims` = 3.
            Then, the flattened matrix will have a shape :math:`[2 * 3 * 4, 5 * 6] = [24, 30]` .
            Default: 1.
        weight_attr (ParamAttr, optional): The attribute for the learnable weight.
            The default value is None, and the weight will be initialized to zero.
            For detailed information, please refer to :attr:`paddle.ParamAttr`.
        bias_attr (ParamAttr|bool, optional): The attribute of the learnable bias. 
            If it is set to False, no bias will be added to the output.
            If it is set to None or one kind of ParamAttr, a bias parameter will
            be created according to ParamAttr. For detailed information, please refer
            to :attr:`paddle.ParamAttr`. The default value is None and the bias will be
            initialized to zero. 
        activation (str, optional): Activation to be applied to the output of
            this layer, such as tanh, softmax, sigmoid, relu. For more information,
            please refer to :ref:`api_guide_activations_en` . Default: None.
        name (str, optional): The default value is None. Normally there is no need for user to set
            it. For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Tensor, its shape is :math:`[batch\_size, *, size]` , and the data type is same with input.

    Raises:
        ValueError: If dimensions of the input tensor is less than 2.

    Examples:
        .. code-block:: python

          import paddle
          paddle.enable_static()

          # When input is a single tensor
          x = paddle.static.data(name="x", shape=[1, 2, 2], dtype="float32")
          # x: [[[0.1 0.2]
          #      [0.3 0.4]]]
          out = paddle.static.nn.fc(
              x=x,
              size=1,
              num_flatten_dims=2,
              weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.5)),
              bias_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0)))
          # out: [[[1.15]
          #        [1.35]]]

          # When input is multiple tensors
          x0 = paddle.static.data(name="x0", shape=[1, 2, 2], dtype="float32")
          # x0: [[[0.1 0.2]
          #       [0.3 0.4]]]
          x1 = paddle.static.data(name="x1", shape=[1, 1, 3], dtype="float32")
          # x1: [[[0.1 0.2 0.3]]]
          out = paddle.static.nn.fc(
              x=[x0, x1],
              size=2,
              weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.5)),
              bias_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0)))
          # out: [[1.8 1.8]]
    """
    return paddle.fluid.layers.fc(input=x,
                                  size=size,
                                  num_flatten_dims=num_flatten_dims,
                                  param_attr=weight_attr,
                                  bias_attr=bias_attr,
                                  act=activation,
                                  name=name)
