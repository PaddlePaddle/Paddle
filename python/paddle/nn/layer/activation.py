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

# TODO: define activation functions of neural network

__all__ = [
    #       'PReLU',
    'ReLU',
    'Sigmoid',
    #       'Softmax',
    'LogSoftmax',
    'HSigmoid'
]

from ...fluid.dygraph import layers
from ...fluid import core
from ...fluid.framework import in_dygraph_mode
from .. import functional


class HSigmoid(layers.Layer):
    """
	:alias_main: paddle.nn.HSigmoid
	:alias: paddle.nn.HSigmoid,paddle.nn.layer.HSigmoid,paddle.nn.layer.activation.HSigmoid

    Hierarchical Sigmoid Layer.
    
    The hierarchical sigmoid organizes the classes into a complete binary tree to reduce the computational complexity
    and speed up the model training, especially the training of language model.
    Each leaf node of the complete binary tree represents a class(word) and each non-leaf node acts as a binary classifier.
    For each class(word), there's a unique path from root to itself, hsigmoid calculate the cost for each non-leaf node on
    the path, and sum them to get a total cost.
    Comparing to softmax, the OP can reduce the computational complexity from :math:`O(N)` to :math:`O(logN)`, where :math:`N`
    represents the number of classes or the size of word dict.

    The OP supports default tree and custom tree. For the default tree, you can refer to `Hierarchical Probabilistic Neural
    Network Language Model <http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf>_`. For the custom
    tree, you need to set :attr:`is_custom` to True, and do the following steps (take the language model as an example):

    1. Using a custom word dict to build a binary tree, each leaf node should be an word in the word dict.
    2. Creating a dict map word_id -> path that from the word to the root node, we call it path_table.
    3. Creating a dict map word_id -> code of path that from the word to the root node, we call it path_code.
       Code means the label of each binary classifier, 1 indicate true, 0 indicate false.
    4. Now, each word should has its path and code along the path, you can pass a batch of path and code related
       to the same batch of inputs.

    Parameters:
        feature_size (int): The feature size.
        num_classes (int): The number of classes or the size of word dict, must be greater than 2.
            If the default tree is used (:attr:`is_custom` is set to False), :attr:`num_classes`
            should not be None. If the custom tree is used (:attr:`is_custom` is set to True),
            :attr:`num_classes` should be the number of non-leaf nodes, which indicates the num of
            classes using by the binary classifier.
        param_attr (ParamAttr, optional): The parameter attribute for the learnable parameters/weights
            of hsigmoid. If it is set to None or one attribute of ParamAttr, hsigmoid will create a
            ParamAttr as param_attr. If the Initializer of the param_attr is not set, the parameter is
            initialized with Xavier. Default: None.
        bias_attr (ParamAttr|bool, optional): The parameter attribute for the bias of hsigmoid. If it
            is set to False, no bias will be added. If it is set to None or one attribute of ParamAttr,
            hsigmoid will create a ParamAttr as bias_attr. If the Initializer of the bias_attr is not
            set, the bias is initialized zero. Default: None.
        is_custom (bool, optional): Whether use custom binary tree. If it's True, `path_table` and 
            `path_code` should be passed to its forward method, otherwise `path_table` and `path_code`
            should not be passed to its forward method. Default: False.
        is_sparse (bool, optional): Whether use sparse updating instead of dense updating, if it's True, the
            gradient of W and input will be sparse. Default: False.

    Returns:
        None

    Examples:
        .. code-block:: python

          from paddle import fluid, nn
          import paddle.fluid.dygraph as dg
          import paddle.nn.functional as F
          import numpy as np

          main = fluid.Program()
          start = fluid.Program()
          feature_size = 6
          num_classes = 8
          with fluid.unique_name.guard():
              with fluid.program_guard(main, start):
                  x = fluid.data("input", [-1, feature_size],
                              dtype="float32")
                  label = fluid.data("labels", [-1, 1], dtype="int64")
                  hsm = nn.HSigmoid(feature_size, num_classes)
                  y = hsm(x, label)

          place = fluid.CPUPlace()
          exe = fluid.Executor(place)
          exe.run(start)
          feed_dict = {
              "input": np.random.randn(4, feature_size).astype(np.float32),
              "labels": np.random.randint(0, num_classes, (4, 1)).astype(np.int64),
          }
          y_np, = exe.run(main, feed=feed_dict, fetch_list=[y])
          print(y_np.shape)

          # (4, 1)
    """

    def __init__(self,
                 feature_size,
                 num_classes,
                 param_attr=None,
                 bias_attr=None,
                 is_custom=False,
                 is_sparse=False,
                 dtype="float32"):
        super(HSigmoid, self).__init__()
        if (num_classes < 2) and (not is_custom):
            raise ValueError(
                "num_classes must not be less than 2 with default tree")

        if (not is_custom) and (is_sparse):
            print("Sparse mode should not be used without custom tree")
            is_sparse = False

        self._feature_size = feature_size
        self._num_classes = num_classes
        self._is_custom = is_custom
        self._is_sparse = is_sparse

        self._param_attr = param_attr
        self._bias_attr = bias_attr

        self._dtype = dtype

        remote_prefetch = is_sparse
        print("With sparse mode, if your models has only"
              " small parameter prefetch may cause speed down")

        C = self._num_classes if is_custom else self._num_classes - 1
        self.weight = self.create_parameter(
            [C, self._feature_size],
            attr=self._param_attr,
            is_bias=False,
            dtype=self._dtype)
        self.bias = self.create_parameter(
            [C, 1], attr=self._bias_attr, is_bias=True, dtype=self._dtype)

    def forward(self, input, label, path_table=None, path_code=None):
        out = functional.hsigmoid(
            input,
            label,
            self.weight,
            self.bias,
            self._num_classes,
            path_table=path_table,
            path_code=path_code,
            is_sparse=self._is_sparse)
        return out


class ReLU(layers.Layer):
    """
	:alias_main: paddle.nn.ReLU
	:alias: paddle.nn.ReLU,paddle.nn.layer.ReLU,paddle.nn.layer.activation.ReLU

    ReLU Activation.

    .. math:

        out = max(x, 0)

    Parameters:
        inplace (bool, optional): If inplace is True, the input and output of 
            ``ReLU`` are the same variable. Otherwise, the input and output of
            ``ReLU`` are different variables. Default False. Note that if x is
            more than one OPs' input, inplace must be False.
    
    Returns:
        None
    
    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import paddle.nn as nn
          import numpy as np

          data = np.array([-2, 0, 1]).astype('float32')
          my_relu = nn.ReLU()
          with fluid.dygraph.guard():
              data = fluid.dygraph.to_variable(data)
              res = my_relu(data)  # [0, 0, 1]
    """

    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self._inplace = inplace

    def forward(self, input):
        return functional.relu(input, self._inplace)


class Sigmoid(layers.Layer):
    """
	:alias_main: paddle.nn.Sigmoid
	:alias: paddle.nn.Sigmoid,paddle.nn.layer.Sigmoid,paddle.nn.layer.activation.Sigmoid

    Sigmoid Activation.
    
    .. math:

        output = \frac{1}{1 + e^{-input}}

    Parameters:
        inplace (bool, optional): If inplace is True, the input and output
            are the same variable. Otherwise, the input and output
            are different variables. Default False. Note that if x is
            more than one OPs' input, inplace must be False.
    
    Returns:
        None
    
    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import paddle.nn as nn
          import numpy as np
          input = fluid.data(name="input", shape=[None, 4])
          output = nn.Sigmoid()(input)
          place = fluid.CPUPlace()
          exe = fluid.Executor(place)
          exe.run(fluid.default_startup_program())
          input_data = np.array([1.0, 2.0, 3.0, 4.0]).astype('float32')
          output_data = exe.run(feed={"input": input_data},
                                fetch_list=[output])
          print(output_data) # [0.7310586, 0.880797, 0.95257413, 0.98201376]
    """

    def __init__(self, inplace=False):
        super(Sigmoid, self).__init__()
        self._inplace = inplace

    def forward(self, input):
        return functional.sigmoid(input, self._inplace)


class LogSoftmax(layers.Layer):
    """
	:alias_main: paddle.nn.LogSoftmax
	:alias: paddle.nn.LogSoftmax,paddle.nn.layer.LogSoftmax,paddle.nn.layer.activation.LogSoftmax

    This operator implements the log_softmax layer. The calculation process is as follows:

    .. math::

        Out[i, j] = log(softmax(x)) 
                  = log(\\frac{\exp(X[i, j])}{\sum_j(exp(X[i, j])})

    Parameters:
        axis (int, optional): The index of dimension to perform softmax calculations, it should be in
            range :math:`[-1, rank-1]`, while :math:`rank` is the rank of input variable. Default: None. 
            None and -1 means the last dimension.
        dtype (np.dtype|core.VarDesc.VarType|str): The desired data type of returned tensor. If specified,
            the input tensor is casted to dtype before the operation is performed. This is useful for
            preventing data type overflows. Default: None. Supported dtype: float32 or float64
 
    Returns:
        None

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import paddle.nn as nn
          import numpy as np

          data = np.array([[[-2.0, 3.0, -4.0, 5.0],
                            [3.0, -4.0, 5.0, -6.0],
                            [-7.0, -8.0, 8.0, 9.0]],
                           [[1.0, -2.0, -3.0, 4.0],
                            [-5.0, 6.0, 7.0, -8.0],
                            [6.0, 7.0, 8.0, 9.0]]]).astype('float32')
          my_log_softnmax = nn.LogSoftmax()
          with fluid.dygraph.guard():
              data = fluid.dygraph.to_variable(data)
              res = my_log_softnmax(data)
              # [[[ -7.1278396   -2.1278396   -9.127839    -0.12783948]
              #   [ -2.1270514   -9.127051    -0.12705144 -11.127051  ]
              #   [-16.313261   -17.313261    -1.3132617   -0.31326184]]
              #  [[ -3.0518122   -6.051812    -7.051812    -0.051812  ]
              #   [-12.313267    -1.3132664   -0.3132665  -15.313267  ]
              #   [ -3.4401896   -2.4401896   -1.4401896   -0.44018966]]]
    """

    def __init__(self, axis=None):
        super(LogSoftmax, self).__init__()
        self._axis = axis

    def forward(self, input):
        return functional.log_softmax(input, self._axis)
