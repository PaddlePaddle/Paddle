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
from ...fluid.layers import brelu  #DEFINE_ALIAS
from ...fluid.layers import elu  #DEFINE_ALIAS
from ...fluid.layers import erf  #DEFINE_ALIAS
from ...fluid.layers import gelu  #DEFINE_ALIAS
from ...fluid.layers import hard_shrink  #DEFINE_ALIAS
from ...fluid.layers import hard_sigmoid  #DEFINE_ALIAS
from ...fluid.layers import hard_swish  #DEFINE_ALIAS
from ...fluid.layers import leaky_relu  #DEFINE_ALIAS
from ...fluid.layers import logsigmoid  #DEFINE_ALIAS
from ...fluid.layers import maxout  #DEFINE_ALIAS
from ...fluid.layers import relu6  #DEFINE_ALIAS
from ...fluid.layers import selu  #DEFINE_ALIAS
from ...fluid.layers import soft_relu  #DEFINE_ALIAS
from ...fluid.layers import softmax  #DEFINE_ALIAS
from ...fluid.layers import softplus  #DEFINE_ALIAS
from ...fluid.layers import softshrink  #DEFINE_ALIAS
from ...fluid.layers import softsign  #DEFINE_ALIAS
from ...fluid.layers import swish  #DEFINE_ALIAS
from ...fluid.layers import tanh_shrink  #DEFINE_ALIAS
from ...fluid.layers import thresholded_relu  #DEFINE_ALIAS

__all__ = [
    'brelu',
    'elu',
    'erf',
    'gelu',
    'hard_shrink',
    'hard_sigmoid',
    'hard_swish',
    'hsigmoid',
    'leaky_relu',
    'logsigmoid',
    'maxout',
    #       'prelu',
    'relu',
    'relu6',
    'selu',
    'sigmoid',
    'soft_relu',
    'softmax',
    'softplus',
    'softshrink',
    'softsign',
    'swish',
    'tanh_shrink',
    'thresholded_relu',
    'log_softmax'
]

import warnings
from ...fluid.layer_helper import LayerHelper
from ...fluid.framework import in_dygraph_mode, convert_np_dtype_to_dtype_
from ...fluid import core
from ...fluid.data_feeder import check_variable_and_dtype


def hsigmoid(input,
             label,
             weight,
             bias,
             num_classes,
             path_table=None,
             path_code=None,
             is_sparse=False):
    """
	:alias_main: paddle.nn.functional.hsigmoid
	:alias: paddle.nn.functional.hsigmoid,paddle.nn.functional.activation.hsigmoid

    The hierarchical sigmoid organizes the classes into a complete binary tree to reduce the computational complexity
    and speed up the model training, especially the training of language model.
    Each leaf node of the complete binary tree represents a class(word) and each non-leaf node acts as a binary classifier.
    For each class(word), there's a unique path from root to itself, hsigmoid calculate the cost for each non-leaf node on
    the path, and sum them to get a total cost.
    Comparing to softmax, the OP can reduce the computational complexity from :math:`O(N)` to :math:`O(logN)`, where :math:`N`
    represents the number of classes or the size of word dict.

    The OP supports default tree and custom tree. For the default tree, you can refer to `Hierarchical Probabilistic Neural
    Network Language Model <http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf>`_. For the custom
    tree, you need to set :attr:`is_custom` to True, and do the following steps (take the language model as an example):

    1. Using a custom word dict to build a binary tree, each leaf node should be an word in the word dict.
    2. Creating a dict map word_id -> path that from the word to the root node, we call it path_table.
    3. Creating a dict map word_id -> code of path that from the word to the root node, we call it path_code.
       Code means the label of each binary classifier, 1 indicate true, 0 indicate false.
    4. Now, each word should has its path and code along the path, you can pass a batch of path and code related
       to the same batch of inputs.

    Parameters:
        input (Variable): A tensor with the shape [N, D], where N is the size of mini-batch,
            and D is the feature size. Its data type supports float32 and float64.
        label (Variable): A tensor contains the labels of training data. Its shape is [N, 1]
            and data type is int64.
        weight (Variable): A tensor with shape (num_classes - 1, D) if not using custom tree(path_code and path_table is None), or (num_classes, D) if using custom tree.
        bias (Variable): A tensor with shape (num_classes - 1, 1) if not using custom tree(path_code and path_table is None), or (num_classes, 1) if using custom tree.
        num_classes (int): The number of classes or the size of word dict, must be greater than 2.
            If the default tree is used (:attr:`is_custom` is set to False), :attr:`num_classes`
            should not be None. If the custom tree is used (:attr:`is_custom` is set to True),
            :attr:`num_classes` should be the number of non-leaf nodes, which indicates the num of
            classes using by the binary classifier.
        path_table (Variable, optional): A tensor that stores each batch of samples' path from leaf to root
            node, its shape is [N, L] and data type is int64, where L is the length of path. For each sample i,
            path_table[i] is a np.array like structure and each element in this array is the indexes in parent
            nodes' weight matrix. Default: None.
        path_code (Variable, optional): A tensor that stores each batch of samples' code of path from leaf
            to root node, its shape is [N, L] and data type is int64, which is the same as :attr:`path_table`.
            Each code of path is consisted with the code of nodes from leaf to root node. Default: None.
        is_sparse (bool, optional): Whether use sparse updating instead of dense updating, if it's True, the
            gradient of W and input will be sparse. Default: False.

    Returns:
        Variable: A tensor with the cost of hierarchical sigmoid, its shape is [N, 1] and data type is the same as :attr:`input`.

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
                    w = fluid.data("weight", (num_classes -1, feature_size), dtype="float32")
                    b = fluid.data("bias", (num_classes -1, ), dtype="float32")
                    y = F.hsigmoid(x, label, w, b, num_classes)

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(start)
            feed_dict = {
                "input": np.random.randn(4, feature_size).astype(np.float32),
                "labels": np.random.randint(0, num_classes, (4, 1)).astype(np.int64),
                "weight": np.random.randn(num_classes - 1, feature_size).astype(np.float32),
                "bias": np.random.randn(num_classes - 1, ).astype(np.float32),
            }
            y_np, = exe.run(main, feed=feed_dict, fetch_list=[y])
            print(y_np.shape)

          # (4, 1)
    """

    attrs = {
        "num_classes": num_classes,
        "is_sparse": is_sparse,
        "remote_prefetch": is_sparse
    }

    inputs = {
        "X": input,
        "W": weight,
        "Bias": bias,
        "PathTable": path_table,
        "PathCode": path_code,
        "Label": label
    }

    helper = LayerHelper('hierarchical_sigmoid', **locals())
    dtype = helper.input_dtype()

    out = helper.create_variable_for_type_inference(dtype)
    pre_out = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": out, "PreOut": pre_out, "W_Out": weight}

    helper.append_op(
        type="hierarchical_sigmoid",
        inputs=inputs,
        outputs=outputs,
        attrs=attrs)
    return out


def relu(input, inplace=False, name=None):
    """
	:alias_main: paddle.nn.functional.relu
	:alias: paddle.nn.functional.relu,paddle.nn.functional.activation.relu

    ReLU Activation.

    .. math:

        out = max(x, 0)

    Parameters:
        input (Variable): The input variable. A multi-dimension Tensor with type float16, float32, or float64.
        inplace (bool, optional): If inplace is True, the input and output of ``ReLU`` are the same variable.
            Otherwise, the input and output of ``ReLU`` are different variables. Default: False. Note that if x is
            more than one OPs' input, inplace must be False.
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Output of relu operator, a Tensor with shape same as input

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import paddle.nn.functional as functional
          import numpy as np

          data = np.array([-2, 0, 1]).astype('float32')
          with fluid.dygraph.guard():
              data = fluid.dygraph.to_variable(data)
              res = functional.relu(data)  # [0, 0, 1]
    """

    if in_dygraph_mode():
        if inplace:
            warnings.warn(
                "Inplace on ReLU is not allowed and will be discarded in dygraph mode currently."
            )
        return core.ops.relu(input)

    check_variable_and_dtype(input, 'input', ['float16', 'float32', 'float64'],
                             'relu')

    helper = LayerHelper('relu', **locals())
    outs = input if inplace else helper.create_variable_for_type_inference(
        input.dtype)
    helper.append_op(type='relu', inputs={'X': [input]}, outputs={'Out': outs})
    return outs


def sigmoid(input, inplace=False, name=None):
    """
	:alias_main: paddle.nn.functional.sigmoid
	:alias: paddle.nn.functional.sigmoid,paddle.nn.functional.activation.sigmoid

    Sigmoid Activation.

    .. math:

        output = \frac{1}{1 + e^{-input}}
    
    Parameters:
        input (Variable): The input variable. A multi-dimension Tensor with type float16, float32, or float64.
        inplace (bool, optional): If inplace is True, the input and output are the same variable.
            Otherwise, the input and output of are different variables. Default: False. Note that if x is
            more than one OPs' input, inplace must be False.
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .
    
    Returns:
        Output of sigmoid operator, a Tensor with shape same as input
    
    Examples:
        .. code-block:: python
          
          import paddle.fluid as fluid
          import paddle.nn.functional as functional
          import numpy as np
          # In the static graph mode
          input = fluid.data(name="input", shape=[None, 4])
          output = functional.sigmoid(input)
          place = fluid.CPUPlace()
          exe = fluid.Executor(place)
          exe.run(fluid.default_startup_program())
          input_data = np.array([1.0, 2.0, 3.0, 4.0]).astype('float32')
          output_data = exe.run(feed={"input": input_data},
                                fetch_list=[output])
          print(output_data) # [0.7310586, 0.880797, 0.95257413, 0.98201376]
          # In the dynamic graph mode
          with fluid.dygraph.guard():
              input = fluid.dygraph.to_variable(input_data)
              output = functional.sigmoid(input)
              print(output) # [0.7310586, 0.880797, 0.95257413, 0.98201376]
    """

    if in_dygraph_mode():
        if inplace:
            warnings.warn(
                "Inplace on sigmoid is not allowed and will be discarded in dygraph mode currently."
            )
        return core.ops.sigmoid(input)

    check_variable_and_dtype(input, 'input', ['float16', 'float32', 'float64'],
                             'sigmoid')
    helper = LayerHelper("sigmoid", **locals())
    outputs = helper.create_variable_for_type_inference(input.dtype)
    helper.append_op(
        type='sigmoid', inputs={'X': [input]}, outputs={'Out': outputs})
    return outputs


def log_softmax(input, axis=None, dtype=None, name=None):
    """
	:alias_main: paddle.nn.functional.log_softmax
	:alias: paddle.nn.functional.log_softmax,paddle.nn.functional.activation.log_softmax

    This operator implements the log_softmax layer. The calculation process is as follows:

    .. math::

        Out[i, j] = log(softmax(x)) 
                  = log(\\frac{\exp(X[i, j])}{\sum_j(exp(X[i, j])})

    Parameters:
        input (Variable): The input variable. A multi-dimension Tensor with type float32, or float64.
        axis (int, optional): The index of dimension to perform softmax calculations, it should be in
            range :math:`[-1, rank-1]`, while :math:`rank` is the rank of input variable. Default: None. 
            None and -1 means the last dimension.
        dtype (np.dtype|core.VarDesc.VarType|str): The desired data type of returned tensor. If specified,
            the input tensor is casted to dtype before the operation is performed. This is useful for
            preventing data type overflows. Default: None. Supported dtype: float32 or float64
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .
 
    Returns:
        Variable: ``Tensor`` indicates the output of softmax. The data type and shape are the same as ``input``.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import paddle.nn.functional as F
          import numpy as np

          data = np.array([[[-2.0, 3.0, -4.0, 5.0],
                            [3.0, -4.0, 5.0, -6.0],
                            [-7.0, -8.0, 8.0, 9.0]],
                           [[1.0, -2.0, -3.0, 4.0],
                            [-5.0, 6.0, 7.0, -8.0],
                            [6.0, 7.0, 8.0, 9.0]]]).astype('float32')
          with fluid.dygraph.guard():
              data = fluid.dygraph.to_variable(data)
              res = F.log_softmax(data, -1)
              # [[[ -7.1278396   -2.1278396   -9.127839    -0.12783948]
              #   [ -2.1270514   -9.127051    -0.12705144 -11.127051  ]
              #   [-16.313261   -17.313261    -1.3132617   -0.31326184]]
              #  [[ -3.0518122   -6.051812    -7.051812    -0.051812  ]
              #   [-12.313267    -1.3132664   -0.3132665  -15.313267  ]
              #   [ -3.4401896   -2.4401896   -1.4401896   -0.44018966]]]
    """

    axis = -1 if axis is None else axis
    dtype = convert_np_dtype_to_dtype_(dtype) if dtype is not None else dtype

    if in_dygraph_mode():
        outs_cast = input if dtype is None \
            else core.ops.cast(input, 'in_dtype', input.dtype, 'out_dtype', dtype)
        outs_softmax = core.ops.softmax(outs_cast, 'axis', axis, 'use_cudnn',
                                        False)
        return core.ops.log(outs_softmax)

    if dtype is None:
        check_variable_and_dtype(
            input, 'input', ['float16', 'float32', 'float64'], 'log_softmax')

    helper = LayerHelper("log_softmax", **locals())
    outs_cast = input
    if dtype is not None:
        outs_cast = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type='cast',
            inputs={'X': input},
            outputs={'Out': outs_cast},
            attrs={'in_dtype': input.dtype,
                   'out_dtype': dtype})

    outs_softmax = helper.create_variable_for_type_inference(outs_cast.dtype)
    helper.append_op(
        type='softmax',
        inputs={'X': outs_cast},
        outputs={'Out': outs_softmax},
        attrs={'axis': axis,
               'use_cudnn': False})

    outs_log = helper.create_variable_for_type_inference(outs_softmax.dtype)
    helper.append_op(
        type='log', inputs={'X': outs_softmax}, outputs={'Out': outs_log})

    return outs_log
