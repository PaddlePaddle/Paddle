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
    'ELU',
    'GELU',
    'Hardshrink',
    #       'PReLU',
    'ReLU',
    'LeakyReLU',
    'Sigmoid',
    #       'Softmax',
    'LogSigmoid',
    'LogSoftmax',
    'HSigmoid'
]

from ...fluid.dygraph import layers
from ...fluid import core
from ...fluid.framework import in_dygraph_mode
from .. import functional as F


class ELU(layers.Layer):
    """
    ELU Activation.

    ..  math::
    
        ELU(x) = max(0, x) + min(0, \\alpha * (e^{x}-1))

    Parameters:
        alpha (float, optional): The 'alpha' value of the ELU formulation. Default is 1.0.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.
    
    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.
    
    Examples:
        .. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()

        x = paddle.to_tensor(np.array([[-1,6],[1,15.6]]))
        m = paddle.nn.ELU(0.2)
        out = m(x) 
        # [[-0.12642411  6.        ]
        #  [ 1.          15.6      ]]
    """

    def __init__(self, alpha=1.0, name=None):
        super(ELU, self).__init__()
        self._alpha = alpha
        self._name = name

    def forward(self, x):
        return F.elu(x, self._alpha, self._name)


class GELU(layers.Layer):
    """
    GELU Activation.

    If approximate is True

    ..  math::

        GELU(x) = 0.5 * x * (1 + tanh(\\sqrt{\\frac{2}{\\pi}} * (x + 0.044715x^{3})))

    else

    ..  math::

        GELU(x) = 0.5 * x * (1 + erf(\\frac{x}{\\sqrt{2}}))

    Parameters:
        approximate (bool, optional): Wether to enable approximation. Default is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.
    
    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.
    
    Examples:
        .. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()

        data = np.random.randn(2, 3).astype("float32")
        x = paddle.to_tensor(data)

        m = paddle.nn.GELU()
        out = m(x)

        data
        # array([[ 0.87165993, -1.0541513 , -0.37214822],
        #         [ 0.15647964,  0.32496083,  0.33045998]], dtype=float32)
        out
        # array([[ 0.70456535, -0.15380788, -0.13207214],
        #        [ 0.08796856,  0.20387867,  0.2080159 ]], dtype=float32)
    """

    def __init__(self, approximate=False, name=None):
        super(GELU, self).__init__()
        self._approximate = approximate
        self._name = name

    def forward(self, x):
        return F.gelu(x, self._approximate, self._name)


class Hardshrink(layers.Layer):
    """
    Hardshrink Activation

    .. math::

        hardshrink(x)=
            \left\{
            \begin{aligned}
            &x, & & if \ x > threshold \\
            &x, & & if \ x < -threshold \\
            &0, & & if \ others
            \end{aligned}
            \right.

    Parameters:
        threshold (float, optional): The value of threshold for hardthrink. Default is 0.5
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:

        .. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()

        x = paddle.to_variable(np.array([-1, 0.3, 2.5]))
        m = paddle.nn.Hardshrink()
        out = m(x) # [-1., 0., 2.5]
    """

    def __init__(self, threshold=0.5, name=None):
        super(Hardshrink, self).__init__()
        self._threshold = threshold
        self._name = name

    def forward(self, x):
        return F.hardshrink(x, self._threshold, self._name)


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
        out = F.hsigmoid(
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
    ReLU Activation.

    .. math:

        ReLU(x) = max(x, 0)

    Parameters:
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.
    
    Examples:
        .. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()

        x = paddle.to_tensor(np.array([-2, 0, 1]).astype('float32'))
        m = paddle.nn.ReLU()
        out = m(x) # [0., 0., 1.]
    """

    def __init__(self, name=None):
        super(ReLU, self).__init__()
        self._name = name

    def forward(self, x):
        return F.relu(x, self._name)


class LeakyReLU(layers.Layer):
    """
    Leaky ReLU Activation.

    .. math:

        out = max(x, alpha * x)

    Parameters:
        alpha (float, optional): Slope of the activation function at :math:`x < 0` .
            Default: 0.01.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.
    
    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.
    
    Examples:
        .. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()

        lrelu = paddle.nn.LeakyReLU()
        x = paddle.to_tensor(np.array([-2, 0, 1], 'float32'))
        out = lrelu(x)  # [-0.02, 0., 1.]
    """

    def __init__(self, alpha=1e-2, name=None):
        super(LeakyReLU, self).__init__()
        self._alpha = alpha
        self._name = name

    def forward(self, x):
        return F.leaky_relu(x, self._alpha, self._name)


class Sigmoid(layers.Layer):
    """
    this interface is used to construct a callable object of the ``Sigmoid`` class. This layer calcluate the `sigmoid` of input x.
    
    .. math::

        output = \\frac{1}{1 + e^{-x}}
    
    Parameters:
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        x: N-D tensor, available dtype is float16, float32, float64.

    Returns:
        A callable object of Sigmoid.
    
    Examples:

        .. code-block:: python

          import numpy as np
          import paddle

          paddle.disable_static()
          input_data = np.array([1.0, 2.0, 3.0, 4.0]).astype('float32')
          m = paddle.nn.Sigmoid()
          x = paddle.to_variable(input_data)
          output = m(x)
          print(output.numpy()) # [0.7310586, 0.880797, 0.95257413, 0.98201376]
    """

    def __init__(self, name=None):
        super(Sigmoid, self).__init__()
        self.name = name

    def forward(self, x):
        return F.sigmoid(x, self.name)


class LogSigmoid(layers.Layer):
    """
    LogSigmoid Activation.
    
    .. math:

        LogSigmoid(x) = \log \frac{1}{1 + e^{-x}}

    Parameters:
        x (Tensor): The input Tensor with data type float32, or float64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.
    
    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.
    
    Examples:
        .. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()

        x = paddle.to_tensor(np.array([1.0, 2.0, 3.0, 4.0]))
        m = paddle.nn.LogSigmoid()
        out = m(x) # [0.7310586, 0.880797, 0.95257413, 0.98201376]
    """

    def __init__(self, name=None):
        super(LogSigmoid, self).__init__()
        self._name = name

    def forward(self, x):
        return F.logsigmoid(x, self._name)


class LogSoftmax(layers.Layer):
    """
    This operator implements the log_softmax layer. The calculation process is as follows:

    .. math::

        Out[i, j] = log(softmax(x)) 
                  = log(\\frac{\exp(X[i, j])}{\sum_j(exp(X[i, j])})

    Parameters:
        axis (int, optional): The axis along which to perform log_softmax
            calculations. It should be in range [-D, D), where D is the
            dimensions of the input Tensor . If ``axis`` < 0, it works the
            same way as :math:`axis + D` . Default is -1.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.
 
    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()

        x = np.array([[[-2.0, 3.0, -4.0, 5.0],
                        [3.0, -4.0, 5.0, -6.0],
                        [-7.0, -8.0, 8.0, 9.0]],
                        [[1.0, -2.0, -3.0, 4.0],
                        [-5.0, 6.0, 7.0, -8.0],
                        [6.0, 7.0, 8.0, 9.0]]])
        m = paddle.nn.LogSoftmax()
        x = paddle.to_tensor(x)
        out = m(x)
        # [[[ -7.1278396   -2.1278396   -9.127839    -0.12783948]
        #   [ -2.1270514   -9.127051    -0.12705144 -11.127051  ]
        #   [-16.313261   -17.313261    -1.3132617   -0.31326184]]
        #  [[ -3.0518122   -6.051812    -7.051812    -0.051812  ]
        #   [-12.313267    -1.3132664   -0.3132665  -15.313267  ]
        #   [ -3.4401896   -2.4401896   -1.4401896   -0.44018966]]]
    """

    def __init__(self, axis=-1, name=None):
        super(LogSoftmax, self).__init__()
        self._axis = axis
        self._name = name

    def forward(self, x):
        return F.log_softmax(x, self._axis)
