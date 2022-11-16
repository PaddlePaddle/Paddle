# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from .. import core
from ..layers import utils
from ..layers import nn as F
from .. import dygraph_utils
from . import layers
from ..framework import (
    Variable,
    _non_static_mode,
    OpProtoHolder,
    Parameter,
    _dygraph_tracer,
    _varbase_creator,
    default_main_program,
    _global_flags,
    in_dygraph_mode,
    _in_legacy_dygraph,
)
from ..data_feeder import (
    convert_dtype,
    check_variable_and_dtype,
    check_type,
    check_dtype,
)
from ..param_attr import ParamAttr
from ..initializer import Normal, Constant, NumpyArrayInitializer
from .. import unique_name
from .layer_object_helper import LayerObjectHelper
from ..data_feeder import check_variable_and_dtype, check_type
import numpy as np
import numbers
import logging
import os
import paddle.utils.deprecated as deprecated
from paddle import _C_ops, _legacy_C_ops

__all__ = [
    'BatchNorm',
    'Dropout',
    'Embedding',
    'GRUUnit',
    'InstanceNorm',
    'LayerNorm',
    'NCE',
    'PRelu',
    'BilinearTensorProduct',
    'Conv2DTranspose',
    'GroupNorm',
    'SpectralNorm',
    'TreeConv',
    'Flatten',
]


class InstanceNorm(layers.Layer):
    r"""
    This interface is used to construct a callable object of the ``InstanceNorm`` class.
    For more details, refer to code examples.

    Can be used as a normalizer function for convolution or fully_connected operations.
    The required data format for this layer is one of the following:

    DataLayout: NCHW `[batch, in_channels, in_height, in_width]`

    Refer to `Instance Normalization: The Missing Ingredient for Fast Stylization <https://arxiv.org/pdf/1607.08022.pdf>`_
    for more details.

    :math:`input` is the input features over a mini-batch.

    ..  math::

        \\mu_{\\beta} &\\gets \\frac{1}{HW} \\sum_{i=1}^{HW} x_i \\qquad &//\\
        \\ mean\ of\ one\  feature\ map\ in\ mini-batch \\\\
        \\sigma_{\\beta}^{2} &\\gets \\frac{1}{HW} \\sum_{i=1}^{HW}(x_i - \\
        \\mu_{\\beta})^2 \\qquad &//\ variance\ of\ one\ feature\ map\ in\ mini-batch \\\\
        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}} \\qquad &//\ normalize \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta \\qquad &//\ scale\ and\ shift

    Note:
        `H` means height of feature map, `W` means width of feature map.

    Parameters:
        num_channels(int): Indicate the number of channels of the input ``Tensor``.
        epsilon(float, optional): A value added to the denominator for
            numerical stability. Default is 1e-5.
        param_attr(ParamAttr|bool, optional): The parameter attribute for Parameter `scale`
             of instance_norm. If it is set to None or one attribute of ParamAttr, instance_norm
	     will create ParamAttr as param_attr, the name of scale can be set in ParamAttr.
	     If the Initializer of the param_attr is not set, the parameter is initialized
	     one. If it is set to False, will not create param_attr. Default: None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of instance_norm.
             If it is set to None or one attribute of ParamAttr, instance_norm
	     will create ParamAttr as bias_attr, the name of bias can be set in ParamAttr.
	     If the Initializer of the bias_attr is not set, the bias is initialized zero.
             If it is set to False, will not create bias_attr. Default: None.
        dtype(str, optional): Indicate the data type of the input ``Tensor``,
             which can be float32 or float64. Default: float32.

    Returns:
        None.

    Examples:

        .. code-block:: python

          import paddle.fluid as fluid
          from paddle.fluid.dygraph.base import to_variable
          import numpy as np
          import paddle

          # x's shape is [1, 3, 1, 2]
          x = np.array([[[[1.0, 8.0]], [[10.0, 5.0]], [[4.0, 6.0]]]]).astype('float32')
          with fluid.dygraph.guard():
              x = to_variable(x)
              instanceNorm = paddle.nn.InstanceNorm(3)
              ret = instanceNorm(x)
              # ret's shape is [1, 3, 1, 2]; value is [-1 1 0.999999 -0.999999 -0.999995 0.999995]
              print(ret)

    """

    def __init__(
        self,
        num_channels,
        epsilon=1e-5,
        param_attr=None,
        bias_attr=None,
        dtype='float32',
    ):
        super().__init__()

        if param_attr == False or bias_attr == False:
            assert (
                bias_attr == param_attr
            ), "param_attr and bias_attr must be set to False at the same time in InstanceNorm"
        self._epsilon = epsilon
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._dtype = dtype

        if param_attr != False and bias_attr != False:
            self.scale = self.create_parameter(
                attr=self._param_attr,
                shape=[num_channels],
                dtype=self._dtype,
                default_initializer=Constant(1.0),
                is_bias=False,
            )
            self.bias = self.create_parameter(
                attr=self._bias_attr,
                shape=[num_channels],
                dtype=self._dtype,
                default_initializer=Constant(0.0),
                is_bias=True,
            )
        else:
            self.scale = None
            self.bias = None

    def forward(self, input):
        if in_dygraph_mode():
            out = _C_ops.instance_norm(
                input, self.scale, self.bias, self._epsilon
            )
            return out
        if _in_legacy_dygraph():
            out, _, _ = _legacy_C_ops.instance_norm(
                input, self.scale, self.bias, 'epsilon', self._epsilon
            )
            return out

        check_variable_and_dtype(
            input, 'input', ['float32', 'float64'], "InstanceNorm"
        )

        attrs = {"epsilon": self._epsilon}

        if self.scale and self.bias:
            inputs = {"X": [input], "Scale": [self.scale], "Bias": [self.bias]}
        else:
            inputs = {"X": [input]}

        saved_mean = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True
        )
        saved_variance = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True
        )
        instance_norm_out = self._helper.create_variable_for_type_inference(
            self._dtype
        )

        outputs = {
            "Y": [instance_norm_out],
            "SavedMean": [saved_mean],
            "SavedVariance": [saved_variance],
        }

        self._helper.append_op(
            type="instance_norm", inputs=inputs, outputs=outputs, attrs=attrs
        )
        return instance_norm_out


class BatchNorm(layers.Layer):
    r"""

    This interface is used to construct a callable object of the ``BatchNorm`` class.
    For more details, refer to code examples.
    It implements the function of the Batch Normalization Layer and can be used
    as a normalizer function for conv2d and fully connected operations.
    The data is normalized by the mean and variance of the channel based on the current batch data.
    Refer to `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/pdf/1502.03167.pdf>`_
    for more details.

    When use_global_stats = False, the :math:`\mu_{\beta}`
    and :math:`\sigma_{\beta}^{2}` are the statistics of one mini-batch.
    Calculated as follows:

    ..  math::

        \mu_{\beta} &\gets \frac{1}{m} \sum_{i=1}^{m} x_i \qquad &
        //\ mini-batch\ mean \\
        \sigma_{\beta}^{2} &\gets \frac{1}{m} \sum_{i=1}^{m}(x_i - \mu_{\beta})^2 \qquad &
        //\ mini-batch\ variance \\

    - :math:`x` : mini-batch data
    - :math:`m` : the size of the mini-batch data

    When use_global_stats = True, the :math:`\\mu_{\\beta}`
    and :math:`\\sigma_{\\beta}^{2}` are not the statistics of one mini-batch.
    They are global or running statistics (moving_mean and moving_variance). It usually got from the
    pre-trained model. Calculated as follows:

    .. math::
        moving\_mean = moving\_mean * momentum + \mu_{\beta} * (1. - momentum) \quad &// global mean \\
        moving\_variance = moving\_variance * momentum + \sigma_{\beta}^{2} * (1. - momentum) \quad &// global variance \\

    The normalization function formula is as follows:

    ..  math::

        \hat{x_i} &\gets \frac{x_i - \mu_\beta} {\sqrt{\
        \sigma_{\beta}^{2} + \epsilon}} \qquad &//\ normalize \\
        y_i &\gets \gamma \hat{x_i} + \beta \qquad &//\ scale\ and\ shift


    - :math:`\epsilon` : add a smaller value to the variance to prevent division by zero
    - :math:`\gamma` : trainable proportional parameter
    - :math:`\beta` : trainable deviation parameter

    Parameters:
        num_channels(int): Indicate the number of channels of the input ``Tensor``.
        act(str, optional): Activation to be applied to the output of batch normalization. Default: None.
        is_test (bool, optional): A flag indicating whether it is in test phrase or not.
             This flag only has effect on static graph mode. For dygraph mode, please use ``eval()``.
             Default: False.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        epsilon(float, optional): The small value added to the variance to prevent division by zero. Default: 1e-5.
        param_attr(ParamAttr, optional): The parameter attribute for Parameter `scale`
             of batch_norm. If it is set to None or one attribute of ParamAttr, batch_norm
             will create ParamAttr as param_attr. If the Initializer of the param_attr
             is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr(ParamAttr, optional): The parameter attribute for the bias of batch_norm.
             If it is set to None or one attribute of ParamAttr, batch_norm
             will create ParamAttr as bias_attr. If the Initializer of the bias_attr
             is not set, the bias is initialized zero. Default: None.
        dtype(str, optional): Indicate the data type of the input ``Tensor``,
             which can be float32 or float64. Default: float32.
        data_layout(str, optional): Specify the input data format, the data format can be "NCHW" or "NHWC". Default: NCHW.
        in_place(bool, optional): Make the input and output of batch norm reuse memory. Default: False.
        moving_mean_name(str, optional): The name of moving_mean which store the global Mean. Default: None.
        moving_variance_name(str, optional): The name of the moving_variance which store the global Variance. Default: None.
        do_model_average_for_mean_and_var(bool, optional): Whether parameter mean and variance should do model
            average when model average is enabled. Default: True.
        use_global_stats(bool, optional): Whether to use global mean and
            variance. In inference or test mode, set use_global_stats to true
            or is_test to true, and the behavior is equivalent.
            In train mode, when setting use_global_stats True, the global mean
            and variance are also used during train period. Default: False.
        trainable_statistics(bool, optional): Whether to calculate mean and var in eval mode. In eval mode, when
            setting trainable_statistics True, mean and variance will be calculated by current batch statistics.
            Default: False.

    Returns:
        None

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          from paddle.fluid.dygraph.base import to_variable
          import numpy as np

          x = np.random.random(size=(3, 10, 3, 7)).astype('float32')
          with fluid.dygraph.guard():
              x = to_variable(x)
              batch_norm = fluid.BatchNorm(10)
              hidden1 = batch_norm(x)
    """

    def __init__(
        self,
        num_channels,
        act=None,
        is_test=False,
        momentum=0.9,
        epsilon=1e-05,
        param_attr=None,
        bias_attr=None,
        dtype='float32',
        data_layout='NCHW',
        in_place=False,
        moving_mean_name=None,
        moving_variance_name=None,
        do_model_average_for_mean_and_var=True,
        use_global_stats=False,
        trainable_statistics=False,
    ):
        super().__init__()
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._act = act
        self._use_mkldnn = _global_flags()["FLAGS_use_mkldnn"]

        assert (
            bias_attr is not False
        ), "bias_attr should not be False in batch_norm."

        if dtype == "float16":
            self._dtype = "float32"
        else:
            self._dtype = dtype

        param_shape = [num_channels]

        # create parameter
        self.weight = self.create_parameter(
            attr=self._param_attr,
            shape=param_shape,
            dtype=self._dtype,
            default_initializer=Constant(1.0),
        )
        self.weight.stop_gradient = (
            use_global_stats and self._param_attr.learning_rate == 0.0
        )

        self.bias = self.create_parameter(
            attr=self._bias_attr,
            shape=param_shape,
            dtype=self._dtype,
            is_bias=True,
        )
        self.bias.stop_gradient = (
            use_global_stats and self._param_attr.learning_rate == 0.0
        )

        self._mean = self.create_parameter(
            attr=ParamAttr(
                name=moving_mean_name,
                initializer=Constant(0.0),
                trainable=False,
                do_model_average=do_model_average_for_mean_and_var,
            ),
            shape=param_shape,
            dtype=self._dtype,
        )
        self._mean.stop_gradient = True

        self._variance = self.create_parameter(
            attr=ParamAttr(
                name=moving_variance_name,
                initializer=Constant(1.0),
                trainable=False,
                do_model_average=do_model_average_for_mean_and_var,
            ),
            shape=param_shape,
            dtype=self._dtype,
        )
        self._variance.stop_gradient = True

        self._in_place = in_place
        self._data_layout = data_layout
        self._momentum = momentum
        self._epsilon = epsilon
        self._is_test = is_test
        self._fuse_with_relu = False
        self._use_global_stats = use_global_stats
        self._trainable_statistics = trainable_statistics

    def forward(self, input):
        # create output
        # mean and mean_out share the same memory
        mean_out = self._mean
        # variance and variance out share the same memory
        variance_out = self._variance

        if _non_static_mode():
            if in_dygraph_mode():
                batch_norm_out, t1, t2, t3, t4, _ = _C_ops.batch_norm(
                    input,
                    self._mean,
                    self._variance,
                    self.weight,
                    self.bias,
                    not self.training,
                    self._momentum,
                    self._epsilon,
                    self._data_layout,
                    self._use_global_stats,
                    self._trainable_statistics,
                )
                return dygraph_utils._append_activation_in_dygraph(
                    batch_norm_out, act=self._act, use_mkldnn=self._use_mkldnn
                )

            elif _in_legacy_dygraph():
                attrs = (
                    "momentum",
                    self._momentum,
                    "epsilon",
                    self._epsilon,
                    "is_test",
                    not self.training,
                    "data_layout",
                    self._data_layout,
                    "use_mkldnn",
                    self._use_mkldnn,
                    "fuse_with_relu",
                    self._fuse_with_relu,
                    "use_global_stats",
                    self._use_global_stats,
                    'trainable_statistics',
                    self._trainable_statistics,
                )
                batch_norm_out, _, _, _, _, _ = _legacy_C_ops.batch_norm(
                    input,
                    self.weight,
                    self.bias,
                    self._mean,
                    self._variance,
                    None,
                    mean_out,
                    variance_out,
                    *attrs
                )

            return dygraph_utils._append_activation_in_dygraph(
                batch_norm_out, act=self._act, use_mkldnn=self._use_mkldnn
            )

        check_variable_and_dtype(
            input, 'input', ['float16', 'float32', 'float64'], 'BatchNorm'
        )

        attrs = {
            "momentum": self._momentum,
            "epsilon": self._epsilon,
            "is_test": self._is_test,
            "data_layout": self._data_layout,
            "use_mkldnn": False,
            "fuse_with_relu": self._fuse_with_relu,
            "use_global_stats": self._use_global_stats,
            "trainable_statistics": self._trainable_statistics,
        }

        inputs = {
            "X": [input],
            "Scale": [self.weight],
            "Bias": [self.bias],
            "Mean": [self._mean],
            "Variance": [self._variance],
        }

        saved_mean = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True
        )
        saved_variance = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True
        )
        reserve_space = self._helper.create_variable_for_type_inference(
            dtype=self._helper.input_dtype(input), stop_gradient=True
        )

        batch_norm_out = (
            input
            if self._in_place
            else self._helper.create_variable_for_type_inference(self._dtype)
        )

        outputs = {
            "Y": [batch_norm_out],
            "MeanOut": [mean_out],
            "VarianceOut": [variance_out],
            "SavedMean": [saved_mean],
            "SavedVariance": [saved_variance],
        }
        if reserve_space is not None:
            outputs["ReserveSpace"] = [reserve_space]

        self._helper.append_op(
            type="batch_norm", inputs=inputs, outputs=outputs, attrs=attrs
        )

        # Currently, we don't support inplace in dygraph mode
        return self._helper.append_activation(batch_norm_out, self._act)


class Dropout(layers.Layer):
    """
    This interface is used to construct a callable object of the ``Dropout`` class.
    For more details, refer to code examples.

    Drop or keep each element of input independently. Dropout is a regularization
    technique for reducing overfitting by preventing neuron co-adaption during
    training. The dropout operator randomly sets (according to the given dropout
    probability) the outputs of some units to zero, while others are remain
    unchanged.

    Dropout layer can be removed for efficiency concern.

    Parameters:
        p (float, optional): Probability of setting units to zero. Default: 0.5
        seed (int, optional): A Python integer used to create random seeds. If this
                    parameter is set to None, a random seed is used.
                    NOTE: If an integer seed is given, always the same output
                    units will be dropped. DO NOT use a fixed seed in training. Default: None.
        dropout_implementation(string, optional): ['downgrade_in_infer'(default)|'upscale_in_train']

                                        1. downgrade_in_infer(default), downgrade the outcome at inference

                                           - train: out = input * mask
                                           - inference: out = input * (1.0 - p)

                                           (mask is a tensor same shape with input, value is 0 or 1
                                           ratio of 0 is dropout_prob)
                                        2. upscale_in_train, upscale the outcome at training time

                                           - train: out = input * mask / ( 1.0 - p )
                                           - inference: out = input

                                           (mask is a tensor same shape with input, value is 0 or 1
                                           ratio of 0 is p)
        is_test (bool, optional): A flag indicating whether it is in test phrase or not.
                    This flag only has effect on static graph mode. For dygraph mode, please use ``eval()``.
                    Default: False.

    Returns:
        None

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            from paddle.fluid.dygraph.base import to_variable
            import numpy as np

            x = np.random.random(size=(3, 10, 3, 7)).astype('float32')
            with fluid.dygraph.guard():
                x = to_variable(x)
                m = fluid.dygraph.Dropout(p=0.5)
                droped_train = m(x)
                # switch to eval mode
                m.eval()
                droped_eval = m(x)
    """

    def __init__(
        self,
        p=0.5,
        seed=None,
        dropout_implementation="downgrade_in_infer",
        is_test=False,
    ):
        super().__init__()
        assert isinstance(p, (float, int)), "p argument should be a number"
        assert 0 <= p <= 1, "p argument should between 0 and 1"
        self._dropout_prob = p
        assert seed is None or isinstance(
            seed, int
        ), "seed argument should be None or a integer"
        self._seed = seed
        assert dropout_implementation in (
            'downgrade_in_infer',
            'upscale_in_train',
        ), "dropout_implementation argument should be 'downgrade_in_infer' or 'upscale_in_train'"
        self._dropout_implementation = dropout_implementation
        self._is_test = is_test

    def forward(self, input):
        # fast return for p == 0
        if self._dropout_prob == 0:
            return input
        prog = default_main_program()
        if (self._seed is None or self._seed == 0) and prog.random_seed != 0:
            self._seed = prog.random_seed
        attrs = {
            'dropout_prob': self._dropout_prob,
            'is_test': not self.training
            if _non_static_mode()
            else self._is_test,
            'fix_seed': self._seed is not None,
            'seed': self._seed if self._seed is not None else 0,
            'dropout_implementation': self._dropout_implementation,
        }

        if _non_static_mode():
            attrs = sum(attrs.items(), ())
            out, mask = _legacy_C_ops.dropout(input, *attrs)
            return out

        out = self._helper.create_variable_for_type_inference(dtype=input.dtype)
        mask = self._helper.create_variable_for_type_inference(
            dtype=core.VarDesc.VarType.UINT8, stop_gradient=True
        )

        self._helper.append_op(
            type='dropout',
            inputs={'X': [input]},
            outputs={'Out': [out], 'Mask': [mask]},
            attrs=attrs,
        )
        return out


class Embedding(layers.Layer):
    r"""
    :alias_main: paddle.nn.Embedding
        :alias: paddle.nn.Embedding,paddle.nn.layer.Embedding,paddle.nn.layer.common.Embedding
        :old_api: paddle.fluid.dygraph.Embedding

    **Embedding Layer**

    This interface is used to construct a callable object of the ``Embedding`` class.
    For specific usage, refer to code examples. It implements the function of the Embedding Layer.
    This layer is used to lookup embeddings vector of ids provided by :attr:`input` .
    It automatically constructs a 2D embedding matrix based on the
    input :attr:`size` (vocab_size, emb_size) and :attr:`dtype` .

    The shape of output Tensor is generated by appending an emb_size dimension to the
    last dimension of the input Tensor shape.

    **Note:** The id in :attr:`input` must satisfy :math:`0 =< id < size[0]` ,
    otherwise the program will throw an exception and exit.

    .. code-block:: text

        Case 1:

        input is a Tensor. padding_idx = -1
            input.data = [[1, 3], [2, 4], [4, 127]
            input.shape = [3, 2]
        Given size = [128, 16]
        output is a Tensor:
            out.shape = [3, 2, 16]
            out.data = [[[0.129435295, 0.244512452, ..., 0.436322452],
                        [0.345421456, 0.524563927, ..., 0.144534654]],

                        [[0.345249859, 0.124939536, ..., 0.194353745],
                        [0.945345345, 0.435394634, ..., 0.435345365]],

                        [[0.945345345, 0.435394634, ..., 0.435345365],
                        [0.0,         0.0,         ..., 0.0        ]]]  # padding data
        The input padding_idx is less than 0, it is automatically converted to padding_idx = -1 + 128 = 127
        It will pad all-zero data when ids is 127.

    Parameters:
        size(tuple|list): The shape of the look up table parameter. It should have two elements which indicate the size
            of the dictionary of embeddings and the size of each embedding vector respectively.
        is_sparse(bool): The flag indicating whether to use sparse update. This parameter only
            affects the performance of the backwards gradient update. It is recommended to set
            True because sparse update is faster. But some optimizer does not support sparse update,
            such as :ref:`api_fluid_optimizer_AdadeltaOptimizer` , :ref:`api_fluid_optimizer_AdamaxOptimizer` ,
            :ref:`api_fluid_optimizer_DecayedAdagradOptimizer` , :ref:`api_fluid_optimizer_FtrlOptimizer` ,
            :ref:`api_fluid_optimizer_LambOptimizer` and :ref:`api_fluid_optimizer_LarsMomentumOptimizer` .
            In these case, is_sparse must be False. Default: False.
        is_distributed(bool): Whether to store the embedding matrix in a distributed manner. Only used
            in multi-machine distributed CPU training. Default: False.
        padding_idx(int|long|None): padding_idx needs to be in the interval [-vocab_size, vocab_size).
            If :math:`padding\_idx < 0`, the :math:`padding\_idx` will automatically be converted
            to :math:`vocab\_size + padding\_idx` . It will output all-zero padding data whenever lookup
            encounters :math:`padding\_idx` in id. And the padding data will not be updated while training.
            If set None, it makes no effect to output. Default: None.
        param_attr(ParamAttr): To specify the weight parameter property. Default: None, which means the
            default weight parameter property is used. See usage for details in :ref:`api_fluid_ParamAttr` . In addition,
            user-defined or pre-trained word vectors can be loaded with the :attr:`param_attr` parameter.
            The local word vector needs to be transformed into numpy format, and the shape of local word
            vector should be consistent with :attr:`size` . Then :ref:`api_fluid_initializer_NumpyArrayInitializer`
            is used to load custom or pre-trained word vectors. See code example 2 for details.
        dtype(np.dtype|core.VarDesc.VarType|str): It refers to the data type of output Tensor.
            It must be "float32" or "float64". Default: "float32".

    Attribute:
        **weight** (Parameter): the learnable weights of this layer.

    Returns:
        Variable: Embedding Tensor or LoDTensor mapped by input. The data type is the same as :attr:`dtype` .

    Examples:

        .. code-block:: python

          import paddle.fluid as fluid
          import paddle.fluid.dygraph.base as base
          import numpy as np

          # example 1
          inp_word = np.array([[2, 3, 5], [4, 2, 1]]).astype('int64')
          inp_word.shape  # [2, 3]
          dict_size = 20
          with fluid.dygraph.guard():
              emb = fluid.dygraph.Embedding(
                  size=[dict_size, 32],
                  param_attr='emb.w',
                  is_sparse=False)
              static_rlt3 = emb(base.to_variable(inp_word))
              static_rlt3.shape  # [2, 3, 32]

          # example 2: load custom or pre-trained word vectors
          weight_data = np.random.random(size=(128, 100))  # word vectors with numpy format
          w_param_attrs = fluid.ParamAttr(
              name="emb_weight",
              learning_rate=0.5,
              initializer=fluid.initializer.NumpyArrayInitializer(weight_data),
              trainable=True)
          with fluid.dygraph.guard():
              emb = fluid.dygraph.Embedding(
                  size=[128, 100],
                  param_attr= w_param_attrs,
                  is_sparse=False)
              static_rlt3 = emb(base.to_variable(inp_word))
    """

    def __init__(
        self,
        size,
        is_sparse=False,
        is_distributed=False,
        padding_idx=None,
        param_attr=None,
        dtype='float32',
    ):
        super().__init__()
        self._size = size
        self._is_sparse = is_sparse
        self._is_distributed = is_distributed
        self._padding_idx = (
            -1
            if padding_idx is None
            else padding_idx
            if padding_idx >= 0
            else (size[0] + padding_idx)
        )

        self._param_attr = param_attr
        self._dtype = dtype
        self._remote_prefetch = self._is_sparse and (not self._is_distributed)
        if self._remote_prefetch:
            assert self._is_sparse is True and self._is_distributed is False

        self.weight = self.create_parameter(
            attr=self._param_attr,
            shape=self._size,
            dtype=self._dtype,
            is_bias=False,
        )

    def forward(self, input):
        if _non_static_mode():
            return _legacy_C_ops.lookup_table_v2(
                self.weight,
                input,
                'is_sparse',
                self._is_sparse,
                'is_distributed',
                self._is_distributed,
                'remote_prefetch',
                self._remote_prefetch,
                'padding_idx',
                self._padding_idx,
            )

        check_variable_and_dtype(
            input,
            'input',
            ['uint8', 'int8', 'int16', 'int32', 'int64'],
            'Embedding',
        )
        attrs = {
            'is_sparse': self._is_sparse,
            'is_distributed': self._is_distributed,
            'remote_prefetch': self._remote_prefetch,
            'padding_idx': self._padding_idx,
        }

        out = self._helper.create_variable_for_type_inference(self._dtype)
        self._helper.append_op(
            type='lookup_table_v2',
            inputs={'Ids': input, 'W': self.weight},
            outputs={'Out': out},
            attrs=attrs,
        )

        return out


class LayerNorm(layers.Layer):
    r"""
    :alias_main: paddle.nn.LayerNorm
        :alias: paddle.nn.LayerNorm,paddle.nn.layer.LayerNorm,paddle.nn.layer.norm.LayerNorm
        :old_api: paddle.fluid.dygraph.LayerNorm

    This interface is used to construct a callable object of the ``LayerNorm`` class.
    For more details, refer to code examples.
    It implements the function of the Layer Normalization Layer and can be applied to mini-batch input data.
    Refer to `Layer Normalization <https://arxiv.org/pdf/1607.06450v1.pdf>`_

    The formula is as follows:

    ..  math::

        \\mu & = \\frac{1}{H}\\sum_{i=1}^{H} x_i

        \\sigma & = \\sqrt{\\frac{1}{H}\sum_{i=1}^{H}{(x_i - \\mu)^2} + \\epsilon}

        y & = f(\\frac{g}{\\sigma}(x - \\mu) + b)

    - :math:`x`: the vector representation of the summed inputs to the neurons in that layer.
    - :math:`H`: the number of hidden units in a layers
    - :math:`\\epsilon`: the small value added to the variance to prevent division by zero.
    - :math:`g`: the trainable scale parameter.
    - :math:`b`: the trainable bias parameter.

    Parameters:
        normalized_shape(int or list or tuple): Input shape from an expected input of
            size :math:`[*, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1]]`.
            If it is a single integer, this module will normalize over the last dimension
            which is expected to be of that specific size.
        scale(bool, optional): Whether to learn the adaptive gain :math:`g` after
            normalization. Default: True.
        shift(bool, optional): Whether to learn the adaptive bias :math:`b` after
            normalization. Default: True.
        epsilon(float, optional): The small value added to the variance to prevent
            division by zero. Default: 1e-05.
        param_attr(ParamAttr, optional): The parameter attribute for the learnable
            gain :math:`g`. If :attr:`scale` is False, :attr:`param_attr` is
            omitted. If :attr:`scale` is True and :attr:`param_attr` is None,
            a default :code:`ParamAttr` would be added as scale. The
            :attr:`param_attr` is initialized as 1 if it is added. Default: None.
        bias_attr(ParamAttr, optional): The parameter attribute for the learnable
            bias :math:`b`. If :attr:`shift` is False, :attr:`bias_attr` is
            omitted. If :attr:`shift` is True and :attr:`param_attr` is None,
            a default :code:`ParamAttr` would be added as bias. The
            :attr:`bias_attr` is initialized as 0 if it is added. Default: None.
        act(str, optional): Activation to be applied to the output of layer normalization.
                  Default: None.
        dtype (str, optional): Data type, it can be "float32" or "float64". Default: "float32".

    Returns:
        None

    Examples:

        .. code-block:: python

          import paddle.fluid as fluid
          from paddle.fluid.dygraph.base import to_variable
          import numpy

          x = numpy.random.random((3, 32, 32)).astype('float32')
          with fluid.dygraph.guard():
              x = to_variable(x)
              layerNorm = fluid.LayerNorm([32, 32])
              ret = layerNorm(x)

    """

    def __init__(
        self,
        normalized_shape,
        scale=True,
        shift=True,
        epsilon=1e-05,
        param_attr=None,
        bias_attr=None,
        act=None,
        dtype='float32',
    ):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = [normalized_shape]

        self._normalized_shape = list(normalized_shape)
        self._scale = scale
        self._shift = shift
        self._epsilon = epsilon
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._act = act
        self._dtype = dtype
        param_shape = [np.prod(self._normalized_shape)]
        if self._scale:
            self.weight = self.create_parameter(
                attr=self._param_attr,
                shape=param_shape,
                dtype=self._dtype,
                default_initializer=Constant(1.0),
            )
        else:
            if self._param_attr:
                logging.warn("param_attr are only available with scale is True")
            self.weight = None

        if self._shift:
            assert self._bias_attr is not False
            self.bias = self.create_parameter(
                attr=self._bias_attr,
                shape=param_shape,
                dtype=self._dtype,
                is_bias=True,
            )
        else:
            if self._bias_attr:
                logging.warn("bias_attr are only available with shift is True")
            self.bias = None

    def forward(self, input):
        input_shape = list(input.shape)
        input_ndim = len(input_shape)
        normalized_ndim = len(self._normalized_shape)
        self._begin_norm_axis = input_ndim - normalized_ndim
        if (
            input_ndim < normalized_ndim
            or input_shape[self._begin_norm_axis :] != self._normalized_shape
        ):
            str_normalized_shape = str(self._normalized_shape)
            raise ValueError(
                'Given normalized_shape is '
                + str_normalized_shape
                + ', expected input with shape [*, '
                + str_normalized_shape[1:]
                + ', but got input shape '
                + str(input_shape)
            )

        if _non_static_mode():
            if in_dygraph_mode():
                pre_act, _, _, = _C_ops.layer_norm(
                    input,
                    self.weight,
                    self.bias,
                    self._epsilon,
                    self._begin_norm_axis,
                    False,
                )
                return dygraph_utils._append_activation_in_dygraph(
                    pre_act, act=self._act
                )
            else:
                pre_act, _, _ = _legacy_C_ops.layer_norm(
                    input,
                    self.weight,
                    self.bias,
                    'epsilon',
                    self._epsilon,
                    'begin_norm_axis',
                    self._begin_norm_axis,
                )
                return dygraph_utils._append_activation_in_dygraph(
                    pre_act, act=self._act
                )

        check_variable_and_dtype(
            input, 'input', ['float32', 'float64'], 'LayerNorm'
        )

        inputs = dict()
        inputs['X'] = [input]
        if self._scale:
            inputs['Scale'] = [self.weight]
        if self._shift:
            inputs['Bias'] = [self.bias]
        attrs = {
            "epsilon": self._epsilon,
            "begin_norm_axis": self._begin_norm_axis,
        }

        # create output
        mean_out = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True
        )
        variance_out = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True
        )
        layer_norm_out = self._helper.create_variable_for_type_inference(
            self._dtype
        )

        self._helper.append_op(
            type="layer_norm",
            inputs=inputs,
            outputs={
                "Y": layer_norm_out,
                "Mean": mean_out,
                "Variance": variance_out,
            },
            attrs={
                "epsilon": self._epsilon,
                "begin_norm_axis": self._begin_norm_axis,
            },
        )

        return self._helper.append_activation(layer_norm_out, act=self._act)


class GRUUnit(layers.Layer):
    """
    **GRU unit layer**

    It creates a callable object from GRUUnit class.
    If origin_mode is True, then the equation of a gru step is from paper
    `Learning Phrase Representations using RNN Encoder-Decoder for Statistical
    Machine Translation <https://arxiv.org/pdf/1406.1078.pdf>`_

        .. math::
            u_t & = actGate(xu_{t} + W_u h_{t-1} + b_u)

            r_t & = actGate(xr_{t} + W_r h_{t-1} + b_r)

            m_t & = actNode(xm_t + W_c dot(r_t, h_{t-1}) + b_m)

            h_t & = dot(u_t, h_{t-1}) + dot((1-u_t), m_t)

    If origin_mode is False, then the equation of a gru step is from paper
    `Empirical Evaluation of Gated Recurrent Neural Networks on Sequence
    Modeling <https://arxiv.org/pdf/1412.3555.pdf>`_

        .. math::
            u_t & = actGate(xu_{t} + W_u h_{t-1} + b_u)

            r_t & = actGate(xr_{t} + W_r h_{t-1} + b_r)

            m_t & = actNode(xm_t + W_c dot(r_t, h_{t-1}) + b_m)

            h_t & = dot((1-u_t), h_{t-1}) + dot(u_t, m_t)


    The inputs of gru unit includes :math:`z_t`, :math:`h_{t-1}`. In terms
    of the equation above, the :math:`z_t` is split into 3 parts -
    :math:`xu_t`, :math:`xr_t` and :math:`xm_t`. This means that in order to
    implement a full GRU unit operator for an input, a fully
    connected layer has to be applied, such that :math:`z_t = W_{fc}x_t`.

    The terms :math:`u_t` and :math:`r_t` represent the update and reset gates
    of the GRU cell. Unlike LSTM, GRU has one lesser gate. However, there is
    an intermediate candidate hidden output, which is denoted by :math:`m_t`.
    This layer has three outputs :math:`h_t`, :math:`dot(r_t, h_{t-1})`
    and concatenation of :math:`u_t`, :math:`r_t` and :math:`m_t`.

    Parameters:
        size (int): The input dimension value.
        param_attr(ParamAttr, optional): The parameter attribute for the learnable
            hidden-hidden weight matrix.

            **Note**:

                1. The shape of the weight matrix is :math:`[T, 3*D]`, where D is the hidden size.
                2. All elements in the weight matrix can be divided into two parts. The first
                   part are weights of the update gate and reset gate with shape :math:`[D, 2*D]`,
                   and the second part are weights for candidate hidden state with shape :math:`[D, D]`.


            If it is set to None or one attribute of ParamAttr, gru_unit will
            create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. The default
            value is None.
        bias_attr (ParamAttr|bool, optional): The parameter attribute for the bias
            of GRU.Note that the bias with :math:`[1, 3*D]` concatenates
            the bias in the update gate, reset gate and candidate calculations.
            If it is set to False, no bias will be applied to the update gate,
            reset gate and candidate calculations. If it is set to None or one
            attribute of ParamAttr, gru_unit will create ParamAttr as
            bias_attr. If the Initializer of the bias_attr is not set, the bias
            is initialized zero. The default value is None.
        activation (str): The activation type for cell (actNode).
                             The default value is 'tanh'.
        gate_activation (str): The activation type for gates (actGate).
                                  The default value is 'sigmoid'.
        dtype(str): The dtype of the layers. The data type can be set as
            'float32', 'float64'. The default value is 'float32'.

    Attribute:
        **weight** (Parameter): the learnable weights of this layer.

        **bias** (Parameter): the learnable bias of this layer.

    Returns:
        tuple: The hidden value, reset-hidden value and gate values. The hidden value
        is a 2-D tensor with shape  :math:`[T, D]` . The reset-hidden value is a
        2-D tensor with shape  :math:`[T, D]` . The gate value is a 2-D tensor with
        shape  :math:`[T, 3*D]`.

    Examples:

        .. code-block:: python

          import paddle.fluid as fluid
          import paddle.fluid.dygraph.base as base
          import numpy

          lod = [[2, 4, 3]]
          D = 5
          T = sum(lod[0])

          input = numpy.random.rand(T, 3 * D).astype('float32')
          hidden_input = numpy.random.rand(T, D).astype('float32')
          with fluid.dygraph.guard():
              x = numpy.random.random((3, 32, 32)).astype('float32')
              gru = fluid.dygraph.GRUUnit(size=D * 3)
              dy_ret = gru(
                base.to_variable(input), base.to_variable(hidden_input))

    """

    def __init__(
        self,
        size,
        param_attr=None,
        bias_attr=None,
        activation='tanh',
        gate_activation='sigmoid',
        origin_mode=False,
        dtype='float32',
    ):
        super().__init__()
        self._bias_attr = bias_attr
        activation_dict = dict(
            identity=0,
            sigmoid=1,
            tanh=2,
            relu=3,
        )
        self.activation = activation_dict[activation]
        self.gate_activation = activation_dict[gate_activation]

        self._dtype = dtype
        size = size // 3
        # create weight
        self.weight = self.create_parameter(
            attr=param_attr, shape=[size, 3 * size], dtype=dtype
        )

        # create bias
        bias_size = [1, 3 * size]
        self._bias_size = bias_size
        self.bias = self.create_parameter(
            attr=bias_attr, shape=bias_size, dtype=dtype, is_bias=True
        )

    def forward(self, input, hidden):
        if _non_static_mode():
            gate, reset_hidden_pre, updated_hidden = _legacy_C_ops.gru_unit(
                input,
                hidden,
                self.weight,
                self.bias,
                'activation',
                self.activation,
                'gate_activation',
                self.gate_activation,
            )
            return updated_hidden, reset_hidden_pre, gate

        check_variable_and_dtype(
            input, 'input', ['float32', 'float64'], 'GRUUnit'
        )
        check_variable_and_dtype(
            hidden, 'hidden', ['float32', 'float64'], 'GRUUnit'
        )
        inputs = {
            'Input': [input],
            'HiddenPrev': [hidden],
            'Weight': [self.weight],
        }
        if self.bias is not None:
            inputs['Bias'] = [self.bias]
        gate = self._helper.create_variable_for_type_inference(self._dtype)
        reset_hidden_pre = self._helper.create_variable_for_type_inference(
            self._dtype
        )
        updated_hidden = self._helper.create_variable_for_type_inference(
            self._dtype
        )
        self._helper.append_op(
            type='gru_unit',
            inputs=inputs,
            outputs={
                'Gate': gate,
                'ResetHiddenPrev': reset_hidden_pre,
                'Hidden': updated_hidden,
            },
            attrs={
                'activation': self.activation,
                'gate_activation': self.gate_activation,
            },
        )

        return updated_hidden, reset_hidden_pre, gate


class NCE(layers.Layer):
    """
    This interface is used to construct a callable object of the ``NCE`` class.
    For more details, refer to code examples.
    It implements the function of the ``NCE`` loss function.
    By default this function uses a uniform distribution for sampling, and it
    compute and return the noise-contrastive estimation training loss. See
    `Noise-contrastive estimation: A new estimation principle for unnormalized statistical models <http://www.jmlr.org/proceedings/papers/v9/gutmann10a/gutmann10a.pdf>`_ .

    Parameters:
        num_total_classes (int): Total number of classes in all samples.
        dim (int): Dimension of input (possibly embedding dim).
        param_attr (ParamAttr, optional): The parameter attribute for learnable weights(Parameter)
             of nce. If it is set to None or one attribute of ParamAttr, nce
             will create ParamAttr as param_attr. If the Initializer of the param_attr
             is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr or bool, optional): The attribute for the bias of nce.
             If it is set to False, no bias will be added to the output units.
             If it is set to None or one attribute of ParamAttr, nce
             will create ParamAttr as bias_attr. If the Initializer of the bias_attr
             is not set, the bias is initialized zero. Default: None.
        num_neg_samples (int, optional): The number of negative classes. The default value is 10.
        sampler (str, optional): The sampler used to sample class from negative classes.
                       It can be 'uniform', 'log_uniform' or 'custom_dist'.
                       default: 'uniform'.
        custom_dist (float[], optional): A float[] with size=num_total_classes.
                       It is used when sampler is set to 'custom_dist'.
                       custom_dist[i] is the probability of i-th class to be sampled.
                       Default: None.
        seed (int, optional): The seed used in sampler. Default: 0.
        is_sparse(bool, optional): The flag indicating whether to use sparse update. If is_sparse is True, the weight@GRAD and bias@GRAD will be changed to SelectedRows. Default: False.
        dtype (str, optional): Data type, it can be "float32" or "float64". Default: "float32".

    Attribute:
        **weight** (Parameter): the learnable weights of this layer.

        **bias** (Parameter or None): the learnable bias of this layer.

    Returns:
        None

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle.fluid as fluid

            window_size = 5
            dict_size = 20
            label_word = int(window_size // 2) + 1
            inp_word = np.array([[1], [2], [3], [4], [5]]).astype('int64')
            nid_freq_arr = np.random.dirichlet(np.ones(20) * 1000).astype('float32')

            with fluid.dygraph.guard():
                words = []
                for i in range(window_size):
                    words.append(fluid.dygraph.base.to_variable(inp_word[i]))

                emb = fluid.Embedding(
                    size=[dict_size, 32],
                    param_attr='emb.w',
                    is_sparse=False)

                embs3 = []
                for i in range(window_size):
                    if i == label_word:
                        continue

                    emb_rlt = emb(words[i])
                    embs3.append(emb_rlt)

                embs3 = fluid.layers.concat(input=embs3, axis=1)
                nce = fluid.NCE(
                             num_total_classes=dict_size,
                             dim=embs3.shape[1],
                             num_neg_samples=2,
                             sampler="custom_dist",
                             custom_dist=nid_freq_arr.tolist(),
                             seed=1,
                             param_attr='nce.w',
                             bias_attr='nce.b')

                wl = fluid.layers.unsqueeze(words[label_word], axes=[0])
                nce_loss3 = nce(embs3, wl)

    """

    def __init__(
        self,
        num_total_classes,
        dim,
        sample_weight=None,
        param_attr=None,
        bias_attr=None,
        num_neg_samples=None,
        sampler="uniform",
        custom_dist=None,
        seed=0,
        is_sparse=False,
        dtype='float32',
    ):
        super().__init__()
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._num_total_classes = num_total_classes
        self._dtype = dtype
        self._inputs = dict()
        self._inputs['SampleWeight'] = (
            sample_weight if sample_weight is not None else []
        )
        if sampler == "uniform":
            sampler = 0
        elif sampler == "log_uniform":
            sampler = 1
        elif sampler == "custom_dist":
            assert custom_dist is not None
            # assert isinstance(custom_dist, Variable)

            custom_dist_len = len(custom_dist)
            alias_probs_ = [0] * custom_dist_len
            alias_ = [0] * custom_dist_len
            bigs = []
            littles = []
            for i in range(custom_dist_len):
                normal_prob = custom_dist[i] * custom_dist_len
                if normal_prob - 1.0 > 0:
                    bigs.append((i, normal_prob))
                elif 1.0 - normal_prob > 0:
                    littles.append((i, normal_prob))
                else:
                    alias_probs_[i] = normal_prob
                    alias_[i] = -1

            while len(bigs) and len(littles):
                big = bigs.pop(0)
                little = littles.pop(0)

                big_idx = big[0]
                big_prob = big[1]

                alias_probs_[little[0]] = little[1]
                alias_[little[0]] = big_idx
                big_left = big[1] + little[1] - 1
                if big_left - 1.0 > 0:
                    bigs.append((big_idx, big_left))
                elif 1.0 - big_left > 0:
                    littles.append((big_idx, big_left))
                else:
                    alias_probs_[big_idx] = big_left
                    alias_[big_idx] = -1

            if len(bigs):
                big = bigs.pop(0)
                alias_probs_[big[0]] = 1.0
                alias_[big[0]] = -1
            if len(littles):
                little = littles.pop(0)
                alias_probs_[little[0]] = 1.0
                alias_[little[0]] = -1

            def _init_by_numpy_array(numpy_array):
                ret = self.create_parameter(
                    attr=ParamAttr(),
                    shape=numpy_array.shape,
                    dtype=numpy_array.dtype,
                    default_initializer=NumpyArrayInitializer(numpy_array),
                )
                ret.stop_gradient = True
                return ret

            self._inputs['CustomDistProbs'] = _init_by_numpy_array(
                np.array(custom_dist).astype('float32')
            )
            self._inputs['CustomDistAlias'] = _init_by_numpy_array(
                np.array(alias_).astype('int32')
            )
            self._inputs['CustomDistAliasProbs'] = _init_by_numpy_array(
                np.array(alias_probs_).astype('float32')
            )
            sampler = 2
        else:
            raise Exception("Unsupported sampler type.")

        if num_neg_samples is None:
            num_neg_samples = 10
        else:
            num_neg_samples = int(num_neg_samples)
        self._num_neg_samples = num_neg_samples
        remote_prefetch = is_sparse
        print(
            "With sparse mode, if your models has only small parameter prefetch may cause speed down"
        )
        self._attrs = {
            'num_total_classes': int(num_total_classes),
            'num_neg_samples': num_neg_samples,
            'seed': seed,
            'sampler': sampler,
            'is_sparse': is_sparse,
            'remote_prefetch': remote_prefetch,
        }

        self.weight = self.create_parameter(
            attr=self._param_attr,
            shape=[self._num_total_classes, dim],
            is_bias=False,
            dtype=self._dtype,
        )
        if self._bias_attr:
            self.bias = self.create_parameter(
                attr=self._bias_attr,
                shape=[self._num_total_classes, 1],
                is_bias=True,
                dtype=self._dtype,
            )
            self._inputs['Bias'] = self.bias
        self._inputs['Weight'] = self.weight

    def forward(self, input, label, sample_weight=None):
        if _non_static_mode():
            attrs = (
                'num_total_classes',
                self._attrs['num_total_classes'],
                'num_neg_samples',
                self._attrs['num_neg_samples'],
                'seed',
                self._attrs['seed'],
                'sampler',
                self._attrs['sampler'],
                'is_sparse',
                self._attrs['is_sparse'],
                'remote_prefetch',
                self._attrs['remote_prefetch'],
            )
            cost, _, _ = _legacy_C_ops.nce(
                input,
                label,
                self.weight,
                self.bias,
                self._inputs['SampleWeight'],
                self._inputs['CustomDistProbs'],
                self._inputs['CustomDistAlias'],
                self._inputs['CustomDistAliasProbs'],
                *attrs
            )
            return cost / (self._num_neg_samples + 1)

        check_variable_and_dtype(input, "input", ['float32', 'float64'], "NCE")
        check_variable_and_dtype(label, "label", ['int64'], "NCE")
        check_type(
            sample_weight, 'sample_weight', (Variable, type(None)), 'NCE'
        )
        assert isinstance(input, Variable)
        assert isinstance(label, Variable)

        self._inputs['Input'] = input
        self._inputs['Label'] = label
        self._inputs['SampleWeight'] = (
            sample_weight if sample_weight is not None else []
        )

        cost = self._helper.create_variable_for_type_inference(
            dtype=input.dtype
        )
        sample_logits = self._helper.create_variable_for_type_inference(
            dtype=input.dtype
        )
        sample_labels = self._helper.create_variable_for_type_inference(
            dtype=label.dtype
        )

        self._helper.append_op(
            type='nce',
            inputs=self._inputs,
            outputs={
                'Cost': cost,
                'SampleLogits': sample_logits,
                'SampleLabels': sample_labels,
            },
            attrs=self._attrs,
        )
        return cost / (self._num_neg_samples + 1)


class PRelu(layers.Layer):
    r"""
    This interface is used to construct a callable object of the ``PRelu`` class.
    For more details, refer to code examples.
    It implements three activation methods of the ``PRelu`` activation function.

    Equation:

    .. math::
        y = \max(0, x) + \\alpha * \min(0, x)

    Parameters:
        mode (str): The mode for weight sharing. It supports all, channel
          and element. all: all elements share same weight
          channel:elements in a channel share same weight
          element:each element has a weight
        channel (int, optional): The number of channels.
          This argument is required when mode is "channel".
          Default: None.
        input_shape (list or tuple, optional): The shape of input.
          This argument is required when mode is "element".
          Default: None.
        param_attr(ParamAttr, optional): The parameter attribute for the learnable
          weight (alpha). Default: None.
        dtype (str, optional): Data type, it can be "float32" or "float64". Default: "float32".

    Attribute:
        **weight** (Parameter): the learnable weights of this layer.

    Returns:
        None

    Examples:

        .. code-block:: python

          import paddle.fluid as fluid
          from paddle.fluid.dygraph.base import to_variable
          import numpy as np

          inp_np = np.ones([5, 200, 100, 100]).astype('float32')
          with fluid.dygraph.guard():
              inp_np = to_variable(inp_np)
              prelu0 = fluid.PRelu(
                 mode='all',
                 param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(1.0)))
              dy_rlt0 = prelu0(inp_np)
              prelu1 = fluid.PRelu(
                 mode='channel',
                 channel=200,
                 param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(1.0)))
              dy_rlt1 = prelu1(inp_np)
              prelu2 = fluid.PRelu(
                 mode='element',
                 input_shape=inp_np.shape,
                 param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(1.0)))
              dy_rlt2 = prelu2(inp_np)

    """

    def __init__(
        self,
        mode,
        channel=None,
        input_shape=None,
        param_attr=None,
        dtype='float32',
    ):
        # need specify name_scope since snake-cased 'PRelu' is 'p_relu'
        super().__init__(name_scope='prelu')
        self._mode = mode
        self._param_attr = param_attr
        self._dtype = dtype
        if mode == 'all':
            self._alpha_shape = [1]
        elif mode == 'channel':
            assert isinstance(
                channel, int
            ), "channel argument is required when mode is 'channel'."
            # NOTE(zhiqiu): The _alpha_shape should be [1, channel] + [1] * len(input_shape[2:]), not [1, channel, 1, 1].
            # However, the suffix 1 in the list is useless, since the tensor is viewed as one demension array during kernel calculation.
            # And, input_shape is not required when mode is 'channel', so it is simplified.
            # NOTE(zhiqiu): Revert shape to [1, channel, 1, 1] for compatibility with saved model of old version.
            self._alpha_shape = [1, channel, 1, 1]
        elif mode == 'element':
            assert isinstance(
                input_shape, (list, tuple)
            ), "input_shape argument is required when mode is 'element'."
            self._alpha_shape = [1] + list(input_shape)[1:]
        else:
            raise ValueError('mode should be one of all, channel, element.')
        self.weight = self.create_parameter(
            attr=self._param_attr,
            shape=self._alpha_shape,
            dtype='float32',
            is_bias=False,
            default_initializer=Constant(1.0),
        )

    def forward(self, input):
        if in_dygraph_mode():
            return _C_ops.prelu(input, self.weight, "NCHW", self._mode)

        check_variable_and_dtype(input, 'input', ['float32'], 'PRelu')
        out = self._helper.create_variable_for_type_inference(self._dtype)
        self._helper.append_op(
            type="prelu",
            inputs={"X": input, 'Alpha': self.weight},
            attrs={"mode": self._mode},
            outputs={"Out": out},
        )
        return out


class BilinearTensorProduct(layers.Layer):
    r"""

    **Add Bilinear Tensor Product Layer**

    This layer performs bilinear tensor product on two inputs.
    For example:

    .. math::
      out_{i} = x * W_{i} * {y^\mathrm{T}}, i=0,1,...,size-1

    In this formula:
     - :math:`x`: the first input contains M elements, shape is [batch_size, M].
     - :math:`y`: the second input contains N elements, shape is [batch_size, N].
     - :math:`W_{i}`: the i-th learned weight, shape is [M, N]
     - :math:`out_{i}`: the i-th element of out, shape is [batch_size, size].
     - :math:`y^\mathrm{T}`: the transpose of :math:`y`.

    Parameters:
       input1_dim (int): The dimension of each first input.
       input2_dim (int): The dimension of each second input.
       output_dim (int): The dimension of output of this layer.
       name (str, optional): The default value is None. Normally there is no need for user
           to set this property. For more information, please refer to :ref:`api_guide_Name`. Default: None.
       act (str, optional): Activation to be applied to the output of this layer. The default value is None.
       param_attr (ParamAttr, optional): The parameter attribute for the learnable w, parameters/weights of
           this layer. The default value is None.
       bias_attr (ParamAttr, optional): The parameter attribute for the bias
           of this layer. If it is set to False, no bias will be added to the output units.
           If it is set to None, the bias is initialized zero. The default value is None.
       dtype (str, optional): Data type, it can be "float32" or "float64". Default: "float32".

    Attribute:
        **weight** (Parameter): the learnable weights of this layer.

        **bias** (Parameter): the learnable bias of this layer.

    Returns:
       Tensor: A 2-D Tensor of shape [batch_size, size].

    Examples:
       .. code-block:: python

        import paddle
        import numpy

        layer1 = numpy.random.random((5, 5)).astype('float32')
        layer2 = numpy.random.random((5, 4)).astype('float32')
        bilinearTensorProduct = paddle.nn.BilinearTensorProduct(
            input1_dim=5, input2_dim=4, output_dim=1000)
        ret = bilinearTensorProduct(paddle.to_tensor(layer1),
                                    paddle.to_tensor(layer2))

    """

    def __init__(
        self,
        input1_dim,
        input2_dim,
        output_dim,
        name=None,
        act=None,
        param_attr=None,
        bias_attr=None,
        dtype='float32',
    ):
        super().__init__()
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._act = act
        self._name = name
        self._input1_dim = input1_dim
        self._input2_dim = input2_dim
        self._output_dim = output_dim
        self._inputs = dict()
        self._dtype = dtype

        param_shape = [self._output_dim, self._input1_dim, self._input2_dim]
        self.weight = self.create_parameter(
            attr=self._param_attr,
            shape=param_shape,
            dtype=self._dtype,
            is_bias=False,
        )
        bias_size = [1, self._output_dim]
        self.bias = self.create_parameter(
            attr=self._bias_attr,
            shape=bias_size,
            dtype=self._dtype,
            is_bias=True,
        )

    @deprecated(
        since="2.0.0",
        update_to="paddle.nn.Bilinear",
        reason="New name and new args in Bilinear, easier to use.",
    )
    def forward(self, x, y):
        check_variable_and_dtype(
            x, 'x', ['float32', 'float64'], 'BilinearTensorProduct'
        )
        check_variable_and_dtype(
            y, 'y', ['float32', 'float64'], 'BilinearTensorProduct'
        )
        self._inputs = {"X": x, "Y": y, "Weight": self.weight}
        if self.bias is not None:
            self._inputs["Bias"] = self.bias
        if self._name is not None:
            out = self._helper.create_variable(
                name=".".join([self.full_name(), self._name]),
                dtype=self._dtype,
                persistable=False,
            )
        else:
            out = self._helper.create_variable(
                dtype=self._dtype, persistable=False
            )
        self._helper.append_op(
            type="bilinear_tensor_product",
            inputs=self._inputs,
            outputs={"Out": out},
        )

        # add activation
        return self._helper.append_activation(out, act=self._act)


class Conv2DTranspose(layers.Layer):
    r"""
    This interface is used to construct a callable object of the ``Conv2DTranspose`` class.
    For more details, refer to code examples.
    The convolution2D transpose layer calculates the output based on the input,
    filter, and dilations, strides, paddings. Input and output
    are in NCHW format. Where N is batch size, C is the number of feature map,
    H is the height of the feature map, and W is the width of the feature map.
    Filter's shape is [MCHW] , where M is the number of input feature map,
    C is the number of output feature map, H is the height of the filter,
    and W is the width of the filter. If the groups is greater than 1,
    C will equal the number of input feature map divided by the groups.
    If bias attribution and activation type are provided, bias is added to
    the output of the convolution, and the corresponding activation function
    is applied to the final result.
    The details of convolution transpose layer, please refer to the following explanation and references
    `conv2dtranspose <http://www.matthewzeiler.com/wp-content/uploads/2017/07/cvpr2010.pdf>`_ .

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \\ast X + b)

    Where:

    * :math:`X`: Input value, a ``Tensor`` with NCHW format.
    * :math:`W`: Filter value, a ``Tensor`` with shape [MCHW] .
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D ``Tensor`` with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{in}, C_{out}, H_f, W_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        .. math::

           H^\prime_{out} &= (H_{in} - 1) * strides[0] - 2 * paddings[0] + dilations[0] * (H_f - 1) + 1 \\\\
           W^\prime_{out} &= (W_{in} - 1) * strides[1] - 2 * paddings[1] + dilations[1] * (W_f - 1) + 1 \\\\
           H_{out} &\in [ H^\prime_{out}, H^\prime_{out} + strides[0] ) \\\\
           W_{out} &\in [ W^\prime_{out}, W^\prime_{out} + strides[1] )

    Parameters:
        num_channels(int): The number of channels in the input image.
        num_filters(int): The number of the filter. It is as same as the output
            feature map.
        filter_size(int or tuple): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_H, filter_size_W).
            Otherwise, the filter will be a square.
        output_size(int or tuple, optional): The output image size. If output size is a
            tuple, it must contain two integers, (image_H, image_W). None if use
            filter_size, padding, and stride to calculate output_size.
            if output_size and filter_size are specified at the same time, They
            should follow the formula above. Default: None.
        padding(int or tuple, optional): The padding size. If padding is a tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: 0.
        stride(int or tuple, optional): The stride size. If stride is a tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: 1.
        dilation(int or tuple, optional): The dilation size. If dilation is a tuple, it must
            contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. Default: 1.
        groups(int, optional): The groups number of the Conv2D transpose layer. Inspired by
            grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
            when group=2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            Default: 1.
        param_attr (ParamAttr, optional): The parameter attribute for learnable weights(Parameter)
            of conv2d_transpose. If it is set to None or one attribute of ParamAttr, conv2d_transpose
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr or bool, optional): The attribute for the bias of conv2d_transpose.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d_transpose
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn(bool, optional): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True.
        act (str, optional): Activation type, if it is set to None, activation is not appended.
            Default: None.
        dtype (str, optional): Data type, it can be "float32" or "float64". Default: "float32".

    Attribute:
        **weight** (Parameter): the learnable weights of filters of this layer.

        **bias** (Parameter or None): the learnable bias of this layer.

    Returns:
        None

    Examples:
       .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np

          with fluid.dygraph.guard():
              data = np.random.random((3, 32, 32, 5)).astype('float32')
              conv2DTranspose = fluid.dygraph.nn.Conv2DTranspose(
                    num_channels=32, num_filters=2, filter_size=3)
              ret = conv2DTranspose(fluid.dygraph.base.to_variable(data))

    """

    def __init__(
        self,
        num_channels,
        num_filters,
        filter_size,
        output_size=None,
        padding=0,
        stride=1,
        dilation=1,
        groups=None,
        param_attr=None,
        bias_attr=None,
        use_cudnn=True,
        act=None,
        dtype='float32',
    ):
        super().__init__()
        assert (
            param_attr is not False
        ), "param_attr should not be False in conv2d_transpose."
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._act = act
        self._groups = groups
        self._num_channels = num_channels
        self._num_filters = num_filters
        self._use_cudnn = use_cudnn
        self._padding = padding
        self._stride = stride
        self._dilation = dilation
        self._filter_size = filter_size
        self._output_size = output_size
        self._dtype = dtype

        if (
            self._num_channels == self._groups
            and self._num_filters == self._num_channels
            and not self._use_cudnn
        ):
            self._op_type = 'depthwise_conv2d_transpose'
        else:
            self._op_type = 'conv2d_transpose'

        self._padding = utils.convert_to_list(self._padding, 2, 'padding')
        self._stride = utils.convert_to_list(self._stride, 2, 'stride')
        self._dilation = utils.convert_to_list(self._dilation, 2, 'dilation')

        self._filter_size = utils.convert_to_list(
            self._filter_size, 2, 'conv2d_transpose.filter_size'
        )

        if self._output_size is None:
            self._output_size = []
        elif isinstance(self._output_size, list):
            if utils._contain_var(self._output_size):
                self._output_size = utils._convert_to_tensor_list(
                    self._output_size
                )
            else:
                self._output_size = utils.convert_to_list(
                    self._output_size, 2, 'output_size'
                )
        elif isinstance(self._output_size, int):
            self._output_size = utils.convert_to_list(
                self._output_size, 2, 'output_size'
            )
        elif isinstance(self._output_size, Variable):
            check_dtype(
                self._output_size.dtype,
                'output_size',
                ['int32', 'int64'],
                'Conv2DTranspose',
            )
            if len(self._output_size.shape) == 1 and (
                self._output_size.shape[0] == 1
                or self._output_size.shape[0] == 2
            ):
                if self._output_size.shape[0] == 1:
                    self._output_size = [self._output_size, self._output_size]
            else:
                raise ValueError(
                    "output_size must contain one or two integers."
                )
        else:
            raise ValueError("output_size should be list or int or Tensor")
        self._padding = utils.convert_to_list(self._padding, 2, 'padding')
        self._groups = 1 if self._groups is None else self._groups
        filter_shape = [
            self._num_channels,
            self._num_filters // self._groups,
        ] + self._filter_size

        self.weight = self.create_parameter(
            dtype=self._dtype, shape=filter_shape, attr=self._param_attr
        )

        self.bias = self.create_parameter(
            attr=self._bias_attr,
            shape=[self._num_filters],
            dtype=self._dtype,
            is_bias=True,
        )

    def forward(self, input):
        if _non_static_mode():
            op = getattr(_legacy_C_ops, self._op_type)
            out = op(
                input,
                self.weight,
                'output_size',
                self._output_size,
                'strides',
                self._stride,
                'paddings',
                self._padding,
                'dilations',
                self._dilation,
                'groups',
                self._groups,
                'use_cudnn',
                self._use_cudnn,
            )
            pre_bias = out
            pre_act = dygraph_utils._append_bias_in_dygraph(
                pre_bias, self.bias, 1
            )
            return dygraph_utils._append_activation_in_dygraph(
                pre_act, act=self._act
            )

        check_variable_and_dtype(
            input, 'input', ['float16', 'float32', 'float64'], "Conv2DTranspose"
        )

        inputs = {'Input': [input], 'Filter': [self.weight]}
        attrs = {
            'output_size': self._output_size,
            'strides': self._stride,
            'paddings': self._padding,
            'dilations': self._dilation,
            'groups': self._groups,
            'use_cudnn': self._use_cudnn,
        }

        pre_bias = self._helper.create_variable_for_type_inference(
            dtype=input.dtype
        )
        self._helper.append_op(
            type=self._op_type,
            inputs=inputs,
            outputs={'Output': pre_bias},
            attrs=attrs,
        )

        if self.bias is not None:
            pre_act = self._helper.create_variable_for_type_inference(
                dtype=self._dtype
            )
            self._helper.append_op(
                type='elementwise_add',
                inputs={'X': [pre_bias], 'Y': [self.bias]},
                outputs={'Out': [pre_act]},
                attrs={'axis': 1},
            )
        else:
            pre_act = pre_bias

        out = self._helper.append_activation(pre_act, act=self._act)
        return out


class SequenceConv(layers.Layer):
    """
    This function creates the op for sequence_conv, using the inputs and
    other convolutional configurations for the filters and stride as given
    in the input parameters to the function.

    Parameters:
        name_scope(str): The name of this class.
        num_filters (int): number of filters.
        filter_size (int): the filter size (H and W). Default: 3.
        filter_stride (int): stride of the filter. Default: 1.
        padding (bool|None): if True, add paddings. Default: None
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias of sequence_conv.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, sequence_conv
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        param_attr (ParamAttr|None): The parameter attribute for learnable parameters/weights
            of sequence_conv. If it is set to None or one attribute of ParamAttr, sequence_conv
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        act (str): Activation type, if it is set to None, activation is not appended.
            Default: None.

    Attributes:
        weight (Parameter): the learnable weights of filters of this layer.
        bias (Parameter|None): the learnable bias of this layer.

    Returns:
        Variable: output of sequence_conv
    """

    def __init__(
        self,
        name_scope,
        num_filters,
        filter_size=3,
        filter_stride=1,
        padding=None,
        bias_attr=None,
        param_attr=None,
        act=None,
    ):
        assert (
            not _non_static_mode()
        ), "SequenceConv is not supported by dynamic graph mode yet!"
        super().__init__(name_scope)
        self._num_filters = num_filters
        self._filter_size = filter_size
        self._filter_stride = filter_stride
        self._padding = padding
        self._bias_attr = bias_attr
        self._param_attr = param_attr
        self._act = act

    def _build_once(self, input):
        self._dtype = self._helper.input_dtype(input)
        filter_shape = [self._filter_size * input.shape[1], self._num_filters]
        self.weight = self.create_parameter(
            attr=self._param_attr, shape=filter_shape, dtype=self._dtype
        )

        self.bias = self.create_parameter(
            attr=self._bias_attr,
            shape=[self._num_filters],
            dtype=self._dtype,
            is_bias=True,
        )

    def forward(self, input):
        pre_bias = self._helper.create_variable_for_type_inference(self._dtype)
        self._helper.append_op(
            type='sequence_conv',
            inputs={
                'X': [input],
                'Filter': [self.weight],
            },
            outputs={"Out": pre_bias},
            attrs={
                'contextStride': self._filter_stride,
                'contextStart': -int(self._filter_size // 2),
                'contextLength': self._filter_size,
            },
        )

        if self.bias is not None:
            pre_act = self._helper.create_variable_for_type_inference(
                dtype=self._dtype
            )
            self._helper.append_op(
                type='elementwise_add',
                inputs={'X': [pre_bias], 'Y': [self.bias]},
                outputs={'Out': [pre_act]},
                attrs={'axis': 1},
            )
        else:
            pre_act = pre_bias

        return self._helper.append_activation(pre_act, act=self._act)


class RowConv(layers.Layer):
    """
    ***Row-convolution operator***

    The row convolution is called lookahead convolution.  This operator was introduced in the following paper for DeepSpeech2:
    http://www.cs.cmu.edu/~dyogatam/papers/wang+etal.iclrworkshop2016.pdf

    The main motivation is that a bidirectional RNN, useful in DeepSpeech like speech models, learns representation for a sequence by performing a
    forward and a backward pass through the entire sequence. However, unlike
    unidirectional RNNs, bidirectional RNNs are challenging to deploy in an online
    and low-latency setting. The lookahead convolution incorporates information
    from future subsequences in a computationally efficient manner to improve
    unidirectional recurrent neural networks. The row convolution operator is
    different from the 1D sequence convolution, and is computed as follows:

    Given an input sequence X of length t and input dimension D, and a filter (W) of size context * D.

    More details about row_conv please refer to the design document https://github.com/PaddlePaddle/Paddle/issues/2228#issuecomment-303903645 .

    Parameters:
        name_scope(str): The name of this class.
        future_context_size (int): Future context size. Please note, the shape
            of convolution kernel is [future_context_size + 1, D].
        param_attr (ParamAttr): Attributes of parameters, including
            name, initializer etc. Default: None.
        act (str): Non-linear activation to be applied to output variable. Default: None.

    Attributes:
        weight (Parameter): the learnable weights of this layer.

    Returns:
        the output(Out) is a LodTensor, which supports variable time-length input sequences.
        The underlying tensor in this LodTensor is a matrix with shape T x N, i.e., the same shape as X.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy

          with fluid.dygraph.guard():
              x = numpy.random.random((16)).astype('float32')
              rowConv = fluid.dygraph.nn.RowConv(
                    'RowConv', future_context_size=2)
              ret = rowConv(fluid.dygraph.base.to_variable(x))

    """

    def __init__(
        self, name_scope, future_context_size, param_attr=None, act=None
    ):
        assert (
            not _non_static_mode()
        ), "RowConv is not supported by dynamic graph mode yet!"
        super().__init__(name_scope)
        self._act = act
        self._param_attr = param_attr
        self._future_context_size = future_context_size

    def _build_once(self, input):
        self._dtype = self._helper.input_dtype(input)
        filter_shape = [self._future_context_size + 1, input.shape[1]]
        self.weight = self.create_parameter(
            attr=self._param_attr,
            shape=filter_shape,
            dtype=self._dtype,
            is_bias=False,
        )

    def forward(self, input):
        out = self._helper.create_variable_for_type_inference(self._dtype)
        self._helper.append_op(
            type='row_conv',
            inputs={'X': [input], 'Filter': [self.weight]},
            outputs={'Out': [out]},
        )
        return self._helper.append_activation(out, act=self._act)


class GroupNorm(layers.Layer):
    """
    :alias_main: paddle.nn.GroupNorm
        :alias: paddle.nn.GroupNorm,paddle.nn.layer.GroupNorm,paddle.nn.layer.norm.GroupNorm
        :old_api: paddle.fluid.dygraph.GroupNorm

    This interface is used to construct a callable object of the ``GroupNorm`` class.
    For more details, refer to code examples.
    It implements the function of the Group Normalization Layer.
    Refer to `Group Normalization <https://arxiv.org/abs/1803.08494>`_ .

    Parameters:
        channels(int): The number of channels of input.
        groups(int): The number of groups that divided from channels.
        epsilon(float, optional): The small value added to the variance to prevent
                                  division by zero. Default: 1e-05.
        param_attr(ParamAttr, optional): The parameter attribute for the learnable
                                         scale :math:`g`. If it is set to False, no scale will be added to the output units.
                                         If it is set to None, the bias is initialized one. Default: None.
        bias_attr(ParamAttr, optional): The parameter attribute for the learnable
                                        bias :math:`b`. If it is set to False, no bias will be added to the output units.
                                        If it is set to None, the bias is initialized zero. Default: None.
        act(str, optional): Activation to be applied to the output of group normalization. Default: None.
        data_layout(str, optional): Specify the input data format. Only NCHW is supported. Default: NCHW.

    Returns:
        None

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np

          with fluid.dygraph.guard():
              x = np.random.random((8, 32, 32)).astype('float32')
              groupNorm = fluid.dygraph.nn.GroupNorm(channels=32, groups=4)
              ret = groupNorm(fluid.dygraph.base.to_variable(x))

    """

    def __init__(
        self,
        channels,
        groups,
        epsilon=1e-05,
        param_attr=None,
        bias_attr=None,
        act=None,
        data_layout='NCHW',
        dtype='float32',
    ):
        super().__init__()
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._epsilon = epsilon
        self._channels = channels
        self._groups = groups
        self._act = act
        self._dtype = dtype
        if data_layout != 'NCHW':
            raise ValueError("unsupported data layout:" + data_layout)

        param_shape = [self._channels]

        self.weight = self.create_parameter(
            attr=self._param_attr or False,
            shape=param_shape,
            dtype=self._dtype,
            default_initializer=Constant(1.0),
        )

        self.bias = self.create_parameter(
            attr=self._bias_attr or False,
            shape=param_shape,
            dtype=self._dtype,
            is_bias=True,
        )

    def forward(self, input):
        mean_out = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True
        )
        variance_out = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True
        )
        if in_dygraph_mode():
            out = _C_ops.group_norm(
                input,
                self.weight,
                self.bias,
                self._epsilon,
                self._groups,
                "NCHW",
            )

            return dygraph_utils._append_activation_in_dygraph(out, self._act)

        elif _in_legacy_dygraph():
            attrs = ('epsilon', self._epsilon, 'groups', self._groups)
            out, _, _ = _legacy_C_ops.group_norm(
                input, self.weight, self.bias, mean_out, variance_out, *attrs
            )

            return dygraph_utils._append_activation_in_dygraph(out, self._act)
        else:
            inputs = {'X': input}
            if self.bias is not None:
                inputs['Bias'] = self.bias
            if self.weight is not None:
                inputs['Scale'] = self.weight

            # create output
            group_norm_out = self._helper.create_variable_for_type_inference(
                dtype=self._dtype
            )

            self._helper.append_op(
                type="group_norm",
                inputs=inputs,
                outputs={
                    "Y": group_norm_out,
                    "Mean": mean_out,
                    "Variance": variance_out,
                },
                attrs={"epsilon": self._epsilon, "groups": self._groups},
            )

            return self._helper.append_activation(group_norm_out, self._act)


class SpectralNorm(layers.Layer):
    r"""
    This interface is used to construct a callable object of the ``SpectralNorm`` class.
    For more details, refer to code examples. It implements the function of the Spectral Normalization Layer.
    This layer calculates the spectral normalization value of weight parameters of
    fc, conv1d, conv2d, conv3d layers which should be 2-D, 3-D, 4-D, 5-D
    Parameters. Calculations are showed as follows.

    Step 1:
    Generate vector U in shape of [H], and V in shape of [W].
    While H is the :attr:`dim` th dimension of the input weights,
    and W is the product result of remaining dimensions.

    Step 2:
    :attr:`power_iters` should be a positive integer, do following
    calculations with U and V for :attr:`power_iters` rounds.

    .. math::

        \mathbf{v} := \frac{\mathbf{W}^{T} \mathbf{u}}{\|\mathbf{W}^{T} \mathbf{u}\|_2}

        \mathbf{u} := \frac{\mathbf{W}^{T} \mathbf{v}}{\|\mathbf{W}^{T} \mathbf{v}\|_2}

    Step 3:
    Calculate :math:`\sigma(\mathbf{W})` and normalize weight values.

    .. math::

        \sigma(\mathbf{W}) = \mathbf{u}^{T} \mathbf{W} \mathbf{v}

        \mathbf{W} = \frac{\mathbf{W}}{\sigma(\mathbf{W})}


    Refer to `Spectral Normalization <https://arxiv.org/abs/1802.05957>`_ .

    Parameters:
        weight_shape(list or tuple): The shape of weight parameter.
        dim(int, optional): The index of dimension which should be permuted to the first before reshaping Input(Weight) to matrix, it should be set as 0 if Input(Weight) is the weight of fc layer, and should be set as 1 if Input(Weight) is the weight of conv layer. Default: 0.
        power_iters(int, optional): The number of power iterations to calculate spectral norm. Default: 1.
        eps(float, optional): The epsilon for numerical stability in calculating norms. Default: 1e-12.
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .
        dtype (str, optional): Data type, it can be "float32" or "float64". Default: "float32".

    Returns:
        None

    Examples:
       .. code-block:: python

            import paddle
            x = paddle.rand((2,8,32,32))

            spectral_norm = paddle.nn.SpectralNorm(x.shape, dim=1, power_iters=2)
            spectral_norm_out = spectral_norm(x)

            print(spectral_norm_out.shape) # [2, 8, 32, 32]

    """

    def __init__(
        self, weight_shape, dim=0, power_iters=1, eps=1e-12, dtype='float32'
    ):
        super().__init__()
        self._power_iters = power_iters
        self._eps = eps
        self._dim = dim
        self._dtype = dtype

        self._weight_shape = list(weight_shape)
        assert (
            np.prod(self._weight_shape) > 0
        ), "Any dimension of `weight_shape` cannot be equal to 0."
        assert dim < len(self._weight_shape), (
            "The input `dim` should be less than the "
            "length of `weight_shape`, but received dim="
            "{}".format(dim)
        )
        h = self._weight_shape[self._dim]
        w = np.prod(self._weight_shape) // h

        self.weight_u = self.create_parameter(
            attr=ParamAttr(),
            shape=[h],
            dtype=self._dtype,
            default_initializer=Normal(0.0, 1.0),
        )
        self.weight_u.stop_gradient = True

        self.weight_v = self.create_parameter(
            attr=ParamAttr(),
            shape=[w],
            dtype=self._dtype,
            default_initializer=Normal(0.0, 1.0),
        )
        self.weight_v.stop_gradient = True

    def forward(self, weight):
        if in_dygraph_mode():
            return _C_ops.spectral_norm(
                weight,
                self.weight_u,
                self.weight_v,
                self._dim,
                self._power_iters,
                self._eps,
            )

        check_variable_and_dtype(
            weight, "weight", ['float32', 'float64'], 'SpectralNorm'
        )
        inputs = {'Weight': weight, 'U': self.weight_u, 'V': self.weight_v}
        out = self._helper.create_variable_for_type_inference(self._dtype)
        self._helper.append_op(
            type="spectral_norm",
            inputs=inputs,
            outputs={
                "Out": out,
            },
            attrs={
                "dim": self._dim,
                "power_iters": self._power_iters,
                "eps": self._eps,
            },
        )

        return out


class TreeConv(layers.Layer):
    """
    This interface is used to construct a callable object of the ``TreeConv`` class.
    For more details, refer to code examples.
    Tree-Based Convolution is a kind of convolution based on tree structure.
    Tree-Based Convolution is a part of Tree-Based Convolution Neural Network(TBCNN),
    which is used to classify tree structures, such as Abstract Syntax Tree.
    Tree-Based Convolution proposed a kind of data structure called continuous binary tree,
    which regards multiway tree as binary tree.
    The paper of Tree-Based Convolution Operator is here: `tree-based convolution <https://arxiv.org/abs/1409.5718v1/>`_ .

    Parameters:
        feature_size(int): last dimension of nodes_vector.
        output_size(int): output feature width.
        num_filters(int, optional): number of filters, Default: 1.
        max_depth(int, optional): max depth of filters, Default: 2.
        act(str, optional): activation function, Default: tanh.
        param_attr(ParamAttr, optional): the parameter attribute for the filters, Default: None.
        bias_attr(ParamAttr, optional): the parameter attribute for the bias of this layer, Default: None.
        name(str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name` .
        dtype (str, optional): Data type, it can be "float32" or "float64". Default: "float32".

    Attribute:
        **weight** (Parameter): the learnable weights of filters of this layer.

        **bias** (Parameter or None): the learnable bias of this layer.

    Returns:
        None

    Examples:

        .. code-block:: python

          import paddle.fluid as fluid
          import numpy

          with fluid.dygraph.guard():
              nodes_vector = numpy.random.random((1, 10, 5)).astype('float32')
              edge_set = numpy.random.random((1, 9, 2)).astype('int32')
              treeConv = fluid.dygraph.nn.TreeConv(
                feature_size=5, output_size=6, num_filters=1, max_depth=2)
              ret = treeConv(fluid.dygraph.base.to_variable(nodes_vector), fluid.dygraph.base.to_variable(edge_set))
    """

    def __init__(
        self,
        feature_size,
        output_size,
        num_filters=1,
        max_depth=2,
        act='tanh',
        param_attr=None,
        bias_attr=None,
        name=None,
        dtype='float32',
    ):
        super().__init__()
        self._name = name
        self._feature_size = feature_size
        self._output_size = output_size
        self._act = act
        self._max_depth = max_depth
        self._num_filters = num_filters
        self._bias_attr = bias_attr
        self._param_attr = param_attr
        self._dtype = dtype
        w_shape = [self._feature_size, 3, self._output_size, self._num_filters]
        if self._bias_attr:
            self.bias = self.create_parameter(
                attr=self._bias_attr,
                shape=[self._num_filters],
                dtype=self._dtype,
                is_bias=True,
            )
        self.weight = self.create_parameter(
            attr=self._param_attr,
            shape=w_shape,
            dtype=self._dtype,
            is_bias=False,
        )

    def forward(self, nodes_vector, edge_set):
        check_type(nodes_vector, 'nodes_vector', (Variable), 'TreeConv')
        check_type(edge_set, 'edge_set', (Variable), 'TreeConv')
        if self._name:
            out = self.create_variable(
                name=self._name, dtype=self._dtype, persistable=False
            )
        else:
            out = self._helper.create_variable_for_type_inference(
                dtype=self._dtype
            )
        self._helper.append_op(
            type='tree_conv',
            inputs={
                'NodesVector': nodes_vector,
                'EdgeSet': edge_set,
                'Filter': self.weight,
            },
            outputs={
                'Out': out,
            },
            attrs={'max_depth': self._max_depth},
        )
        if self._bias_attr:
            pre_activation = self._helper.create_variable_for_type_inference(
                dtype=self._dtype
            )
            self._helper.append_op(
                type='elementwise_add',
                inputs={'X': [out], 'Y': [self.bias]},
                outputs={'Out': [pre_activation]},
                attrs={'axis': 1},
            )
        else:
            pre_activation = out
        return self._helper.append_activation(pre_activation, act=self._act)


class Flatten(layers.Layer):
    """
    This interface is used to construct a callable object of the ``FLatten`` class.
    For more details, refer to code examples.
    It implements flatten a contiguous range of dims into a tensor.

    Parameters:
        start_axis(int): first dim to flatten (default = 1)
        stop_axis(int): last dim to flatten (default = -1).

    Returns:
        None

    Examples:

        .. code-block:: python

          import paddle
          import numpy as np

          inp_np = np.ones([5, 2, 3, 4]).astype('float32')
          inp_np = paddle.to_tensor(inp_np)
          flatten = paddle.nn.Flatten(start_axis=1, stop_axis=2)
          flatten_res = flatten(inp_np)

    """

    def __init__(self, start_axis=1, stop_axis=-1):
        super().__init__()
        self.start_axis = start_axis
        self.stop_axis = stop_axis

    def forward(self, input):
        out = paddle.tensor.manipulation.flatten(
            input, start_axis=self.start_axis, stop_axis=self.stop_axis
        )
        return out
