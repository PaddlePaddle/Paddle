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

from __future__ import print_function

from six.moves import reduce

from .. import core
from ..layers import utils
from . import layers
from ..framework import Variable, OpProtoHolder, Parameter
from ..layers import layer_function_generator
from ..param_attr import ParamAttr
from ..initializer import Normal, Constant
__all__ = [
    'Conv2D', 'Pool2D', 'FC', 'BatchNorm', 'Embedding', 'GRUUnit', 'LayerNorm'
]


class Conv2D(layers.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=None,
                 use_cudnn=True,
                 act=None,
                 param_attr=None,
                 bias_attr=None,
                 dtype=core.VarDesc.VarType.FP32):
        assert param_attr is not False, "param_attr should not be False here."
        super(Conv2D, self).__init__(name_scope)
        self._groups = groups
        self._stride = utils.convert_to_list(stride, 2, 'stride')
        self._padding = utils.convert_to_list(padding, 2, 'padding')
        self._dilation = utils.convert_to_list(dilation, 2, 'dilation')
        self._act = act
        if not isinstance(use_cudnn, bool):
            raise ValueError("use_cudnn should be True or False")
        self._use_cudnn = use_cudnn
        self._num_channels = num_channels
        if (self._num_channels == self._groups and
                num_filters % self._num_channels == 0 and not self._use_cudnn):
            self._l_type = 'depthwise_conv2d'
        else:
            self._l_type = 'conv2d'

        if groups is None:
            num_filter_channels = num_channels
        else:
            if num_channels % groups != 0:
                raise ValueError("num_channels must be divisible by groups.")
            num_filter_channels = num_channels // groups
        filter_size = utils.convert_to_list(filter_size, 2, 'filter_size')
        filter_shape = [num_filters, int(num_filter_channels)] + filter_size

        def _get_default_param_initializer():
            filter_elem_num = filter_size[0] * filter_size[1] * num_channels
            std = (2.0 / filter_elem_num)**0.5
            return Normal(0.0, std, 0)

        self._filter_param = self.create_parameter(
            attr=param_attr,
            shape=filter_shape,
            dtype=self._dtype,
            default_initializer=_get_default_param_initializer())

        if self._use_cudnn:
            self.create_variable(
                name="kCUDNNFwdAlgoCache",
                persistable=True,
                type=core.VarDesc.VarType.RAW)
            self.create_variable(
                name="kCUDNNBwdDataAlgoCache",
                persistable=True,
                type=core.VarDesc.VarType.RAW)
            self.create_variable(
                name="kCUDNNBwdFilterAlgoCache",
                persistable=True,
                type=core.VarDesc.VarType.RAW)

        self._bias_param = self.create_parameter(
            attr=bias_attr,
            shape=[num_filters],
            dtype=self._dtype,
            is_bias=True)

    def forward(self, input):
        pre_bias = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)

        self._helper.append_op(
            type=self._l_type,
            inputs={
                'Input': input,
                'Filter': self._filter_param,
            },
            outputs={"Output": pre_bias},
            attrs={
                'strides': self._stride,
                'paddings': self._padding,
                'dilations': self._dilation,
                'groups': self._groups if self._groups else 1,
                'use_cudnn': self._use_cudnn,
                'use_mkldnn': False,
            })

        pre_act = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)

        self._helper.append_op(
            type='elementwise_add',
            inputs={'X': [pre_bias],
                    'Y': [self._bias_param]},
            outputs={'Out': [pre_act]},
            attrs={'axis': 1})

        # Currently, we don't support inplace in imperative mode
        return self._helper.append_activation(pre_act, act=self._act)


class Pool2D(layers.Layer):
    def __init__(self,
                 name_scope,
                 pool_size=-1,
                 pool_type="max",
                 pool_stride=1,
                 pool_padding=0,
                 global_pooling=False,
                 use_cudnn=True,
                 ceil_mode=False,
                 exclusive=True,
                 dtype=core.VarDesc.VarType.FP32):
        if pool_type not in ["max", "avg"]:
            raise ValueError(
                "Unknown pool_type: '%s'. It can only be 'max' or 'avg'.",
                str(pool_type))

        if global_pooling is False and pool_size == -1:
            raise ValueError(
                "When the global_pooling is False, pool_size must be passed "
                "and be a valid value. Received pool_size: " + str(pool_size))

        if not isinstance(use_cudnn, bool):
            raise ValueError("use_cudnn should be True or False")

        super(Pool2D, self).__init__(name_scope, dtype=dtype)

        self._pool_type = pool_type
        self._pool_size = utils.convert_to_list(pool_size, 2, 'pool_size')
        self._pool_padding = utils.convert_to_list(pool_padding, 2,
                                                   'pool_padding')
        self._pool_stride = utils.convert_to_list(pool_stride, 2, 'pool_stride')
        self._global_pooling = global_pooling
        self._use_cudnn = use_cudnn
        self._ceil_mode = ceil_mode
        self._exclusive = exclusive
        self._l_type = 'pool2d'

    def forward(self, input):
        pool_out = self._helper.create_variable_for_type_inference(self._dtype)

        self._helper.append_op(
            type=self._l_type,
            inputs={"X": input},
            outputs={"Out": pool_out},
            attrs={
                "pooling_type": self._pool_type,
                "ksize": self._pool_size,
                "global_pooling": self._global_pooling,
                "strides": self._pool_stride,
                "paddings": self._pool_padding,
                "use_cudnn": self._use_cudnn,
                "ceil_mode": self._ceil_mode,
                "use_mkldnn": False,
                "exclusive": self._exclusive,
            })
        return pool_out


class FC(layers.Layer):
    def __init__(self,
                 name_scope,
                 size,
                 param_attr=None,
                 bias_attr=None,
                 num_flatten_dims=1,
                 dtype=core.VarDesc.VarType.FP32,
                 act=None):
        super(FC, self).__init__(name_scope)

        self._size = size
        self._num_flatten_dims = num_flatten_dims
        self._dtype = dtype
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._act = act
        self.__w = list()

    @property
    def _w(self, i=0):
        return self.__w[i]

    @_w.setter
    def _w(self, value, i=0):
        assert isinstance(value, Parameter)
        self.__w[i] = value

    def _build_once(self, input):
        i = 0
        for inp, param in self._helper.iter_inputs_and_params(input,
                                                              self._param_attr):
            input_shape = inp.shape

            param_shape = [
                reduce(lambda a, b: a * b, input_shape[self._num_flatten_dims:],
                       1)
            ] + [self._size]
            self.__w.append(
                self.add_parameter(
                    '_w%d' % i,
                    self.create_parameter(
                        attr=param,
                        shape=param_shape,
                        dtype=self._dtype,
                        is_bias=False)))
            i += 1
        size = list([self._size])
        self._b = self.create_parameter(
            attr=self._bias_attr, shape=size, dtype=self._dtype, is_bias=True)

    def forward(self, input):
        mul_results = list()
        i = 0
        for inp, param in self._helper.iter_inputs_and_params(input,
                                                              self._param_attr):
            tmp = self._helper.create_variable_for_type_inference(self._dtype)
            self._helper.append_op(
                type="mul",
                inputs={"X": inp,
                        "Y": self.__w[i]},
                outputs={"Out": tmp},
                attrs={
                    "x_num_col_dims": self._num_flatten_dims,
                    "y_num_col_dims": 1
                })
            i += 1
            mul_results.append(tmp)

        if len(mul_results) == 1:
            pre_bias = mul_results[0]
        else:
            pre_bias = self._helper.create_variable_for_type_inference(
                self._dtype)
            self._helper.append_op(
                type="sum",
                inputs={"X": mul_results},
                outputs={"Out": pre_bias},
                attrs={"use_mkldnn": False})

        if self._b:
            pre_activation = self._helper.create_variable_for_type_inference(
                dtype=self._dtype)
            self._helper.append_op(
                type='elementwise_add',
                inputs={'X': [pre_bias],
                        'Y': [self._b]},
                outputs={'Out': [pre_activation]},
                attrs={'axis': self._num_flatten_dims})
        else:
            pre_activation = pre_bias
        # Currently, we don't support inplace in imperative mode
        return self._helper.append_activation(pre_activation, act=self._act)


class BatchNorm(layers.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 act=None,
                 is_test=False,
                 momentum=0.9,
                 epsilon=1e-05,
                 param_attr=None,
                 bias_attr=None,
                 dtype=core.VarDesc.VarType.FP32,
                 data_layout='NCHW',
                 in_place=False,
                 moving_mean_name=None,
                 moving_variance_name=None,
                 do_model_average_for_mean_and_var=False,
                 fuse_with_relu=False,
                 use_global_stats=False):
        super(BatchNorm, self).__init__(name_scope)
        self._param_attr = param_attr
        self._param_attr = bias_attr
        self._act = act

        assert bias_attr is not False, "bias_attr should not be False in batch_norm."

        if dtype == core.VarDesc.VarType.FP16:
            self._dtype = core.VarDesc.VarType.FP32
        else:
            self._dtype = dtype

        param_shape = [num_channels]

        # create parameter
        self._scale = self.create_parameter(
            attr=self._param_attr,
            shape=param_shape,
            dtype=self._dtype,
            default_initializer=Constant(1.0))
        if use_global_stats and self._param_attr.learning_rate == 0.:
            self._scale._stop_gradient = True

        self._bias = self.create_parameter(
            attr=self._param_attr,
            shape=param_shape,
            dtype=self._dtype,
            is_bias=True)
        if use_global_stats and self._param_attr.learning_rate == 0.:
            self._bias._stop_gradient = True

        self._mean = self.create_parameter(
            attr=ParamAttr(
                name=moving_mean_name,
                initializer=Constant(0.0),
                trainable=False,
                do_model_average=do_model_average_for_mean_and_var),
            shape=param_shape,
            dtype=self._dtype)
        self._mean._stop_gradient = True

        self._variance = self.create_parameter(
            attr=ParamAttr(
                name=moving_variance_name,
                initializer=Constant(1.0),
                trainable=False,
                do_model_average=do_model_average_for_mean_and_var),
            shape=param_shape,
            dtype=self._dtype)
        self._variance._stop_gradient = True

        self._in_place = in_place
        self._momentum = momentum
        self._epsilon = epsilon
        self._is_test = is_test
        self._fuse_with_relu = fuse_with_relu
        self._use_global_stats = use_global_stats

    def _build_once(self, input):
        pass

    def forward(self, input):
        # create output
        # mean and mean_out share the same memory
        mean_out = self._mean
        # variance and variance out share the same memory
        variance_out = self._variance

        saved_mean = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True)
        saved_variance = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True)
        batch_norm_out = input if self._in_place else self._helper.create_variable_for_type_inference(
            self._dtype)

        self._helper.append_op(
            type="batch_norm",
            inputs={
                "X": input,
                "Scale": self._scale,
                "Bias": self._bias,
                "Mean": self._mean,
                "Variance": self._variance
            },
            outputs={
                "Y": batch_norm_out,
                "MeanOut": mean_out,
                "VarianceOut": variance_out,
                "SavedMean": saved_mean,
                "SavedVariance": saved_variance
            },
            attrs={
                "momentum": self._momentum,
                "epsilon": self._epsilon,
                "is_test": self._is_test,
                "use_mkldnn": False,
                "fuse_with_relu": self._fuse_with_relu,
                "use_global_stats": self._use_global_stats
            })

        # Currently, we don't support inplace in imperative mode
        return self._helper.append_activation(batch_norm_out, self._act)


class Embedding(layers.Layer):
    """
    **Embedding Layer**

    This layer is used to lookup embeddings of IDs, provided by :attr:`input`, in
    a lookup table. The result of this lookup is the embedding of each ID in the
    :attr:`input`.

    All the input variables are passed in as local variables to the LayerHelper
    constructor.

    Args:
        name_scope: See base class.
        size(tuple|list): The shape of the look up table parameter. It should
            have two elements which indicate the size of the dictionary of
            embeddings and the size of each embedding vector respectively.
        is_sparse(bool): The flag indicating whether to use sparse update.
        is_distributed(bool): Whether to run lookup table from remote parameter server.
        padding_idx(int|long|None): If :attr:`None`, it makes no effect to lookup.
            Otherwise the given :attr:`padding_idx` indicates padding the output
            with zeros whenever lookup encounters it in :attr:`input`. If
            :math:`padding_idx < 0`, the :attr:`padding_idx` to use in lookup is
            :math:`size[0] + dim`.
        param_attr(ParamAttr): Parameters for this layer
        dtype(np.dtype|core.VarDesc.VarType|str): The type of data : float32, float_16, int etc

    Returns:
        Variable: The tensor variable storing the embeddings of the \
                  supplied inputs.

    Examples:
        .. code-block:: python

          dict_size = len(dataset.ids)
          input = fluid.layers.data(name='ids', shape=[32, 32], dtype='float32')
          embedding = fluid.imperative.Embedding(size=[dict_size, 16])
          fc = embedding(input)
    """

    def __init__(self,
                 name_scope,
                 size,
                 is_sparse=False,
                 is_distributed=False,
                 padding_idx=None,
                 param_attr=None,
                 dtype='float32'):

        super(Embedding, self).__init__(name_scope)
        self._size = size
        self._is_sparse = is_sparse
        self._is_distributed = is_distributed
        self._padding_idx = -1 if padding_idx is None else padding_idx if padding_idx >= 0 else (
            size[0] + padding_idx)

        self._param_attr = param_attr
        self._dtype = dtype
        self._remote_prefetch = self._is_sparse and (not self._is_distributed)
        if self._remote_prefetch:
            assert self._is_sparse is True and self._is_distributed is False

        self._w = self.create_parameter(
            attr=self._param_attr,
            shape=self._size,
            dtype=self._dtype,
            is_bias=False)

    def forward(self, input):
        out = self._helper.create_variable_for_type_inference(self._dtype)
        self._helper.append_op(
            type='lookup_table',
            inputs={'Ids': input,
                    'W': self._w},
            outputs={'Out': out},
            attrs={
                'is_sparse': self._is_sparse,
                'is_distributed': self._is_distributed,
                'remote_prefetch': self._remote_prefetch,
                'padding_idx': self._padding_idx
            })

        return out


class LayerNorm(layers.Layer):
    def __init__(self,
                 name_scope,
                 scale=True,
                 shift=True,
                 begin_norm_axis=1,
                 epsilon=1e-05,
                 param_attr=None,
                 bias_attr=None,
                 act=None):
        """
        ${comment}

        The formula is as follows:

        ..  math::

            \\mu & = \\frac{1}{H}\\sum_{i=1}^{H} a_i

            \\sigma & = \\sqrt{\\frac{1}{H}\sum_{i=1}^{H}(a_i - \\mu)^2}

            h & = f(\\frac{g}{\\sigma}(a - \\mu) + b)

        * :math:`a`: the vector representation of the summed inputs to the neurons
        in that layer.

        * :math:`H`: the number of hidden units in a layers

        * :math:`g`: the trainable scale parameter.

        * :math:`b`: the trainable bias parameter.

        Args:
            input(Variable): The input tensor variable.
            scale(bool): Whether to learn the adaptive gain :math:`g` after
                normalization. Default True.
            shift(bool): Whether to learn the adaptive bias :math:`b` after
                normalization. Default True.
            begin_norm_axis(int): The normalization will be performed along
                dimensions from :attr:`begin_norm_axis` to :attr:`rank(input)`.
                Default 1.
            epsilon(float): The small value added to the variance to prevent
                division by zero. Default 1e-05.
            param_attr(ParamAttr|None): The parameter attribute for the learnable
                gain :math:`g`. If :attr:`scale` is False, :attr:`param_attr` is
                omitted. If :attr:`scale` is True and :attr:`param_attr` is None,
                a default :code:`ParamAttr` would be added as scale. The
                :attr:`param_attr` is initialized as 1 if it is added. Default None.
            bias_attr(ParamAttr|None): The parameter attribute for the learnable
                bias :math:`b`. If :attr:`shift` is False, :attr:`bias_attr` is
                omitted. If :attr:`shift` is True and :attr:`param_attr` is None,
                a default :code:`ParamAttr` would be added as bias. The
                :attr:`bias_attr` is initialized as 0 if it is added. Default None.
            act(str): Activation to be applied to the output of layer normalizaiton.
                      Default None.
        Returns:
            ${y_comment}

        Examples:

            >>> data = fluid.layers.data(name='data', shape=[3, 32, 32],
            >>>                          dtype='float32')
            >>> x = fluid.layers.layer_norm(input=data, begin_norm_axis=1)
        """

        super(LayerNorm, self).__init__(name_scope)
        self._scale = scale
        self._shift = shift
        self._begin_norm_axis = begin_norm_axis
        self._epsilon = epsilon
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._act = act

    def _build_once(self, input):
        self._dtype = self._helper.input_dtype(input)
        input_shape = input.shape
        param_shape = [
            reduce(lambda x, y: x * y, input_shape[self._begin_norm_axis:])
        ]
        if self._scale:
            self._scale_w = self.create_parameter(
                attr=self._param_attr,
                shape=param_shape,
                dtype=self._dtype,
                default_initializer=Constant(1.0))
        if self._shift:
            assert self._bias_attr is not False
            self._bias_w = self.create_parameter(
                attr=self._bias_attr,
                shape=param_shape,
                dtype=self._dtype,
                is_bias=True)

    def forward(self, input):
        inputs = dict()
        inputs['X'] = input
        if self._scale:
            inputs['Scale'] = self._scale_w
        if self._shift:
            inputs['Bias'] = self._bias_w
        # create output
        mean_out = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True)
        variance_out = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True)
        layer_norm_out = self._helper.create_variable_for_type_inference(
            self._dtype)

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
                "begin_norm_axis": self._begin_norm_axis
            })

        return self._helper.append_activation(layer_norm_out)


class GRUUnit(layers.Layer):
    """
    **GRU unit layer**

    if origin_mode is True, then the equation of a gru step is from paper
    `Learning Phrase Representations using RNN Encoder-Decoder for Statistical
    Machine Translation <https://arxiv.org/pdf/1406.1078.pdf>`_

        .. math::
            u_t & = actGate(xu_{t} + W_u h_{t-1} + b_u)

            r_t & = actGate(xr_{t} + W_r h_{t-1} + b_r)

            m_t & = actNode(xm_t + W_c dot(r_t, h_{t-1}) + b_m)

            h_t & = dot(u_t, h_{t-1}) + dot((1-u_t), m_t)

    if origin_mode is False, then the equation of a gru step is from paper
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

    Args:
        input (Variable): The fc transformed input value of current step.
        name_scope (str): See base class.
        hidden (Variable): The hidden value of gru unit from previous step.
        size (integer): The input dimension value.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
            hidden-hidden weight matrix. Note:

            - The shape of the weight matrix is :math:`(T \\times 3D)`, where
              :math:`D` is the hidden size.
            - All elements in the weight matrix can be divided into two parts.
              The first part are weights of the update gate and reset gate with
              shape :math:`(D \\times 2D)`, and the second part are weights for
              candidate hidden state with shape :math:`(D \\times D)`.

            If it is set to None or one attribute of ParamAttr, gru_unit will
            create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias
            of GRU.Note that the bias with :math:`(1 \\times 3D)` concatenates
            the bias in the update gate, reset gate and candidate calculations.
            If it is set to False, no bias will be applied to the update gate,
            reset gate and candidate calculations. If it is set to None or one
            attribute of ParamAttr, gru_unit will create ParamAttr as
            bias_attr. If the Initializer of the bias_attr is not set, the bias
            is initialized zero. Default: None.
        activation (string): The activation type for cell (actNode).
                             Default: 'tanh'
        gate_activation (string): The activation type for gates (actGate).
                                  Default: 'sigmoid'

    Returns:
        tuple: The hidden value, reset-hidden value and gate values.
    """

    def __init__(self,
                 name_scope,
                 size,
                 param_attr=None,
                 bias_attr=None,
                 activation='tanh',
                 gate_activation='sigmoid',
                 origin_mode=False,
                 dtype='float32'):
        super(GRUUnit, self).__init__(name_scope)

        activation_dict = dict(
            identity=0,
            sigmoid=1,
            tanh=2,
            relu=3, )
        activation = activation_dict[activation]
        gate_activation = activation_dict[gate_activation]

        self._dtype = dtype
        size = size // 3
        # create weight
        self._weight = self.create_parameter(
            attr=param_attr, shape=[size, 3 * size], dtype=dtype)

        # create bias
        bias_size = [1, 3 * size]
        self._bias = self.create_parameter(
            attr=bias_attr, shape=bias_size, dtype=dtype, is_bias=True)

    def forward(self, input, hidden):
        inputs = {'Input': input, 'HiddenPrev': hidden, 'Weight': self._weight}
        if self._bias:
            inputs['Bias'] = self._bias

        gate = self._helper.create_variable_for_type_inference(self._dtype)
        reset_hidden_pre = self._helper.create_variable_for_type_inference(
            self._dtype)
        updated_hidden = self._helper.create_variable_for_type_inference(
            self._dtype)
        self._helper.append_op(
            type='gru_unit',
            inputs=inputs,
            outputs={
                'Gate': gate,
                'ResetHiddenPrev': reset_hidden_pre,
                'Hidden': updated_hidden,
            },
            attrs={
                'activation': 2,  # tanh
                'gate_activation': 1,  # sigmoid
            })

        return updated_hidden, reset_hidden_pre, gate
