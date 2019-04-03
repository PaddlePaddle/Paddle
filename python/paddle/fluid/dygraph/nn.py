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
from ..framework import Variable, _in_dygraph_mode, OpProtoHolder, Parameter
from ..param_attr import ParamAttr
from ..initializer import Normal, Constant, NumpyArrayInitializer
import numpy as np

__all__ = [
    'Conv2D', 'Conv3D', 'Pool2D', 'FC', 'BatchNorm', 'Embedding', 'GRUUnit',
    'LayerNorm', 'NCE', 'PRelu', 'BilinearTensorProduct', 'Conv2DTranspose',
    'Conv3DTranspose', 'SequenceConv', 'RowConv', 'GroupNorm', 'SpectralNorm',
    'TreeConv'
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

        # Currently, we don't support inplace in dygraph mode
        return self._helper.append_activation(pre_act, act=self._act)


class Conv3D(layers.Layer):
    """
    **Convlution3D Layer**

    The convolution3D layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input(Input) and
    Output(Output) are in NCDHW format. Where N is batch size C is the number of
    channels, D is the depth of the feature, H is the height of the feature,
    and W is the width of the feature. Convlution3D is similar with Convlution2D
    but adds one dimension(depth). If bias attribution and activation type are
    provided, bias is added to the output of the convolution, and the
    corresponding activation function is applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \\ast X + b)

    In the above equation:

    * :math:`X`: Input value, a tensor with NCDHW format.
    * :math:`W`: Filter value, a tensor with MCDHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{out}, C_{in}, D_f, H_f, W_f)`

        - Output:
          Output shape: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`

        Where

        .. math::

            D_{out}&= \\frac{(D_{in} + 2 * paddings[0] - (dilations[0] * (D_f - 1) + 1))}{strides[0]} + 1 \\\\
            H_{out}&= \\frac{(H_{in} + 2 * paddings[1] - (dilations[1] * (H_f - 1) + 1))}{strides[1]} + 1 \\\\
            W_{out}&= \\frac{(W_{in} + 2 * paddings[2] - (dilations[2] * (W_f - 1) + 1))}{strides[2]} + 1

    Args:
        input (Variable): The input image with [N, C, D, H, W] format.
            num_filters(int): The number of filter. It is as same as the output
            image channel.
        filter_size (int|tuple|None): The filter size. If filter_size is a tuple,
            it must contain three integers, (filter_size_D, filter_size_H, filter_size_W).
            Otherwise, the filter will be a square.
        stride (int|tuple): The stride size. If stride is a tuple, it must
            contain three integers, (stride_D, stride_H, stride_W). Otherwise, the
            stride_D = stride_H = stride_W = stride. Default: stride = 1.
        padding (int|tuple): The padding size. If padding is a tuple, it must
            contain three integers, (padding_D, padding_H, padding_W). Otherwise, the
            padding_D = padding_H = padding_W = padding. Default: padding = 0.
        dilation (int|tuple): The dilation size. If dilation is a tuple, it must
            contain three integers, (dilation_D, dilation_H, dilation_W). Otherwise, the
            dilation_D = dilation_H = dilation_W = dilation. Default: dilation = 1.
        groups (int): The groups number of the Conv3d Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1
        param_attr (ParamAttr|None): The parameter attribute for learnable parameters/weights
            of conv3d. If it is set to None or one attribute of ParamAttr, conv3d
            will create ParamAttr as param_attr. If it is set to None, the parameter
            is initialized with :math:`Normal(0.0, std)`, and the :math:`std` is
            :math:`(\\frac{2.0 }{filter\_elem\_num})^{0.5}`. Default: None.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias of conv3d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv3d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn (bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True
        act (str): Activation type, if it is set to None, activation is not appended.
            Default: None.
        name (str|None): A name for this layer(optional). If set None, the layer
            will be named automatically. Default: None.

    Returns:
        Variable: The tensor variable storing the convolution and \
                  non-linearity activation result.

    Raises:
        ValueError: If the shapes of input, filter_size, stride, padding and
                    groups mismatch.

    Examples:
        .. code-block:: python

          data = fluid.layers.data(name='data', shape=[3, 12, 32, 32], dtype='float32')
          conv3d = fluid.layers.conv3d(input=data, num_filters=2, filter_size=3, act="relu")
    """

    def __init__(self,
                 name_scope,
                 num_filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=None,
                 param_attr=None,
                 bias_attr=None,
                 use_cudnn=True,
                 act=None):
        assert param_attr is not False, "param_attr should not be False here."
        super(Conv3D, self).__init__(name_scope)
        self._groups = groups
        self._stride = utils.convert_to_list(stride, 3, 'stride')
        self._padding = utils.convert_to_list(padding, 3, 'padding')
        self._dilation = utils.convert_to_list(dilation, 3, 'dilation')
        self._act = act
        if not isinstance(use_cudnn, bool):
            raise ValueError("use_cudnn should be True or False")
        self._use_cudnn = use_cudnn
        self._filter_size = filter_size
        self._num_filters = num_filters
        self._param_attr = param_attr
        self._bias_attr = bias_attr

    def _build_once(self, input):
        num_channels = input.shape[1]
        self._dtype = self._helper.input_dtype(input)

        if self._groups is None:
            num_filter_channels = num_channels
        else:
            if num_channels % self._groups != 0:
                raise ValueError("num_channels must be divisible by groups.")
            num_filter_channels = num_channels // self._groups

        filter_size = utils.convert_to_list(self._filter_size, 3, 'filter_size')

        filter_shape = [self._num_filters, num_filter_channels] + filter_size

        def _get_default_param_initializer():
            filter_elem_num = filter_size[0] * filter_size[1] * filter_size[
                2] * num_channels
            std = (2.0 / filter_elem_num)**0.5
            return Normal(0.0, std, 0)

        self._filter_param = self.create_parameter(
            attr=self._param_attr,
            shape=filter_shape,
            dtype=self._dtype,
            default_initializer=_get_default_param_initializer())

        self._bias_param = self.create_parameter(
            attr=self._bias_attr,
            shape=[self._num_filters],
            dtype=self._dtype,
            is_bias=True)

    def forward(self, input):
        pre_bias = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)

        self._helper.append_op(
            type='conv3d',
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
                'use_mkldnn': False
            })

        pre_act = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)

        self._helper.append_op(
            type='elementwise_add',
            inputs={'X': [pre_bias],
                    'Y': [self._bias_param]},
            outputs={'Out': [pre_act]},
            attrs={'axis': 1})

        return self._helper.append_activation(pre_act, act=self._act)


class Conv3DTranspose(layers.Layer):
    def __init__(self,
                 name_scope,
                 num_filters,
                 output_size=None,
                 filter_size=None,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=None,
                 param_attr=None,
                 bias_attr=None,
                 use_cudnn=True,
                 act=None,
                 name=None):
        super(Conv3DTranspose, self).__init__(name_scope)
        if not isinstance(use_cudnn, bool):
            raise ValueError("use_cudnn should be True or False")
        assert param_attr is not False, "param_attr should not be False in conv3d_transpose."
        self._padding = utils.convert_to_list(padding, 3, 'padding')
        self._stride = utils.convert_to_list(stride, 3, 'stride')
        self._dilation = utils.convert_to_list(dilation, 3, 'dilation')
        self._param_attr = param_attr
        self._filter_size = filter_size
        self._output_size = output_size
        self._groups = 1 if groups is None else groups
        self._num_filters = num_filters
        self._use_cudnn = use_cudnn
        self._bias_attr = bias_attr
        self._act = act

    def _build_once(self, input):
        self._dtype = self._helper.input_dtype(input)
        self._input_channel = input.shape[1]

        if self._filter_size is None:
            if self._output_size is None:
                raise ValueError(
                    "output_size must be set when filter_size is None")
            if isinstance(self._output_size, int):
                self._output_size = [self._output_size, self._output_size]

            d_in = input.shape[2]
            h_in = input.shape[3]
            w_in = input.shape[4]

            filter_size_d = (self._output_size[0] -
                             (d_in - 1) * self._stride[0] + 2 * self._padding[0]
                             - 1) // self._dilation[0] + 1
            filter_size_h = (self._output_size[1] -
                             (h_in - 1) * self._stride[1] + 2 * self._padding[1]
                             - 1) // self._dilation[1] + 1
            filter_size_w = (self._output_size[2] -
                             (w_in - 1) * self._stride[2] + 2 * self._padding[2]
                             - 1) // self._dilation[2] + 1
            self._filter_size = [filter_size_d, filter_size_h, filter_size_w]
        else:
            self._filter_size = utils.convert_to_list(
                self._filter_size, 3, 'conv3d_transpose.filter_size')

        filter_shape = [
            self._input_channel, self._num_filters // self._groups
        ] + self._filter_size
        self._img_filter = self.create_parameter(
            dtype=self._dtype, shape=filter_shape, attr=self._param_attr)
        if self._bias_attr:
            self._bias_param = self.create_parameter(
                attr=self._bias_attr,
                shape=[self._num_filters],
                dtype=self._dtype,
                is_bias=True)

    def forward(self, input):
        pre_bias = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)
        self._helper.append_op(
            type="conv3d_transpose",
            inputs={'Input': [input],
                    'Filter': [self._img_filter]},
            outputs={'Output': pre_bias},
            attrs={
                'strides': self._stride,
                'paddings': self._padding,
                'dilations': self._dilation,
                'groups': self._groups if self._groups else 1,
                'use_cudnn': self._use_cudnn
            })

        if self._bias_attr:
            pre_act = self._helper.create_variable_for_type_inference(
                dtype=self._dtype)
            self._helper.append_op(
                type='elementwise_add',
                inputs={'X': [pre_bias],
                        'Y': [self._bias_param]},
                outputs={'Out': [pre_act]},
                attrs={'axis': 1})
        else:
            pre_act = pre_bias

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

    def _build_once(self, input):
        input_shape = input.shape
        param_shape = [
            reduce(lambda a, b: a * b, input_shape[self._num_flatten_dims:], 1)
        ] + [self._size]
        self._w = self.create_parameter(
            attr=self._param_attr,
            shape=param_shape,
            dtype=self._dtype,
            is_bias=False)

        if self._bias_attr:
            size = list([self._size])
            self._b = self.create_parameter(
                attr=self._bias_attr,
                shape=size,
                dtype=self._dtype,
                is_bias=True)
        else:
            self._b = None

    def forward(self, input):
        tmp = self._helper.create_variable_for_type_inference(self._dtype)
        self._helper.append_op(
            type="mul",
            inputs={"X": input,
                    "Y": self._w},
            outputs={"Out": tmp},
            attrs={
                "x_num_col_dims": self._num_flatten_dims,
                "y_num_col_dims": 1
            })

        pre_bias = self._helper.create_variable_for_type_inference(self._dtype)
        self._helper.append_op(
            type="sum",
            inputs={"X": [tmp]},
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
        # Currently, we don't support inplace in dygraph mode
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

        # Currently, we don't support inplace in dygraph mode
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
          embedding = fluid.dygraph.Embedding(size=[dict_size, 16])
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


class NCE(layers.Layer):
    """
    ${comment}

    Args:
        input (Variable): input variable.
        label (Variable): label.
        num_total_classes (int):${num_total_classes_comment}
        sample_weight (Variable|None): A Variable of shape [batch_size, 1]
            storing a weight for each sample. The default weight for each
            sample is 1.0.
        param_attr (ParamAttr|None): The parameter attribute for learnable parameters/weights
             of nce. If it is set to None or one attribute of ParamAttr, nce
             will create ParamAttr as param_attr. If the Initializer of the param_attr
             is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias of nce.
             If it is set to False, no bias will be added to the output units.
             If it is set to None or one attribute of ParamAttr, nce
             will create ParamAttr as bias_attr. If the Initializer of the bias_attr
             is not set, the bias is initialized zero. Default: None.
        num_neg_samples (int): ${num_neg_samples_comment}
        name (str|None): A name for this layer(optional). If set None, the layer
             will be named automatically. Default: None.
        sampler (str): The sampler used to sample class from negtive classes.
                       It can be 'uniform', 'log_uniform' or 'custom_dist'.
                       default: 'uniform'.
        custom_dist (float[]): A float[] with size=num_total_classes.
                       It is used when sampler is set to 'custom_dist'.
                       custom_dist[i] is the probsbility of i-th class to be sampled.
                       default: None.
        seed (int): The seed used in sampler. default: 0.
        is_sparse(bool): The flag indicating whether to use sparse update, the weight@GRAD and bias@GRAD will be changed to SelectedRows.

    Returns:
        Variable: The output nce loss.

    Examples:
        .. code-block:: python

            window_size = 5
            words = []
            for i in xrange(window_size):
                words.append(layers.data(
                    name='word_{0}'.format(i), shape=[1], dtype='int64'))

            dict_size = 10000
            label_word = int(window_size / 2) + 1

            embs = []
            for i in xrange(window_size):
                if i == label_word:
                    continue

                emb = layers.embedding(input=words[i], size=[dict_size, 32],
                                       param_attr='emb.w', is_sparse=True)
                embs.append(emb)

            embs = layers.concat(input=embs, axis=1)
            loss = layers.nce(input=embs, label=words[label_word],
                          num_total_classes=dict_size, param_attr='nce.w',
                          bias_attr='nce.b')

            #or use custom distribution
            dist = fluid.layers.assign(input=np.array([0.05,0.5,0.1,0.3,0.05]).astype("float32"))
            loss = layers.nce(input=embs, label=words[label_word],
                          num_total_classes=5, param_attr='nce.w',
                          bias_attr='nce.b',
                          num_neg_samples=3,
                          sampler="custom_dist",
                          custom_dist=dist)

    """

    def __init__(self,
                 name_scope,
                 num_total_classes,
                 param_attr=None,
                 bias_attr=None,
                 num_neg_samples=None,
                 sampler="uniform",
                 custom_dist=None,
                 seed=0,
                 is_sparse=False):
        super(NCE, self).__init__(name_scope)
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._num_total_classes = num_total_classes

        self._inputs = dict()

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
                    default_initializer=NumpyArrayInitializer(numpy_array))
                ret.stop_gradient = True
                return ret

            self._inputs['CustomDistProbs'] = _init_by_numpy_array(
                np.array(custom_dist).astype('float32'))
            self._inputs['CustomDistAlias'] = _init_by_numpy_array(
                np.array(alias_).astype('int32'))
            self._inputs['CustomDistAliasProbs'] = _init_by_numpy_array(
                np.array(alias_probs_).astype('float32'))
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
            'remote_prefetch': remote_prefetch
        }

    def _build_once(self, input, label, sample_weight=None):
        assert isinstance(input, Variable)
        assert isinstance(label, Variable)

        dim = input.shape[1]
        num_true_class = label.shape[1]
        self._w = self.create_parameter(
            attr=self._param_attr,
            shape=[self._num_total_classes, dim],
            is_bias=False,
            dtype=input.dtype)
        if self._bias_attr:
            self._b = self.create_parameter(
                attr=self._bias_attr,
                shape=[self._num_total_classes, 1],
                is_bias=True,
                dtype=input.dtype)
            self._inputs['Bias'] = self._b
        self._inputs['Weight'] = self._w

    def forward(self, input, label, sample_weight=None):
        assert isinstance(input, Variable)
        assert isinstance(label, Variable)

        self._inputs['Input'] = input
        self._inputs['Label'] = label
        self._inputs['SampleWeight'] = sample_weight if sample_weight is not None else []

        cost = self._helper.create_variable_for_type_inference(
            dtype=input.dtype)
        sample_logits = self._helper.create_variable_for_type_inference(
            dtype=input.dtype)
        sample_labels = self._helper.create_variable_for_type_inference(
            dtype=label.dtype)

        self._helper.append_op(
            type='nce',
            inputs=self._inputs,
            outputs={
                'Cost': cost,
                'SampleLogits': sample_logits,
                'SampleLabels': sample_labels
            },
            attrs=self._attrs)
        return cost / (self._num_neg_samples + 1)


class PRelu(layers.Layer):
    """
    Equation:

    .. math::
        y = \max(0, x) + \\alpha * \min(0, x)

    Args:
        x (Variable): The input tensor.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
          weight (alpha).
        mode (string): The mode for weight sharing. It supports all, channel
          and element. all: all elements share same weight
          channel:elements in a channel share same weight
          element:each element has a weight
        name(str|None): A name for this layer(optional). If set None, the layer
          will be named automatically.

    Returns:
        Variable: The output tensor with the same shape as input.

    Examples:

        .. code-block:: python

            x = fluid.layers.data(name="x", shape=[10,10], dtype="float32")
            mode = 'channel'
            output = fluid.layers.prelu(x,mode)
    """

    def __init__(self, name_scope, mode, param_attr=None):

        super(PRelu, self).__init__(name_scope)
        self._mode = mode
        self._param_attr = param_attr
        if self._mode not in ['all', 'channel', 'element']:
            raise ValueError('mode should be one of all, channel, element.')
        self._alpha_shape = [1]

    def _build_once(self, input):
        if self._mode == 'channel':
            self._alpha_shape = [1, input.shape[1], 1, 1]
        elif self._mode == 'element':
            self._alpha_shape = input.shape
        self._dtype = self._helper.input_dtype(input)
        self._alpha = self.create_parameter(
            attr=self._param_attr,
            shape=self._alpha_shape,
            dtype='float32',
            is_bias=False,
            default_initializer=Constant(1.0))

    def forward(self, input):

        out = self._helper.create_variable_for_type_inference(self._dtype)
        self._helper.append_op(
            type="prelu",
            inputs={"X": input,
                    'Alpha': self._alpha},
            attrs={"mode": self._mode},
            outputs={"Out": out})
        return out


class BilinearTensorProduct(layers.Layer):
    """
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
     - :math:`y^\mathrm{T}`: the transpose of :math:`y_{2}`.

    Args:
       x (Variable): 2-D input tensor with shape [batch_size, M]
       y (Variable): 2-D input tensor with shape [batch_size, N]
       size (int): The dimension of this layer.
       act (str, default None): Activation to be applied to the output of this layer.
       name (str, default None): The name of this layer.
       param_attr (ParamAttr, default None): The parameter attribute for the learnable w.
           parameters/weights of this layer.
       bias_attr (ParamAttr, default None): The parameter attribute for the bias
           of this layer. If it is set to False, no bias will be added to the output units.
           If it is set to None, the bias is initialized zero. Default: None.

    Returns:
       Variable: A 2-D Tensor of shape [batch_size, size].

    Examples:
       .. code-block:: python

         tensor = bilinear_tensor_product(x=layer1, y=layer2, size=1000)
    """

    def __init__(self,
                 name_scope,
                 size,
                 name=None,
                 act=None,
                 param_attr=None,
                 bias_attr=None):
        super(BilinearTensorProduct, self).__init__(name_scope)
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._act = act
        self._size = size
        self._name = name
        self._inputs = dict()

    def _build_once(self, x, y):
        self._dtype = self._helper.input_dtype(x)

        param_shape = [self._size, x.shape[1], y.shape[1]]

        self._w = self.create_parameter(
            attr=self._param_attr,
            shape=param_shape,
            dtype=self._dtype,
            is_bias=False)

        if self._bias_attr:
            bias_size = [1, self._size]
            bias = self.create_parameter(
                attr=self._bias_attr,
                shape=bias_size,
                dtype=self._dtype,
                is_bias=True)
            self._inputs["Bias"] = bias

    def forward(self, x, y):
        self._inputs = {"X": x, "Y": y, "Weight": self._w}
        if self._name is not None:
            out = self._helper.create_variable(
                name=".".join([self.full_name(), self._name]),
                dtype=self._dtype,
                persistable=False)
        else:
            out = self._helper.create_variable(
                dtype=self._dtype, persistable=False)
        self._helper.append_op(
            type="bilinear_tensor_product",
            inputs=self._inputs,
            outputs={"Out": out})

        # add activation
        return self._helper.append_activation(out)


class Conv2DTranspose(layers.Layer):
    """
    **Convlution2D transpose layer**

    The convolution2D transpose layer calculates the output based on the input,
    filter, and dilations, strides, paddings. Input(Input) and output(Output)
    are in NCHW format. Where N is batch size, C is the number of channels,
    H is the height of the feature, and W is the width of the feature.
    Parameters(dilations, strides, paddings) are two elements. These two elements
    represent height and width, respectively. The details of convolution transpose
    layer, please refer to the following explanation and references
    `therein <http://www.matthewzeiler.com/wp-content/uploads/2017/07/cvpr2010.pdf>`_.
    If bias attribution and activation type are provided, bias is added to
    the output of the convolution, and the corresponding activation function
    is applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \\ast X + b)

    Where:

    * :math:`X`: Input value, a tensor with NCHW format.
    * :math:`W`: Filter value, a tensor with MCHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D tensor with shape [M, 1].
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

    Args:
        input(Variable): The input image with [N, C, H, W] format.
        num_filters(int): The number of the filter. It is as same as the output
            image channel.
        output_size(int|tuple|None): The output image size. If output size is a
            tuple, it must contain two integers, (image_H, image_W). None if use
            filter_size, padding, and stride to calculate output_size.
            if output_size and filter_size are specified at the same time, They
            should follow the formula above.
        filter_size(int|tuple|None): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_H, filter_size_W).
            Otherwise, the filter will be a square. None if use output size to
            calculate filter_size.
        padding(int|tuple): The padding size. If padding is a tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: padding = 0.
        stride(int|tuple): The stride size. If stride is a tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: stride = 1.
        dilation(int|tuple): The dilation size. If dilation is a tuple, it must
            contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. Default: dilation = 1.
        groups(int): The groups number of the Conv2d transpose layer. Inspired by
            grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
            when group=2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            Default: groups = 1.
        param_attr (ParamAttr|None): The parameter attribute for learnable parameters/weights
            of conv2d_transpose. If it is set to None or one attribute of ParamAttr, conv2d_transpose
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias of conv2d_transpose.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d_transpose
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn(bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True.
        act (str): Activation type, if it is set to None, activation is not appended.
            Default: None.
        name(str|None): A name for this layer(optional). If set None, the layer
            will be named automatically. Default: True.

    Returns:
        Variable: The tensor variable storing the convolution transpose result.

    Raises:
        ValueError: If the shapes of input, filter_size, stride, padding and
                    groups mismatch.

    Examples:
       .. code-block:: python

          data = fluid.layers.data(name='data', shape=[3, 32, 32], dtype='float32')
          conv2d_transpose = fluid.layers.conv2d_transpose(input=data, num_filters=2, filter_size=3)
    """

    def __init__(self,
                 name_scope,
                 num_filters,
                 output_size=None,
                 filter_size=None,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=None,
                 param_attr=None,
                 bias_attr=None,
                 use_cudnn=True,
                 act=None):
        super(Conv2DTranspose, self).__init__(name_scope)
        assert param_attr is not False, "param_attr should not be False in conv2d_transpose."
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._groups = groups
        self._num_filters = num_filters
        self._use_cudnn = use_cudnn
        self._padding = padding
        self._stride = stride
        self._dilation = dilation
        self._filter_size = filter_size
        self._output_size = output_size
        self._op_type = 'conv2d_transpose'

    def _build_once(self, input):
        input_channel = input.shape[1]
        if (input_channel == self._groups and
                self._num_filters == input_channel and not self._use_cudnn):
            self._op_type = 'depthwise_conv2d_transpose'

        if not isinstance(input, Variable):
            raise TypeError("Input of conv2d_transpose must be Variable")

        self._padding = utils.convert_to_list(self._padding, 2, 'padding')
        self._stride = utils.convert_to_list(self._stride, 2, 'stride')
        self._dilation = utils.convert_to_list(self._dilation, 2, 'dilation')

        if not isinstance(self._use_cudnn, bool):
            raise ValueError("use_cudnn should be True or False")

        if self._filter_size is None:
            if self._output_size is None:
                raise ValueError(
                    "output_size must be set when filter_size is None")
            if isinstance(self._output_size, int):
                self._output_size = [self._output_size, self._output_size]

            h_in = input.shape[2]
            w_in = input.shape[3]

            filter_size_h = (self._output_size[0] -
                             (h_in - 1) * self._stride[0] + 2 * self._padding[0]
                             - 1) // self._dilation[0] + 1
            filter_size_w = (self._output_size[1] -
                             (w_in - 1) * self._stride[1] + 2 * self._padding[1]
                             - 1) // self._dilation[1] + 1
            self._filter_size = [filter_size_h, filter_size_w]
        else:
            self._filter_size = utils.convert_to_list(
                self._output_size, 2, 'conv2d_transpose.filter_size')

        if self._output_size is None:
            self._output_size = []
        elif isinstance(self._output_size, list) or isinstance(
                self._output_size, int):
            self._output_size = utils.convert_to_list(self._output_size, 2,
                                                      'output_size')
        else:
            raise ValueError("output_size should be list or int")
        self._padding = utils.convert_to_list(self._padding, 2, 'padding')
        self._groups = 1 if self._groups is None else self._groups
        filter_shape = [input_channel, self._num_filters // self._groups
                        ] + self._filter_size

        self._img_filter = self.create_parameter(
            dtype=input.dtype, shape=filter_shape, attr=self._param_attr)

    def forward(self, input):
        pre_bias = self._helper.create_variable_for_type_inference(
            dtype=input.dtype)
        self._helper.append_op(
            type=self._op_type,
            inputs={'Input': [input],
                    'Filter': [self._img_filter]},
            outputs={'Output': pre_bias},
            attrs={
                'output_size': self._output_size,
                'strides': self._stride,
                'paddings': self._padding,
                'dilations': self._dilation,
                'groups': self._groups,
                'use_cudnn': self._use_cudnn
            })

        pre_act = self._helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)
        out = self._helper.append_activation(pre_act)
        return out


class SequenceConv(layers.Layer):
    """
    This function creates the op for sequence_conv, using the inputs and
    other convolutional configurations for the filters and stride as given
    in the input parameters to the function.

    Args:
        input (Variable): ${x_comment}
        num_filters (int): number of filters.
        filter_size (int): the filter size (H and W).
        filter_stride (int): stride of the filter.
        padding (bool): if True, add paddings.
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
        name (str|None): A name for this layer(optional). If set None, the layer
            will be named automatically. Default: None.

    Returns:
        Variable: output of sequence_conv
    """

    def __init__(self,
                 name_scope,
                 num_filters,
                 filter_size=3,
                 filter_stride=1,
                 padding=None,
                 bias_attr=None,
                 param_attr=None,
                 act=None):
        assert not _in_dygraph_mode(
        ), "SequenceConv is not supported by dynamic graph mode yet!"
        super(SequenceConv, self).__init__(name_scope)
        self._num_filters = num_filters
        self._filter_size = filter_size
        self._filter_stride = filter_stride
        self._padding = padding
        self._bias_attr = bias_attr
        self._param_attr = param_attr

    def _build_once(self, input):
        self._dtype = self._helper.input_dtype(input)
        filter_shape = [self._filter_size * input.shape[1], self._num_filters]
        self._filter_param = self.create_parameter(
            attr=self._param_attr, shape=filter_shape, dtype=self._dtype)

    def forward(self, input):
        pre_bias = self._helper.create_variable_for_type_inference(self._dtype)
        self._helper.append_op(
            type='sequence_conv',
            inputs={
                'X': [input],
                'Filter': [self._filter_param],
            },
            outputs={"Out": pre_bias},
            attrs={
                'contextStride': self._filter_stride,
                'contextStart': -int(self._filter_size // 2),
                'contextLength': self._filter_size
            })
        pre_act = self._helper.append_bias_op(pre_bias)
        return self._helper.append_activation(pre_act)


class RowConv(layers.Layer):
    def __init__(self,
                 name_scope,
                 future_context_size,
                 param_attr=None,
                 act=None):
        assert not _in_dygraph_mode(
        ), "RowConv is not supported by dynamic graph mode yet!"
        super(RowConv, self).__init__(name_scope)
        self._act = act
        self._param_attr = param_attr
        self._future_context_size = future_context_size

    def _build_once(self, input):
        self._dtype = self._helper.input_dtype(input)
        filter_shape = [self._future_context_size + 1, input.shape[1]]
        self._filter_param = self.create_parameter(
            attr=self._param_attr,
            shape=filter_shape,
            dtype=self._dtype,
            is_bias=False)

    def forward(self, input):
        out = self._helper.create_variable_for_type_inference(self._dtype)
        self._helper.append_op(
            type='row_conv',
            inputs={'X': [input],
                    'Filter': [self._filter_param]},
            outputs={'Out': [out]})
        return self._helper.append_activation(out, act=self._act)


class GroupNorm(layers.Layer):
    """
        **Group Normalization Layer**

        Refer to `Group Normalization <https://arxiv.org/abs/1803.08494>`_ .

        Args:
            name_scope (str): See base class.
            groups(int): The number of groups that divided from channels.
            epsilon(float): The small value added to the variance to prevent
                division by zero.
            param_attr(ParamAttr|None): The parameter attribute for the learnable
                scale :math:`g`. If it is set to False, no scale will be added to the output units.
                If it is set to None, the bias is initialized one. Default: None.
            bias_attr(ParamAttr|None): The parameter attribute for the learnable
                bias :math:`b`. If it is set to False, no bias will be added to the output units.
                If it is set to None, the bias is initialized zero. Default: None.
            act(str): Activation to be applied to the output of group normalizaiton.
            data_layout(string|NCHW): Only NCHW is supported.
            dtype(np.dtype|core.VarDesc.VarType|str): The type of data : float32, float_16, int etc

        Returns:
            Variable: A tensor variable which is the result after applying group normalization on the input.


    """

    def __init__(self,
                 name_scope,
                 groups,
                 epsilon=1e-05,
                 param_attr=None,
                 bias_attr=None,
                 act=None,
                 data_layout='NCHW'):
        super(GroupNorm, self).__init__(name_scope)
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._epsilon = epsilon
        self._groups = groups
        self._act = act
        if data_layout != 'NCHW':
            raise ValueError("unsupported data layout:" + data_layout)

    def _build_once(self, input):
        self._dtype = self._helper.input_dtype(input)
        param_shape = [input.shape[1]]
        if self._bias_attr:
            self._bias = self.create_parameter(
                attr=self._bias_attr,
                shape=param_shape,
                dtype=self._dtype,
                is_bias=True)

        if self._param_attr:
            self._scale = self.create_parameter(
                attr=self._param_attr,
                shape=param_shape,
                dtype=self._dtype,
                default_initializer=Constant(1.0))

    def forward(self, input):
        inputs = {'X': input}
        if self._bias:
            inputs['Bias'] = self._bias
        if self._scale:
            inputs['Scale'] = self._scale

        # create output
        mean_out = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True)
        variance_out = self._helper.create_variable_for_type_inference(
            dtype=self._dtype, stop_gradient=True)
        group_norm_out = self._helper.create_variable_for_type_inference(
            dtype=self._dtype)

        self._helper.append_op(
            type="group_norm",
            inputs=inputs,
            outputs={
                "Y": group_norm_out,
                "Mean": mean_out,
                "Variance": variance_out,
            },
            attrs={"epsilon": self._epsilon,
                   "groups": self._groups})

        return self._helper.append_activation(group_norm_out, self._act)


class SpectralNorm(layers.Layer):
    def __init__(self, name_scope, dim=0, power_iters=1, eps=1e-12, name=None):
        super(SpectralNorm, self).__init__(name_scope)
        self._power_iters = power_iters
        self._eps = eps
        self._dim = dim

    def _build_once(self, weight):
        self._dtype = self._helper.input_dtype(weight)
        input_shape = weight.shape
        h = input_shape[self._dim]
        w = np.prod(input_shape) // h

        self.u = self.create_parameter(
            attr=ParamAttr(),
            shape=[h],
            dtype=self._dtype,
            default_initializer=Normal(0., 1.))
        self.u.stop_gradient = True

        self.v = self.create_parameter(
            attr=ParamAttr(),
            shape=[w],
            dtype=self._dtype,
            default_initializer=Normal(0., 1.))
        self.v.stop_gradient = True

    def forward(self, weight):
        inputs = {'Weight': weight, 'U': self.u, 'V': self.v}
        out = self._helper.create_variable_for_type_inference(self._dtype)
        self._helper.append_op(
            type="spectral_norm",
            inputs=inputs,
            outputs={"Out": out, },
            attrs={
                "dim": self._dim,
                "power_iters": self._power_iters,
                "eps": self._eps,
            })

        return out


class TreeConv(layers.Layer):
    def __init__(self,
                 name_scope,
                 output_size,
                 num_filters=1,
                 max_depth=2,
                 act='tanh',
                 param_attr=None,
                 bias_attr=None,
                 name=None):
        super(TreeConv, self).__init__(name_scope)
        self._name = name
        self._output_size = output_size
        self._act = act
        self._max_depth = max_depth
        self._num_filters = num_filters
        self._bias_attr = bias_attr
        self._param_attr = param_attr

    def _build_once(self, nodes_vector, edge_set):
        assert isinstance(nodes_vector, Variable)
        assert isinstance(edge_set, Variable)
        self._dtype = self._helper.input_dtype(nodes_vector)

        feature_size = nodes_vector.shape[2]
        w_shape = [feature_size, 3, self._output_size, self._num_filters]
        if self._bias_attr:
            self._bias_param = self.create_parameter(
                attr=self._bias_attr,
                shape=[self._num_filters],
                dtype=self._dtype,
                is_bias=True)
        self.W = self.create_parameter(
            attr=self._param_attr,
            shape=w_shape,
            dtype=self._dtype,
            is_bias=False)

    def forward(self, nodes_vector, edge_set):
        if self._name:
            out = self.create_variable(
                name=self._name, dtype=self._dtype, persistable=False)
        else:
            out = self._helper.create_variable_for_type_inference(
                dtype=self._dtype)

        self._helper.append_op(
            type='tree_conv',
            inputs={
                'NodesVector': nodes_vector,
                'EdgeSet': edge_set,
                'Filter': self.W
            },
            outputs={'Out': out, },
            attrs={'max_depth': self._max_depth})
        if self._bias_attr:
            pre_activation = self._helper.create_variable_for_type_inference(
                dtype=self._dtype)
            self._helper.append_op(
                type='elementwise_add',
                inputs={'X': [out],
                        'Y': [self._bias_param]},
                outputs={'Out': [pre_activation]},
                attrs={'axis': 1})
        else:
            pre_activation = out
        return self._helper.append_activation(pre_activation, act=self._act)
