#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

from paddle.nn import Layer
from paddle.nn.functional.conv import _update_padding_nd
from paddle.nn.initializer import Normal
from paddle.utils import convert_to_list

from .. import functional as F

__all__ = []


class _Conv3D(Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        subm=False,
        key=None,
        padding_mode='zeros',
        weight_attr=None,
        bias_attr=None,
        data_format="NDHWC",
    ):
        super().__init__()
        assert (
            weight_attr is not False
        ), "weight_attr should not be False in Conv."
        self._param_attr = weight_attr
        self._bias_attr = bias_attr
        self._groups = groups
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._data_format = data_format
        self._subm = subm
        self._key = key

        assert (
            padding_mode == 'zeros'
        ), "Currently, only support padding_mode='zeros'"
        assert groups == 1, "Currently, only support groups=1"

        valid_format = {'NDHWC'}
        if data_format not in valid_format:
            raise ValueError(
                "data_format must be one of {}, but got data_format='{}'".format(
                    valid_format, data_format
                )
            )

        channel_last = data_format == "NDHWC"

        dims = 3
        self._stride = convert_to_list(stride, dims, 'stride')
        self._dilation = convert_to_list(dilation, dims, 'dilation')
        self._kernel_size = convert_to_list(kernel_size, dims, 'kernel_size')
        self._padding = padding
        self._padding_mode = padding_mode
        self._updated_padding, self._padding_algorithm = _update_padding_nd(
            padding, channel_last, dims
        )

        # the sparse conv restricts the shape is [D, H, W, in_channels, out_channels]
        filter_shape = self._kernel_size + [
            self._in_channels,
            self._out_channels,
        ]

        def _get_default_param_initializer():
            filter_elem_num = np.prod(self._kernel_size) * self._in_channels
            std = (2.0 / filter_elem_num) ** 0.5
            return Normal(0.0, std)

        self.weight = self.create_parameter(
            shape=filter_shape,
            attr=self._param_attr,
            default_initializer=_get_default_param_initializer(),
        )
        self.bias = self.create_parameter(
            attr=self._bias_attr, shape=[self._out_channels], is_bias=True
        )

    def forward(self, x):
        out = F.conv._conv3d(
            x,
            self.weight,
            bias=self.bias,
            stride=self._stride,
            padding=self._updated_padding,
            dilation=self._dilation,
            groups=self._groups,
            subm=self._subm,
            key=self._key,
            data_format=self._data_format,
        )
        return out

    def extra_repr(self):
        main_str = '{_in_channels}, {_out_channels}, kernel_size={_kernel_size}'
        if self._stride != [1] * len(self._stride):
            main_str += ', stride={_stride}'
        if self._padding != 0:
            main_str += ', padding={_padding}'
        if self._padding_mode != 'zeros':
            main_str += ', padding_mode={_padding_mode}'
        if self._dilation != [1] * len(self._dilation):
            main_str += ', dilation={_dilation}'
        if self._groups != 1:
            main_str += ', groups={_groups}'
        main_str += ', data_format={_data_format}'
        return main_str.format(**self.__dict__)


class _Conv2D(Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        subm=False,
        key=None,
        padding_mode='zeros',
        weight_attr=None,
        bias_attr=None,
        data_format="NHWC",
    ):
        super().__init__()
        assert (
            weight_attr is not False
        ), "weight_attr should not be False in Conv."
        self._param_attr = weight_attr
        self._bias_attr = bias_attr
        self._groups = groups
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._data_format = data_format
        self._subm = subm
        self._key = key

        assert (
            padding_mode == 'zeros'
        ), "Currently, only support padding_mode='zeros'"
        assert groups == 1, "Currently, only support groups=1"

        valid_format = {'NHWC'}
        if data_format not in valid_format:
            raise ValueError(
                "data_format must be one of {}, but got data_format='{}'".format(
                    valid_format, data_format
                )
            )

        channel_last = data_format == "NHWC"

        dims = 2
        self._stride = convert_to_list(stride, dims, 'stride')
        self._dilation = convert_to_list(dilation, dims, 'dilation')
        self._kernel_size = convert_to_list(kernel_size, dims, 'kernel_size')
        self._padding = padding
        self._padding_mode = padding_mode
        self._updated_padding, self._padding_algorithm = _update_padding_nd(
            padding, channel_last, dims
        )

        # the sparse conv restricts the shape is [H, W, in_channels, out_channels]
        filter_shape = self._kernel_size + [
            self._in_channels,
            self._out_channels,
        ]

        def _get_default_param_initializer():
            filter_elem_num = np.prod(self._kernel_size) * self._in_channels
            std = (2.0 / filter_elem_num) ** 0.5
            return Normal(0.0, std)

        self.weight = self.create_parameter(
            shape=filter_shape,
            attr=self._param_attr,
            default_initializer=_get_default_param_initializer(),
        )
        self.bias = self.create_parameter(
            attr=self._bias_attr, shape=[self._out_channels], is_bias=True
        )

    def forward(self, x):
        out = F.conv._conv2d(
            x,
            self.weight,
            bias=self.bias,
            stride=self._stride,
            padding=self._updated_padding,
            dilation=self._dilation,
            groups=self._groups,
            subm=self._subm,
            key=self._key,
            data_format=self._data_format,
        )
        return out

    def extra_repr(self):
        main_str = '{_in_channels}, {_out_channels}, kernel_size={_kernel_size}'
        if self._stride != [1] * len(self._stride):
            main_str += ', stride={_stride}'
        if self._padding != 0:
            main_str += ', padding={_padding}'
        if self._padding_mode != 'zeros':
            main_str += ', padding_mode={_padding_mode}'
        if self._dilation != [1] * len(self._dilation):
            main_str += ', dilation={_dilation}'
        if self._groups != 1:
            main_str += ', groups={_groups}'
        main_str += ', data_format={_data_format}'
        return main_str.format(**self.__dict__)


class Conv3D(_Conv3D):
    r"""
    **Sparse Convlution3d Layer**
    The Sparse convolution3d layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input(Input) and
    Output(Output) are multidimensional SparseCooTensors with a shape of
    :math:`[N, D, H, W, C]` . Where N is batch size, C is the number of
    channels, D is the depth of the feature, H is the height of the feature,
    and W is the width of the feature. If bias attribution is provided,
    bias is added to the output of the convolution.
    For each input :math:`X`, the equation is:

    ..  math::

        Out = W \ast X + b

    In the above equation:

    * :math:`X`: Input value, a tensor with NDHWC format.
    * :math:`W`: Filter value, a tensor with DHWCM format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 1-D tensor with shape [M].
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Parameters:
        in_channels(int): The number of input channels in the input image.
        out_channels(int): The number of output channels produced by the convolution.
        kernel_size(int|list|tuple): The size of the convolving kernel.
        stride(int|list|tuple, optional): The stride size. If stride is a list/tuple, it must
            contain three integers, (stride_D, stride_H, stride_W). Otherwise, the
            stride_D = stride_H = stride_W = stride. The default value is 1.
        padding(int|str|tuple|list, optional): The padding size. Padding coule be in one of the following forms.
            1. a string in ['valid', 'same'].
            2. an int, which means each spartial dimension(depth, height, width) is zero paded by size of `padding`
            3. a list[int] or tuple[int] whose length is the number of spartial dimensions, which contains the amount of padding on each side for each spartial dimension. It has the form [pad_d1, pad_d2, ...].
            4. a list[int] or tuple[int] whose length is 2 * number of spartial dimensions. It has the form  [pad_before, pad_after, pad_before, pad_after, ...] for all spartial dimensions.
            5. a list or tuple of pairs of ints. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension are also included. Each pair of integers correspond to the amount of padding for a dimension of the input. Padding in batch dimension and channel dimension should be [0, 0] or (0, 0).
            The default value is 0.
        dilation(int|list|tuple, optional): The dilation size. If dilation is a list/tuple, it must
            contain three integers, (dilation_D, dilation_H, dilation_W). Otherwise, the
            dilation_D = dilation_H = dilation_W = dilation. The default value is 1.
        groups(int, optional): The groups number of the Conv3D Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. The default value is 1, currently, only support groups=1.
        padding_mode(str, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Currently only support ``'zeros'``.
        weight_attr(ParamAttr, optional): The parameter attribute for learnable parameters/weights
            of conv3d. If it is set to None or one attribute of ParamAttr, conv3d
            will create ParamAttr as param_attr. If it is set to None, the parameter
            is initialized with :math:`Normal(0.0, std)`, and the :math:`std` is
            :math:`(\frac{2.0 }{filter\_elem\_num})^{0.5}`. The default value is None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of conv3d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv3d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. The default value is None.
        data_format(str, optional): Data format that specifies the layout of input.
            It can be "NCDHW" or "NDHWC". Currently, only support "NCDHW".

    Attribute:

        **weight** (Parameter): the learnable weights of filters of this layer.

        **bias** (Parameter): the learnable bias of this layer.

    Shape:

        - x: :math:`(N, D_{in}, H_{in}, W_{in}, C_{in})`

        - weight: :math:`(K_{d}, K_{h}, K_{w}, C_{in}, C_{out})`

        - bias: :math:`(C_{out})`

        - output: :math:`(N, D_{out}, H_{out}, W_{out}, C_{out})`

        Where

        ..  math::

           D_{out}&= \frac{(D_{in} + 2 * paddings[0] - (dilations[0] * (kernel\_size[0] - 1) + 1))}{strides[0]} + 1

           H_{out}&= \frac{(H_{in} + 2 * paddings[1] - (dilations[1] * (kernel\_size[1] - 1) + 1))}{strides[1]} + 1

           W_{out}&= \frac{(W_{in} + 2 * paddings[2] - (dilations[2] * (kernel\_size[2] - 1) + 1))}{strides[2]} + 1

    Examples:

        .. code-block:: python

          import paddle

          indices = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
          values = [[1], [2], [3], [4]]
          indices = paddle.to_tensor(indices, dtype='int32')
          values = paddle.to_tensor(values, dtype='float32')
          dense_shape = [1, 1, 3, 4, 1]
          sparse_x = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape, stop_gradient=True)
          conv = paddle.sparse.nn.Conv3D(1, 1, (1, 3, 3))
          y = conv(sparse_x)
          print(y.shape)
          # (1, 1, 1, 2, 1)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        padding_mode='zeros',
        weight_attr=None,
        bias_attr=None,
        data_format="NDHWC",
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            subm=False,
            key=None,
            padding_mode=padding_mode,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format,
        )


class Conv2D(_Conv2D):
    r"""
    **Sparse Convlution2d Layer**

    The Sparse convolution2d layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input(Input) and
    Output(Output) are multidimensional SparseCooTensors with a shape of
    :math:`[N, H, W, C]` . Where N is batch size, C is the number of
    channels, H is the height of the feature,
    and W is the width of the feature. If bias attribution is provided,
    bias is added to the output of the convolution.
    For each input :math:`X`, the equation is:

    ..  math::

        Out = W \ast X + b

    In the above equation:

    * :math:`X`: Input value, a tensor with NHWC format.
    * :math:`W`: Filter value, a tensor with HWCM format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 1-D tensor with shape [M].
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Parameters:
        in_channels(int): The number of input channels in the input image.
        out_channels(int): The number of output channels produced by the convolution.
        kernel_size(int|list|tuple): The size of the convolving kernel.
        stride(int|list|tuple, optional): The stride size. If stride is a list/tuple, it must
            contain three integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. The default value is 1.
        padding(int|str|tuple|list, optional): The padding size. Padding coule be in one of the following forms.

            1. a string in ['valid', 'same'].
            2. an int, which means each spartial dimension(height, width) is zero paded by size of `padding`
            3. a list[int] or tuple[int] whose length is the number of spartial dimensions, which contains the amount of padding on each side for each spartial dimension. It has the form [pad_d1, pad_d2, ...].
            4. a list[int] or tuple[int] whose length is 2 * number of spartial dimensions. It has the form  [pad_before, pad_after, pad_before, pad_after, ...] for all spartial dimensions.
            5. a list or tuple of pairs of ints. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...].

            Note that, the batch dimension and channel dimension are also included. Each pair of integers correspond to the amount of padding for a dimension of the input. Padding in batch dimension and channel dimension should be [0, 0] or (0, 0).
            The default value is 0.
        dilation(int|list|tuple, optional): The dilation size. If dilation is a list/tuple, it must
            contain three integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. The default value is 1.
        groups(int, optional): The groups number of the Conv2D Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. The default value is 1, currently, only support groups=1.
        padding_mode(str, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Currently only support ``'zeros'``.
        weight_attr(ParamAttr, optional): The parameter attribute for learnable parameters/weights
            of conv2d. If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as param_attr. If it is set to None, the parameter
            is initialized with :math:`Normal(0.0, std)`, and the :math:`std` is
            :math:`(\frac{2.0 }{filter\_elem\_num})^{0.5}`. The default value is None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of conv2d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. The default value is None.
        data_format(str, optional): Data format that specifies the layout of input.
            It can be "NCHW" or "NHWC". Currently, only support "NHWC".

    Attribute:
        **weight** (Parameter): the learnable weights of filters of this layer.

        **bias** (Parameter): the learnable bias of this layer.

    Shape:
        - x: :math:`(N, H_{in}, W_{in}, C_{in})`

        - weight: :math:`(K_{h}, K_{w}, C_{in}, C_{out})`

        - bias: :math:`(C_{out})`

        - output: :math:`(N, H_{out}, W_{out}, C_{out})`

        Where

        ..  math::

           H_{out}&= \frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (kernel\_size[0] - 1) + 1))}{strides[0]} + 1

           W_{out}&= \frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (kernel\_size[1] - 1) + 1))}{strides[1]} + 1

    Examples:

        .. code-block:: python

          import paddle

          indices = [[0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
          values = [[1], [2], [3], [4]]
          indices = paddle.to_tensor(indices, dtype='int32')
          values = paddle.to_tensor(values, dtype='float32')
          dense_shape = [1, 3, 4, 1]
          sparse_x = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape, stop_gradient=True)
          conv = paddle.sparse.nn.Conv2D(1, 1, (3, 3))
          y = conv(sparse_x)
          print(y.shape)
          # (1, 1, 2, 1)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        padding_mode='zeros',
        weight_attr=None,
        bias_attr=None,
        data_format="NHWC",
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            subm=False,
            key=None,
            padding_mode=padding_mode,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format,
        )


class SubmConv3D(_Conv3D):
    r"""
    **Submanifold Sparse Convlution3d Layer**
    The submanifold sparse convolution3d layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input(Input) and
    Output(Output) are multidimensional SparseCooTensors with a shape of
    :math:`[N, D, H, W, C]` . Where N is batch size, C is the number of
    channels, D is the depth of the feature, H is the height of the feature,
    and W is the width of the feature. If bias attribution is provided,
    bias is added to the output of the convolution.
    For each input :math:`X`, the equation is:

    ..  math::

        Out = W \ast X + b

    In the above equation:

    * :math:`X`: Input value, a tensor with NDHWC format.
    * :math:`W`: Filter value, a tensor with DHWCM format.
    * :math:`\\ast`: Submanifold Convolution operation, refer to the paper: https://arxiv.org/abs/1706.01307.
    * :math:`b`: Bias value, a 1-D tensor with shape [M].
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Parameters:
        in_channels(int): The number of input channels in the input image.
        out_channels(int): The number of output channels produced by the convolution.
        kernel_size(int|list|tuple): The size of the convolving kernel.
        stride(int|list|tuple, optional): The stride size. If stride is a list/tuple, it must
            contain three integers, (stride_D, stride_H, stride_W). Otherwise, the
            stride_D = stride_H = stride_W = stride. The default value is 1.
        padding(int|str|tuple|list, optional): The padding size. Padding coule be in one of the following forms.
            1. a string in ['valid', 'same'].
            2. an int, which means each spartial dimension(depth, height, width) is zero paded by size of `padding`
            3. a list[int] or tuple[int] whose length is the number of spartial dimensions, which contains the amount of padding on each side for each spartial dimension. It has the form [pad_d1, pad_d2, ...].
            4. a list[int] or tuple[int] whose length is 2 * number of spartial dimensions. It has the form  [pad_before, pad_after, pad_before, pad_after, ...] for all spartial dimensions.
            5. a list or tuple of pairs of ints. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension are also included. Each pair of integers correspond to the amount of padding for a dimension of the input. Padding in batch dimension and channel dimension should be [0, 0] or (0, 0).
            The default value is 0.
        dilation(int|list|tuple, optional): The dilation size. If dilation is a list/tuple, it must
            contain three integers, (dilation_D, dilation_H, dilation_W). Otherwise, the
            dilation_D = dilation_H = dilation_W = dilation. The default value is 1.
        groups(int, optional): The groups number of the Conv3D Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. The default value is 1.
        padding_mode(str, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Currently only support ``'zeros'``.
        key(str, optional): the key is used to save or use the same rulebook,
            the definition and role of rulebook refers to
            https://pdfs.semanticscholar.org/5125/a16039cabc6320c908a4764f32596e018ad3.pdf. The
            default value is None.
        weight_attr(ParamAttr, optional): The parameter attribute for learnable parameters/weights
            of conv3d. If it is set to None or one attribute of ParamAttr, conv3d
            will create ParamAttr as param_attr. If it is set to None, the parameter
            is initialized with :math:`Normal(0.0, std)`, and the :math:`std` is
            :math:`(\frac{2.0 }{filter\_elem\_num})^{0.5}`. The default value is None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of conv3d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv3d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. The default value is None.
        data_format(str, optional): Data format that specifies the layout of input.
            It can be "NCDHW" or "NDHWC". Currently, only support "NCDHW".

    Attribute:

        **weight** (Parameter): the learnable weights of filters of this layer.

        **bias** (Parameter): the learnable bias of this layer.

    Shape:

        - x: :math:`(N, D_{in}, H_{in}, W_{in}, C_{in})`

        - weight: :math:`(K_{d}, K_{h}, K_{w}, C_{in}, C_{out})`

        - bias: :math:`(C_{out})`

        - output: :math:`(N, D_{out}, H_{out}, W_{out}, C_{out})`

        Where

        ..  math::

           D_{out}&= \frac{(D_{in} + 2 * paddings[0] - (dilations[0] * (kernel\_size[0] - 1) + 1))}{strides[0]} + 1

           H_{out}&= \frac{(H_{in} + 2 * paddings[1] - (dilations[1] * (kernel\_size[1] - 1) + 1))}{strides[1]} + 1

           W_{out}&= \frac{(W_{in} + 2 * paddings[2] - (dilations[2] * (kernel\_size[2] - 1) + 1))}{strides[2]} + 1

    Examples:

        .. code-block:: python

          import paddle

          indices = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
          values = [[1], [2], [3], [4]]
          dense_shape = [1, 1, 3, 4, 1]
          indices = paddle.to_tensor(indices, dtype='int32')
          values = paddle.to_tensor(values, dtype='float32')
          sparse_x = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape, stop_gradient=True)
          subm_conv = paddle.sparse.nn.SubmConv3D(1, 1, (1, 3, 3))
          y = subm_conv(sparse_x)
          print(y.shape)
          # (1, 1, 3, 4, 1)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        padding_mode='zeros',
        key=None,
        weight_attr=None,
        bias_attr=None,
        data_format="NDHWC",
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            subm=True,
            key=key,
            padding_mode=padding_mode,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format,
        )


class SubmConv2D(_Conv2D):
    r"""
    **Submanifold Sparse Convlution2d Layer**

    The submanifold sparse convolution2d layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input(Input) and
    Output(Output) are multidimensional SparseCooTensors with a shape of
    :math:`[N, H, W, C]` . Where N is batch size, C is the number of
    channels, H is the height of the feature,
    and W is the width of the feature. If bias attribution is provided,
    bias is added to the output of the convolution.
    For each input :math:`X`, the equation is:

    ..  math::

        Out = W \ast X + b

    In the above equation:

    * :math:`X`: Input value, a tensor with NDHWC format.
    * :math:`W`: Filter value, a tensor with DHWCM format.
    * :math:`\\ast`: Submanifold Convolution operation, refer to the paper: https://arxiv.org/abs/1706.01307.
    * :math:`b`: Bias value, a 1-D tensor with shape [M].
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Parameters:
        in_channels(int): The number of input channels in the input image.
        out_channels(int): The number of output channels produced by the convolution.
        kernel_size(int|list|tuple): The size of the convolving kernel.
        stride(int|list|tuple, optional): The stride size. If stride is a list/tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. The default value is 1.
        padding(int|str|tuple|list, optional): The padding size. Padding coule be in one of the following forms.

            1. a string in ['valid', 'same'].
            2. an int, which means each spartial dimension(depth, height, width) is zero paded by size of `padding`
            3. a list[int] or tuple[int] whose length is the number of spartial dimensions, which contains the amount of padding on each side for each spartial dimension. It has the form [pad_d1, pad_d2, ...].
            4. a list[int] or tuple[int] whose length is 2 * number of spartial dimensions. It has the form  [pad_before, pad_after, pad_before, pad_after, ...] for all spartial dimensions.
            5. a list or tuple of pairs of ints. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...].

            Note that, the batch dimension and channel dimension are also included. Each pair of integers correspond to the amount of padding for a dimension of the input. Padding in batch dimension and channel dimension should be [0, 0] or (0, 0).
            The default value is 0.
        dilation(int|list|tuple, optional): The dilation size. If dilation is a list/tuple, it must
            contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. The default value is 1.
        groups(int, optional): The groups number of the Conv2D Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. The default value is 1.
        padding_mode(str, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Currently only support ``'zeros'``.
        key(str, optional): the key is used to save or use the same rulebook,
            the definition and role of rulebook refers to
            https://pdfs.semanticscholar.org/5125/a16039cabc6320c908a4764f32596e018ad3.pdf. The
            default value is None.
        weight_attr(ParamAttr, optional): The parameter attribute for learnable parameters/weights
            of conv2d. If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as param_attr. If it is set to None, the parameter
            is initialized with :math:`Normal(0.0, std)`, and the :math:`std` is
            :math:`(\frac{2.0 }{filter\_elem\_num})^{0.5}`. The default value is None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of conv2d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. The default value is None.
        data_format(str, optional): Data format that specifies the layout of input.
            It can be "NCHW" or "NHWC". Currently, only support "NHWC".

    Attribute:
        **weight** (Parameter): the learnable weights of filters of this layer.

        **bias** (Parameter): the learnable bias of this layer.

    Shape:
        - x: :math:`(N, H_{in}, W_{in}, C_{in})`

        - weight: :math:`(K_{h}, K_{w}, C_{in}, C_{out})`

        - bias: :math:`(C_{out})`

        - output: :math:`(N, H_{out}, W_{out}, C_{out})`

        Where

        ..  math::

           H_{out}&= \frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (kernel\_size[0] - 1) + 1))}{strides[0]} + 1

           W_{out}&= \frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (kernel\_size[1] - 1) + 1))}{strides[1]} + 1

    Examples:

        .. code-block:: python

          import paddle

          indices = [[0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
          values = [[1], [2], [3], [4]]
          dense_shape = [1, 3, 4, 1]
          indices = paddle.to_tensor(indices, dtype='int32')
          values = paddle.to_tensor(values, dtype='float32')
          sparse_x = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape, stop_gradient=True)
          subm_conv = paddle.sparse.nn.SubmConv2D(1, 1, (3, 3))
          y = subm_conv(sparse_x)
          print(y.shape)
          # (1, 3, 4, 1)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        padding_mode='zeros',
        key=None,
        weight_attr=None,
        bias_attr=None,
        data_format="NHWC",
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            subm=True,
            key=key,
            padding_mode=padding_mode,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format,
        )
