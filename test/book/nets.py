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

import paddle


def simple_img_conv_pool(
    input,
    num_filters,
    filter_size,
    pool_size,
    pool_stride,
    pool_padding=0,
    pool_type='max',
    global_pooling=False,
    conv_stride=1,
    conv_padding=0,
    conv_dilation=1,
    conv_groups=1,
    param_attr=None,
    bias_attr=None,
    act=None,
    use_cudnn=True,
):
    r"""
        :api_attr: Static Graph

    The simple_img_conv_pool api is composed of :ref:`api_fluid_layers_conv2d` and :ref:`api_fluid_layers_pool2d` .

    Args:
        input (Variable): 4-D Tensor, shape is [N, C, H, W], data type can be float32 or float64.
        num_filters(int): The number of filters. It is the same as the output channels.
        filter_size (int|list|tuple): The filter size. If filter_size is a list or
            tuple, it must contain two integers, (filter_size_H, filter_size_W). Otherwise,
            the filter_size_H = filter_size_W = filter_size.
        pool_size (int|list|tuple): The pooling size of pool2d layer. If pool_size
            is a list or tuple, it must contain two integers, (pool_size_H, pool_size_W).
            Otherwise, the pool_size_H = pool_size_W = pool_size.
        pool_stride (int|list|tuple): The pooling stride of pool2d layer. If pool_stride
            is a list or tuple, it must contain two integers, (pooling_stride_H, pooling_stride_W).
            Otherwise, the pooling_stride_H = pooling_stride_W = pool_stride.
        pool_padding (int|list|tuple): The padding of pool2d layer. If pool_padding is a list or
            tuple, it must contain two integers, (pool_padding_H, pool_padding_W).
            Otherwise, the pool_padding_H = pool_padding_W = pool_padding. Default 0.
        pool_type (str): Pooling type can be :math:`max` for max-pooling or :math:`avg` for
            average-pooling. Default :math:`max`.
        global_pooling (bool): Whether to use the global pooling. If global_pooling = true,
            pool_size and pool_padding while be ignored. Default False
        conv_stride (int|list|tuple): The stride size of the conv2d Layer. If stride is a
            list or tuple, it must contain two integers, (conv_stride_H, conv_stride_W). Otherwise,
            the conv_stride_H = conv_stride_W = conv_stride. Default: conv_stride = 1.
        conv_padding (int|list|tuple): The padding size of the conv2d Layer. If padding is
            a list or  tuple, it must contain two integers, (conv_padding_H, conv_padding_W).
            Otherwise, the conv_padding_H = conv_padding_W = conv_padding. Default: conv_padding = 0.
        conv_dilation (int|list|tuple): The dilation size of the conv2d Layer. If dilation is
            a list or tuple, it must contain two integers, (conv_dilation_H, conv_dilation_W).
            Otherwise, the conv_dilation_H = conv_dilation_W = conv_dilation. Default: conv_dilation = 1.
        conv_groups (int): The groups number of the conv2d Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1.
        param_attr (ParamAttr|None): The parameter attribute for learnable parameters/weights
            of conv2d. If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with :math:`Normal(0.0, std)`,
            and the :math:`std` is :math:`(\\frac{2.0 }{filter\_elem\_num})^{0.5}`.
            Default: None.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias of conv2d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        act (str): Activation type for conv2d, if it is set to None, activation is not
            appended. Default: None.
        use_cudnn (bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True

    Return:
        4-D Tensor, the result of input after conv2d and pool2d, with the same data type as :attr:`input`

    Return Type:
        Variable

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            img = paddle.static.data(name='img', shape=[100, 1, 28, 28], dtype='float32')
            conv_pool = fluid.nets.simple_img_conv_pool(input=img,
                                                        filter_size=5,
                                                        num_filters=20,
                                                        pool_size=2,
                                                        pool_stride=2,
                                                        act="relu")
    """
    conv_out = paddle.static.nn.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=conv_stride,
        padding=conv_padding,
        dilation=conv_dilation,
        groups=conv_groups,
        param_attr=param_attr,
        bias_attr=bias_attr,
        act=act,
        use_cudnn=use_cudnn,
    )
    if pool_type == 'max':
        pool_out = paddle.nn.functional.max_pool2d(
            x=conv_out,
            kernel_size=pool_size,
            stride=pool_stride,
            padding=pool_padding,
        )
    else:
        pool_out = paddle.nn.functional.avg_pool2d(
            x=conv_out,
            kernel_size=pool_size,
            stride=pool_stride,
            padding=pool_padding,
        )
    return pool_out


def img_conv_group(
    input,
    conv_num_filter,
    pool_size,
    conv_padding=1,
    conv_filter_size=3,
    conv_act=None,
    param_attr=None,
    conv_with_batchnorm=False,
    conv_batchnorm_drop_rate=0.0,
    pool_stride=1,
    pool_type="max",
    use_cudnn=True,
):
    """
        :api_attr: Static Graph

    The Image Convolution Group is composed of Convolution2d, BatchNorm, DropOut,
    and Pool2D. According to the input arguments, img_conv_group will do serials of
    computation for Input using Convolution2d, BatchNorm, DropOut, and pass the last
    result to Pool2D.

    Args:
        input (Variable): The input is 4-D Tensor with shape [N, C, H, W], the data type of input is float32 or float64.
        conv_num_filter(list|tuple): Indicates the numbers of filter of this group.
        pool_size (int|list|tuple): The pooling size of Pool2D Layer. If pool_size
            is a list or tuple, it must contain two integers, (pool_size_height, pool_size_width).
            Otherwise, the pool_size_height = pool_size_width = pool_size.
        conv_padding (int|list|tuple): The padding size of the Conv2D Layer. If padding is
            a list or tuple, its length must be equal to the length of conv_num_filter.
            Otherwise the conv_padding of all Conv2D Layers are the same. Default 1.
        conv_filter_size (int|list|tuple): The filter size. If filter_size is a list or
            tuple, its length must be equal to the length of conv_num_filter.
            Otherwise the conv_filter_size of all Conv2D Layers are the same. Default 3.
        conv_act (str): Activation type for Conv2D Layer that is not followed by BatchNorm.
            Default: None.
        param_attr (ParamAttr): The parameters to the Conv2D Layer. Default: None
        conv_with_batchnorm (bool|list): Indicates whether to use BatchNorm after Conv2D Layer.
            If conv_with_batchnorm is a list, its length must be equal to the length of
            conv_num_filter. Otherwise, conv_with_batchnorm indicates whether all the
            Conv2D Layer follows a BatchNorm. Default False.
        conv_batchnorm_drop_rate (float|list): Indicates the drop_rate of Dropout Layer
            after BatchNorm. If conv_batchnorm_drop_rate is a list, its length must be
            equal to the length of conv_num_filter. Otherwise, drop_rate of all Dropout
            Layers is conv_batchnorm_drop_rate. Default 0.0.
        pool_stride (int|list|tuple): The pooling stride of Pool2D layer. If pool_stride
            is a list or tuple, it must contain two integers, (pooling_stride_H,
            pooling_stride_W). Otherwise, the pooling_stride_H = pooling_stride_W = pool_stride.
            Default 1.
        pool_type (str): Pooling type can be :math:`max` for max-pooling and :math:`avg` for
            average-pooling. Default :math:`max`.
        use_cudnn (bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True

    Return:
        A Variable holding Tensor representing the final result after serial computation using Convolution2d,
        BatchNorm, DropOut, and Pool2D, whose data type is the same with input.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()

            img = paddle.static.data(name='img', shape=[None, 1, 28, 28], dtype='float32')
            conv_pool = fluid.nets.img_conv_group(input=img,
                                                  conv_padding=1,
                                                  conv_num_filter=[3, 3],
                                                  conv_filter_size=3,
                                                  conv_act="relu",
                                                  pool_size=2,
                                                  pool_stride=2)
    """
    tmp = input
    assert isinstance(conv_num_filter, list) or isinstance(
        conv_num_filter, tuple
    )

    def __extend_list__(obj):
        if not hasattr(obj, '__len__'):
            return [obj] * len(conv_num_filter)
        else:
            assert len(obj) == len(conv_num_filter)
            return obj

    conv_padding = __extend_list__(conv_padding)
    conv_filter_size = __extend_list__(conv_filter_size)
    param_attr = __extend_list__(param_attr)
    conv_with_batchnorm = __extend_list__(conv_with_batchnorm)
    conv_batchnorm_drop_rate = __extend_list__(conv_batchnorm_drop_rate)

    for i in range(len(conv_num_filter)):
        local_conv_act = conv_act
        if conv_with_batchnorm[i]:
            local_conv_act = None

        tmp = paddle.static.nn.conv2d(
            input=tmp,
            num_filters=conv_num_filter[i],
            filter_size=conv_filter_size[i],
            padding=conv_padding[i],
            param_attr=param_attr[i],
            act=local_conv_act,
            use_cudnn=use_cudnn,
        )

        if conv_with_batchnorm[i]:
            tmp = paddle.static.nn.batch_norm(input=tmp, act=conv_act)
            drop_rate = conv_batchnorm_drop_rate[i]
            if abs(drop_rate) > 1e-5:
                tmp = paddle.nn.functional.dropout(x=tmp, p=drop_rate)

    if pool_type == 'max':
        pool_out = paddle.nn.functional.max_pool2d(
            x=tmp,
            kernel_size=pool_size,
            stride=pool_stride,
        )
    else:
        pool_out = paddle.nn.functional.avg_pool2d(
            x=tmp,
            kernel_size=pool_size,
            stride=pool_stride,
        )
    return pool_out
