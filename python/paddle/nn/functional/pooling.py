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

# TODO: define pooling functions
import paddle
from ...fluid import core
from ...fluid.layers import pool2d  #DEFINE_ALIAS
from ...fluid.layers import pool3d  #DEFINE_ALIAS
from ...fluid.layers import adaptive_pool2d  #DEFINE_ALIAS
from ...fluid.layers import adaptive_pool3d  #DEFINE_ALIAS
from ...fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype
from ...fluid.layers import utils
from ...fluid.layer_helper import LayerHelper
from ...fluid.framework import in_dygraph_mode

__all__ = [
    'pool2d', 'pool3d', 'adaptive_pool2d', 'adaptive_pool3d',
    'adaptive_avg_pool2d', 'adaptive_avg_pool3d'
]


def adaptive_avg_pool2d(x, output_size, data_format='NCHW', name=None):
    """

    This operation applies 2D adaptive avg pooling on input tensor. The h and w dimensions
    of the output tensor are determined by the parameter output_size.
    See more detail in :ref:`api_nn_pooling_AdaptiveAvgPool2d` .

    For avg adaptive pool2d:

    ..  math::

       hstart &= floor(i * H_{in} / H_{out})

       hend &= ceil((i + 1) * H_{in} / H_{out})

       wstart &= floor(j * W_{in} / W_{out})

       wend &= ceil((j + 1) * W_{in} / W_{out})

       Output(i ,j) &= \\frac{sum(Input[hstart:hend, wstart:wend])}{(hend - hstart) * (wend - wstart)}

    Args:
        x (Tensor): The input tensor of adaptive avg pool2d operator, which is a 4-D tensor.
                          The data type can be float16, float32, float64, int32 or int64.
        output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain two element, (H, W). H and W can be either a int, or None which means
            the size will be the same as that of the input.
        data_format (str): The data format of the input and output data. An optional string
            from: "NCHW", "NHWC". The default is "NCHW". When it is "NCHW", the data is stored in
            the order of: [batch_size, input_channels, input_height, input_width].
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        Tensor: The output tensor of avg adaptive pool2d result. The data type is same as input tensor.

    Raises:
        ValueError: If `data_format` is not "NCHW" or "NHWC".

    Examples:
        .. code-block:: python

            # adaptive avg pool2d
            # suppose input data in shape of [N, C, H, W], `output_size` is [m, n],
            # output shape is [N, C, m, n], adaptive pool divide H and W dimensions
            # of input data into m * n grids averagely and performs poolings in each
            # grid to get output.
            # adaptive avg pool performs calculations as follow:
            #
            #     for i in range(m):
            #         for j in range(n):
            #             hstart = floor(i * H / m)
            #             hend = ceil((i + 1) * H / m)
            #             wstart = floor(i * W / n)
            #             wend = ceil((i + 1) * W / n)
            #             output[:, :, i, j] = avg(input[:, :, hstart: hend, wstart: wend])
            #
            import paddle
            import numpy as np
            paddle.disable_static()
            input_data = np.random.rand(2, 3, 32, 32)
            x = paddle.to_tensor(input_data)
            # x.shape is [2, 3, 32, 32]
            pool_out = paddle.nn.functional.adaptive_avg_pool2d(
                            x = x,
                            output_size=[3, 3])
            # pool_out.shape is [2, 3, 3, 3]
    """
    if not in_dygraph_mode():
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64', 'int32', 'int64'],
            'adaptive_avg_pool2d')
    check_type(data_format, 'data_format', str, 'adaptive_avg_pool2d')

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'. Received "
            "Attr(data_format): %s." % str(data_format))

    if data_format == "NCHW":
        in_h, in_w = x.shape[2:4]
    else:
        in_h, in_w = x.shape[1:3]

    if isinstance(output_size, int):
        output_size = utils.convert_to_list(output_size, 2, 'output_size')
    else:
        if output_size[0] == None:
            output_size[0] = in_h
        if output_size[1] == None:
            output_size[1] = in_w

    if in_dygraph_mode():
        output = core.ops.pool2d(x, 'pooling_type', 'avg', 'ksize', output_size,
                                 'global_pooling', False, 'adaptive', True,
                                 'data_format', data_format)
        return output

    l_type = 'pool2d'

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    outputs = {"Out": pool_out}

    helper.append_op(
        type=l_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": "avg",
            "ksize": output_size,
            "adaptive": True,
            "data_format": data_format,
        })

    return pool_out


def adaptive_avg_pool3d(x, output_size, data_format='NCDHW', name=None):
    """

    This operation applies 3D adaptive avg pooling on input tensor. The h and w dimensions
    of the output tensor are determined by the parameter output_size.
    See more detail in :ref:`api_nn_pooling_AdaptiveAvgPool3d` .

    For avg adaptive pool3d:

    ..  math::

      dstart &= floor(i * D_{in} / D_{out})

      dend &= ceil((i + 1) * D_{in} / D_{out})

      hstart &= floor(j * H_{in} / H_{out})

      hend &= ceil((j + 1) * H_{in} / H_{out})

      wstart &= floor(k * W_{in} / W_{out})

      wend &= ceil((k + 1) * W_{in} / W_{out})

      Output(i ,j, k) &= \\frac{sum(Input[dstart:dend, hstart:hend, wstart:wend])}{(dend - dstart) * (hend - hstart) * (wend - wstart)}

    Args:
        x (Tensor): The input tensor of adaptive avg pool3d operator, which is a 5-D tensor.
                          The data type can be float16, float32, float64, int32 or int64.
        output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain three elements, (D, H, W). D, H and W can be either a int, or None which means
            the size will be the same as that of the input.
        data_format (str): The data format of the input and output data. An optional string
            from: "NCDHW", "NDHWC". The default is "NCDHW". When it is "NCDHW", the data is stored in
            the order of: [batch_size, input_channels, input_depth, input_height, input_width].
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        Tensor: The output tensor of avg adaptive pool3d result. The data type is same as input tensor.

    Raises:
        ValueError: If `data_format` is not "NCDHW" or "NDHWC".

    Examples:
        .. code-block:: python

            # adaptive avg pool3d
            # suppose input data in shape of [N, C, D, H, W], `output_size` is [l, m, n],
            # output shape is [N, C, l, m, n], adaptive pool divide D, H and W dimensions
            # of input data into l * m * n grids averagely and performs poolings in each
            # grid to get output.
            # adaptive avg pool performs calculations as follow:
            #
            #     for i in range(l):
            #         for j in range(m):
            #             for k in range(n):
            #                 dstart = floor(i * D / l)
            #                 dend = ceil((i + 1) * D / l)
            #                 hstart = floor(j * H / m)
            #                 hend = ceil((j + 1) * H / m)
            #                 wstart = floor(k * W / n)
            #                 wend = ceil((k + 1) * W / n)
            #                 output[:, :, i, j, k] =
            #                     avg(input[:, :, dstart:dend, hstart: hend, wstart: wend])
            import paddle
            import numpy as np
            paddle.disable_static()
            input_data = np.random.rand(2, 3, 8, 32, 32)
            x = paddle.to_tensor(input_data)
            # x.shape is [2, 3, 8, 32, 32]
            pool_out = paddle.nn.functional.adaptive_avg_pool3d(
                            x = x,
                            output_size=[3, 3, 3])
            # pool_out.shape is [2, 3, 3, 3, 3]
    """
    if not in_dygraph_mode():
        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64', 'int32', 'int64'],
            'adaptive_avg_pool3d')
    check_type(data_format, 'data_format', str, 'adaptive_avg_pool3d')

    if data_format not in ["NCDHW", "NDHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCDHW' or 'NDHWC'. Received "
            "Attr(data_format): %s." % str(data_format))

    if data_format == "NCDHW":
        in_l, in_h, in_w = x.shape[2:5]
    else:
        in_l, in_h, in_w = x.shape[1:4]

    if isinstance(output_size, int):
        output_size = utils.convert_to_list(output_size, 3, 'output_size')
    else:
        if output_size[0] == None:
            output_size[0] = in_l
        if output_size[1] == None:
            output_size[1] = in_h
        if output_size[2] == None:
            output_size[2] = in_w

    if in_dygraph_mode():
        output = core.ops.pool3d(x, 'pooling_type', 'avg', 'ksize', output_size,
                                 'global_pooling', False, 'adaptive', True,
                                 'data_format', data_format)
        return output

    l_type = 'pool3d'

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": pool_out}

    helper.append_op(
        type=l_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": "avg",
            "ksize": output_size,
            "adaptive": True,
            "data_format": data_format,
        })

    return pool_out
