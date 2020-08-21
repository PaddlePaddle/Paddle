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

from ...fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype
from ...fluid.layers import utils
from ...fluid.dygraph import layers
from ...fluid.layer_helper import LayerHelper
from .. import functional as F

__all__ = [
    'AdaptiveAvgPool2d',
    'AdaptiveAvgPool3d',
]


class AdaptiveAvgPool2d(layers.Layer):
    """

    This operation applies 2D adaptive avg pooling on input tensor. The h and w dimensions
    of the output tensor are determined by the parameter output_size.

    For avg adaptive pool2d:

    ..  math::

       hstart &= floor(i * H_{in} / H_{out})

       hend &= ceil((i + 1) * H_{in} / H_{out})

       wstart &= floor(j * W_{in} / W_{out})

       wend &= ceil((j + 1) * W_{in} / W_{out})

       Output(i ,j) &= \\frac{sum(Input[hstart:hend, wstart:wend])}{(hend - hstart) * (wend - wstart)}


    Parameters:
        output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain two element, (H, W). H and W can be either a int, or None which means
            the size will be the same as that of the input.
        data_format (str): The data format of the input and output data. An optional string
            from: "NCHW", "NHWC". The default is "NCHW". When it is "NCHW", the data is stored in
            the order of: [batch_size, input_channels, input_height, input_width].
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Shape:
        x (Tensor): The input tensor of adaptive avg pool2d operator, which is a 4-D tensor. The data type can be float16, float32, float64, int32 or int64.
        output (Tensor): The output tensor of adaptive avg pool2d operator, which is a 4-D tensor. The data type is same as input x.

    Returns:
        A callable object of AdaptiveAvgPool2d.

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
            adaptive_avg_pool = paddle.nn.AdaptiveAvgPool2d(output_size=3)
            pool_out = adaptive_avg_pool(x = x)
            # pool_out.shape is [2, 3, 3, 3]
    """

    def __init__(self, output_size, data_format="NCHW", name=None):
        super(AdaptiveAvgPool2d, self).__init__()
        self._output_size = output_size
        self._data_format = data_format
        self._name = name

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x,
            output_size=self._output_size,
            data_format=self._data_format,
            name=self._name)


class AdaptiveAvgPool3d(layers.Layer):
    """

    This operation applies 3D adaptive avg pooling on input tensor. The h and w dimensions
    of the output tensor are determined by the parameter output_size.

    For avg adaptive pool3d:

    ..  math::

      dstart &= floor(i * D_{in} / D_{out})

      dend &= ceil((i + 1) * D_{in} / D_{out})

      hstart &= floor(j * H_{in} / H_{out})

      hend &= ceil((j + 1) * H_{in} / H_{out})

      wstart &= floor(k * W_{in} / W_{out})

      wend &= ceil((k + 1) * W_{in} / W_{out})

      Output(i ,j, k) &= \\frac{sum(Input[dstart:dend, hstart:hend, wstart:wend])}{(dend - dstart) * (hend - hstart) * (wend - wstart)}


    Parameters:
        output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain three elements, (D, H, W). D, H and W can be either a int, or None which means
            the size will be the same as that of the input.
        data_format (str): The data format of the input and output data. An optional string
            from: "NCDHW", "NDHWC". The default is "NCDHW". When it is "NCDHW", the data is stored in
            the order of: [batch_size, input_channels, input_depth, input_height, input_width].
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
    Shape:
        x (Tensor): The input tensor of adaptive avg pool3d operator, which is a 5-D tensor. The data type can be float16, float32, float64, int32 or int64.
        output (Tensor): The output tensor of adaptive avg pool3d operator, which is a 5-D tensor. The data type is same as input x.

    Returns:
        A callable object of AdaptiveAvgPool3d.

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
            adaptive_avg_pool = paddle.nn.AdaptiveAvgPool3d(output_size=3)
            pool_out = adaptive_avg_pool(x = x)
            # pool_out = [2, 3, 3, 3, 3]
    """

    def __init__(self, output_size, data_format="NCDHW", name=None):
        super(AdaptiveAvgPool3d, self).__init__()
        self._output_size = output_size
        self._data_format = data_format
        self._name = name

    def forward(self, x):
        return F.adaptive_avg_pool3d(
            x,
            output_size=self._output_size,
            data_format=self._data_format,
            name=self._name)
