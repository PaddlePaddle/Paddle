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

# TODO: define pooling functions of neural network  
from ...fluid.dygraph import layers
from .. import functional as F

__all__ = [
    'AdaptiveMaxPool2d',
    'AdaptiveMaxPool3d',
]


class AdaptiveMaxPool2d(layers.Layer):
    """
    This operation applies 2D adaptive max pooling on input tensor. The h and w dimensions
    of the output tensor are determined by the parameter output_size. The difference between adaptive pooling and pooling is adaptive one focus on the output size.

    For adaptive max pool2d:
    ..  math::
       hstart &= floor(i * H_{in} / H_{out})
       hend &= ceil((i + 1) * H_{in} / H_{out})
       wstart &= floor(j * W_{in} / W_{out})
       wend &= ceil((j + 1) * W_{in} / W_{out})
       Output(i ,j) &= max(Input[hstart:hend, wstart:wend])

    Parameters:
        output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list, it must contain two element, (H, W). H and W can be either a int, or None which means the size will be the same as that of the input.
        return_indices (bool): If true, the index of max pooling point will be returned along with outputs. It cannot be set in average pooling type. Default False.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Shape:
        x (Tensor): The input tensor of adaptive max pool2d operator, which is a 4-D tensor. The data type can be float32, float64.
        output (Tensor): The output tensor of adaptive max pool2d operator, which is a 4-D tensor. The data type is same as input x.
    
    Returns:
        A callable object of AdaptiveMaxPool2d.

    Examples:
        .. code-block:: python
            # adaptive max pool2d
            # suppose input data in shape of [N, C, H, W], `output_size` is [m, n],
            # output shape is [N, C, m, n], adaptive pool divide H and W dimensions
            # of input data into m * n grids averagely and performs poolings in each
            # grid to get output.
            # adaptive max pool performs calculations as follow:
            #
            #     for i in range(m):
            #         for j in range(n):
            #             hstart = floor(i * H / m)
            #             hend = ceil((i + 1) * H / m)
            #             wstart = floor(i * W / n)
            #             wend = ceil((i + 1) * W / n)
            #             output[:, :, i, j] = max(input[:, :, hstart: hend, wstart: wend])
            #
            import paddle
            import numpy as np
            paddle.disable_static()
            input_data = np.random.rand(2, 3, 32, 32)
            x = paddle.to_tensor(input_data)
            adaptive_max_pool = paddle.nn.AdaptiveMaxPool2d(output_size=3, return_indices=True)
            pool_out, indices = adaptive_max_pool(x = x)
    """

    def __init__(self, output_size, return_indices=False, name=None):
        super(AdaptiveMaxPool2d, self).__init__()
        self._output_size = output_size
        self._return_indices = return_indices
        self._name = name

    def forward(self, x):
        return F.adaptive_max_pool2d(
            x,
            output_size=self._output_size,
            return_indices=self._return_indices,
            name=self._name)


class AdaptiveMaxPool3d(layers.Layer):
    """
   This operation applies 3D adaptive max pooling on input tensor. The h and w dimensions
    of the output tensor are determined by the parameter output_size. The difference between adaptive pooling and pooling is adaptive one focus on the output size.

    For adaptive max pool3d:
    ..  math::
      dstart &= floor(i * D_{in} / D_{out})
      dend &= ceil((i + 1) * D_{in} / D_{out})
      hstart &= floor(j * H_{in} / H_{out})
      hend &= ceil((j + 1) * H_{in} / H_{out})
      wstart &= floor(k * W_{in} / W_{out})
      wend &= ceil((k + 1) * W_{in} / W_{out})
      Output(i ,j, k) &= max(Input[dstart:dend, hstart:hend, wstart:wend])

    Parameters:
        output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain three elements, (D, H, W). D, H and W can be either a int, or None which means
            the size will be the same as that of the input.
        return_indices (bool): If true, the index of max pooling point will be returned along with outputs. Default False.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
    Shape:
        x (Tensor): The input tensor of adaptive max pool3d operator, which is a 5-D tensor. The data type can be float32, float64.
        output (Tensor): The output tensor of adaptive max pool3d operator, which is a 5-D tensor. The data type is same as input x.

    Returns:
        A callable object of AdaptiveMaxPool3d.

    Examples:
        .. code-block:: python
            # adaptive max pool3d
            # suppose input data in shape of [N, C, D, H, W], `output_size` is [l, m, n],
            # output shape is [N, C, l, m, n], adaptive pool divide D, H and W dimensions
            # of input data into l * m * n grids averagely and performs poolings in each
            # grid to get output.
            # adaptive max pool performs calculations as follow:
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
            #                     max(input[:, :, dstart:dend, hstart: hend, wstart: wend])
            import paddle
            import numpy as np
            paddle.disable_static()
            input_data = np.random.rand(2, 3, 8, 32, 32)
            x = paddle.to_tensor(input_data)
            pool = paddle.nn.AdaptiveMaxPool3d(output_size=4)
            out = pool(x)
            # out shape: [2, 3, 4, 4, 4]
            pool, indices = paddle.nn.AdaptiveMaxPool3d(output_size=3, return_indices=True)
            out = pool(x)
            # out shape: [2, 3, 4, 4, 4], indices shape: [2, 3, 4, 4, 4]

            
    """

    def __init__(self, output_size, return_indices="NCDHW", name=None):
        super(AdaptiveMaxPool3d, self).__init__()
        self._output_size = output_size
        self._return_indices = return_indices
        self._name = name

    def forward(self, x):
        return F.adaptive_max_pool3d(
            x,
            output_size=self._output_size,
            return_indices=self._return_indices,
            name=self._name)
