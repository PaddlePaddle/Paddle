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
from ...fluid.layers import pool2d  #DEFINE_ALIAS
from ...fluid.layers import pool3d  #DEFINE_ALIAS
from ...fluid.layers import adaptive_pool2d  #DEFINE_ALIAS
from ...fluid.layers import adaptive_pool3d  #DEFINE_ALIAS

__all__ = [
    'pool2d', 'pool3d', 'adaptive_pool2d', 'adaptive_pool3d',
    'adaptive_max_pool2d', 'adaptive_max_pool3d'
]


def check_input(x, dimension):
    if len(x.shape) != dimension:
        raise ValueError(
            "Excepted Input X is {}-D tensor, but received {}-D {}".format(
                dimension, len(x.shape), type(x)))


def adaptive_max_pool2d(x, output_size, return_indices=False, name=None):
    """
        This operation applies a 2D adaptive max pooling on input tensor.
        See more details in :ref:`api_nn_pooling_AdaptiveMaxPool2d` .

        Args:
            x (Tensor): The input tensor of adaptive max pool2d operator, which is a 4-D tensor. The data type can be float16, float32, float64, int32 or int64.
            output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list, it must contain two elements, (H, W). H and W can be either a int, or None which means the size will be the same as that of the input.
            return_indices (bool): If true, the index of max pooling point will be returned along with outputs. Default False.
            name(str, optional): For detailed information, please refer to :ref:`api_guide_Name`. Usually name is no need to set and None by default.

        Returns:
            Tensor: The output tensor of adaptive max pool2d result. The data type is same as input tensor.

        Examples:
            .. code-block:: python

              # max adaptive pool2d
              # suppose input data in the shape of [N, C, H, W], `output_size` is [m, n]
              # output shape is [N, C, m, n], adaptive pool divide H and W dimensions
              # of input data into m*n grids averagely and performs poolings in each
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
              # x.shape is [2, 3, 32, 32]
              pool_out = paddle.nn.functional.adaptive_max_pool2d(
                            x = x,
                            output_size=[3, 3])
              # pool_out.shape is [2, 3, 3, 3]
    """
    # In kernel, the input is still named 'input', not 'x'
    if not in_dygraph_mode():
        check_variable_and_dtype(x, 'input', ['float32', 'float64'],
                                 'adaptive_max_pool2d')
    check_input(x, 4)
    check_type(output_size, 'pool_size', (int), 'adaptive_max_pool2d')
    check_type(return_indices, 'return_indices', bool, 'adaptive_max_pool2d')

    in_h, in_w = x.shape[2:4]
    if isinstance(output_size, int):
        output_size = utils.convert_to_list(output_size, 2, 'output_size')
    else:
        if output_size[0] == None:
            output_size[0] = in_h
        if output_size[1] == None:
            output_size[1] = in_w

    if in_dygraph_mode():
        return core.ops.max_pool2d_with_index(x, 'pooling_type', 'max', 'ksize',
                                              output_size, 'adaptive', True)

    l_type = 'max_pool2d_with_index'

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    mask = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": pool_out, "Mask": mask}

    helper.append_op(
        type=l_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": 'max',
            "ksize": output_size,
            "adaptive": True,
        })

    return (pool_out, mask) if return_indices else pool_out


def adaptive_max_pool3d(x, output_size, return_indices=False, name=None):
    """
        This operation applies a 3D adaptive max pooling on input tensor.
        See more details in :ref:`api_nn_pooling_AdaptiveMaxPool3d` .

        Args:
            x (Tensor): The input tensor of adaptive max pool3d operator, which is a 5-D tensor. The data type can be float32, float64.
            output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list, it must contain three elements, (D, H, W). D, H and W can be either a int, or None which means the size will be the same as that of the input.
            return_indices (bool): If true, the index of max pooling point will be returned along with outputs. Default False.
            name(str, optional): For detailed information, please refer to :ref:`api_guide_Name`. Usually name is no need to set and None by default.

        Returns:
            Tensor: The output tensor of adaptive max pool3d result. The data type is same as input tensor.

        Examples:
            .. code-block:: python

              # adaptive max pool3d
              # suppose input data in the shape of [N, C, D, H, W], `output_size` is [l, m, n]
              # output shape is [N, C, l, m, n], adaptive pool divide D, H and W dimensions
              # of input data into m*n grids averagely and performs poolings in each
              # grid to get output.
              # adaptive max pool performs calculations as follow:
              #
              #     for i in range(l):
              #         for j in range(m):
              #             for k in range(n):
              #                 dstart = floor(i * D / l)
              #                 dend = ceil((i + 1) * D / l)
              #                 hstart = floor(i * H / m)
              #                 hend = ceil((i + 1) * H / m)
              #                 wstart = floor(i * W / n)
              #                 wend = ceil((i + 1) * W / n)
              #             output[:, :, i, j, k] = max(input[:, :, dstart: dend, hstart: hend, wstart: wend])
              #
              import paddle
              import numpy as np
              paddle.disable_static()
              input_data = np.random.rand(2, 3, 8, 32, 32)
              x = paddle.to_tensor(input_data)
              # x.shape is [2, 3, 8, 32, 32]
              pool_out = paddle.nn.functional.adaptive_max_pool3d(
                            x = x,
                            output_size=[3, 3, 3])
              # pool_out.shape is [2, 3, 3, 3, 3]
    """

    # In kernel, the input is still named 'input', not 'x'
    if not in_dygraph_mode():
        check_variable_and_dtype(x, 'input', ['float32', 'float64'],
                                 'adaptive_max_pool3d')
    check_input(x, 5)
    check_type(output_size, 'pool_size', (int), 'adaptive_max_pool3d')
    check_type(return_indices, 'return_indices', bool, 'adaptive_max_pool3d')

    in_l, in_h, in_w = x.shape[2:5]
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
        return core.ops.max_pool3d_with_index(x, 'pooling_type', 'max', 'ksize',
                                              output_size, 'adaptive', True)

    l_type = 'max_pool3d_with_index'

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    mask = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": pool_out, "Mask": mask}

    helper.append_op(
        type=l_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": 'max',
            "ksize": output_size,
            "adaptive": True,
        })

    return (pool_out, mask) if return_indices else pool_out
