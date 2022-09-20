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

from paddle.fluid.layers import utils
from paddle import _C_ops, _legacy_C_ops, in_dynamic_mode
from paddle.nn.functional.pooling import _update_padding_nd

__all__ = []


def max_pool3d(x,
               kernel_size,
               stride=None,
               padding=0,
               ceil_mode=False,
               data_format="NDHWC",
               name=None):
    """
    Implements sparse max pooling 3d operation.
    See more details in :ref:`api_sparse_pooling_MaxPool3d` .

    Args:
        x (Tensor): The input SparseCooTensor of pooling operator, which is a 5-D tensor with
                          shape [N, D, H, W, C]. The format of input tensor `"NDHWC"`, where N represents batch size, C represents the number of channels, D, H and W represent the depth, height and width of the feature respectively.
        kernel_size (int|list|tuple): The pool kernel size. If the kernel size
            is a tuple or list, it must contain three integers,
            (kernel_size_Depth, kernel_size_Height, kernel_size_Width).
            Otherwise, the pool kernel size will be the cube of an int.
        stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain three integers, [stride_Depth, stride_Height, stride_Width).
            Otherwise, the pool stride size will be a cube of an int.
        padding (string|int|list|tuple): The padding size. Padding could be in one of the following forms.
            1. A string in ['valid', 'same'].
            2. An int, which means the feature map is zero padded by size of `padding` on every sides.
            3. A list[int] or tuple(int) whose length is 3, [pad_depth, pad_height, pad_weight] whose value means the padding size of each dimension.
            4. A list[int] or tuple(int) whose length is 6. [pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right] whose value means the padding size of each side.
            5. A list or tuple of pairs of integers. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension should be [0,0] or (0,0).
            The default value is 0.
        ceil_mode (bool): ${ceil_mode_comment}
        data_format (string): The data format of the input and output data. An optional string from: `"NCDHW"`, `"NDHWC"`.
                        The default is `"NCDHW"`. When it is `"NCDHW"`, the data is stored in the order of:
                        `[batch_size, input_channels, input_depth, input_height, input_width]`. Currently only support `"NDHWC"` .
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        Tensor: The output tensor of pooling result. The data type is same as input tensor.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.fluid.framework import _test_eager_guard

            with _test_eager_guard():
                dense_x = paddle.randn((1, 4, 4, 4, 3))
                sparse_x = dense_x.to_sparse_coo(4)
                kernel_sizes = [3, 3, 3]
                paddings = [0, 0, 0]
                strides = [1, 1, 1]
                out = paddle.incubate.sparse.nn.functional.max_pool3d(sparse_x, kernel_sizes, stride=strides, padding=paddings)
                #[1, 2, 2, 2, 3]
    """

    assert in_dynamic_mode(), "Currently, Sparse API only support dynamic mode"
    assert x.is_sparse_coo(
    ), "Currently, sparse.relu only support the input of SparseCooTensor"
    assert data_format == 'NDHWC', "Currently, sparse.max_pool3d only support data format of 'NDHWC'"

    kernel_size = utils.convert_to_list(kernel_size, 3, 'pool_size')
    if stride is None:
        stride = kernel_size
    else:
        stride = utils.convert_to_list(stride, 3, 'pool_stride')

    channel_last = True

    padding, padding_algorithm = _update_padding_nd(padding,
                                                    3,
                                                    channel_last=channel_last,
                                                    ceil_mode=ceil_mode)

    #TODO(zkh2016): remove the dependency on dilation from the backend
    dilation = [1, 1, 1]

    return _C_ops.sparse_maxpool(x, kernel_size, padding, dilation, stride)
