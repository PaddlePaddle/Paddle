# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


class MaxPool3D(Layer):
    """
    This operation applies 3D max pooling over input features based on the sparse input,
    and kernel_size, stride, padding parameters. Input(X) and Output(Out) are
    in NDHWC format, where N is batch size, C is the number of channels,
    H is the height of the feature,  D is the depth of the feature, and W is the width of the feature.

    Parameters:
        kernel_size(int|list|tuple): The pool kernel size. If the kernel size
            is a tuple or list, it must contain three integers,
            (kernel_size_Depth, kernel_size_Height, kernel_size_Width).
            Otherwise, the pool kernel size will be the cube of an int.
        stride(int|list|tuple, optional): The pool stride size. If pool stride size is a tuple or list,
            it must contain three integers, [stride_Depth, stride_Height, stride_Width).
            Otherwise, the pool stride size will be a cube of an int.
            Default None, then stride will be equal to the kernel_size.
        padding(str|int|list|tuple, optional): The padding size. Padding could be in one of the following forms.
            1. A string in ['valid', 'same'].
            2. An int, which means the feature map is zero padded by size of `padding` on every sides.
            3. A list[int] or tuple(int) whose length is 3, [pad_depth, pad_height, pad_weight] whose value means the padding size of each dimension.
            4. A list[int] or tuple(int) whose length is \6. [pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right] whose value means the padding size of each side.
            5. A list or tuple of pairs of integers. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension should be [0,0] or (0,0).
            The default value is 0.
        ceil_mode(bool, optional): ${ceil_mode_comment}
        return_mask(bool, optional): Whether to return the max indices along with the outputs.
        data_format(str, optional): The data format of the input and output data. An optional string from: `"NCDHW"`,
            `"NDHWC"`. The default is `"NCDHW"`. When it is `"NCDHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_depth, input_height, input_width]`. Currently, only support "NDHWC".
        name(str, optional): For detailed information, please refer to :ref:`api_guide_Name`.
            Usually name is no need to set and None by default.


    Returns:
        A callable object of MaxPool3D.

    Shape:
        - x(Tensor): The input SparseCooTensor of max pool3d operator, which is a 5-D tensor.
          The data type can be float32, float64.
        - output(Tensor): The output tensor of max pool3d  operator, which is a 5-D tensor.
          The data type is same as input x.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn
            import numpy as np

            # max pool3d
            input = paddle.to_tensor(np.random.uniform(-1, 1, [1, 2, 3, 32, 32]).astype(np.float32))
            MaxPool3D = nn.MaxPool3D(kernel_size=2,
                                   stride=2, padding=0)
            output = MaxPool3D(input)
            # output.shape [1, 2, 3, 16, 16]

            # for return_mask=True
            MaxPool3D = nn.MaxPool3D(kernel_size=2, stride=2, padding=0, return_mask=True)
            output, max_indices = MaxPool3D(input)
            # output.shape [1, 2, 3, 16, 16], max_indices.shape [1, 2, 3, 16, 16],
    """

    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 return_mask=False,
                 ceil_mode=False,
                 data_format="NCDHW",
                 name=None):
        super(MaxPool3D, self).__init__()
        self.ksize = kernel_size
        self.stride = stride
        self.padding = padding
        self.return_mask = return_mask
        self.ceil_mode = ceil_mode
        self.data_format = data_format
        self.name = name

    def forward(self, x):
        return F.max_pool3d(
            x,
            kernel_size=self.ksize,
            stride=self.stride,
            padding=self.padding,
            return_mask=self.return_mask,
            ceil_mode=self.ceil_mode,
            data_format=self.data_format,
            name=self.name)

    def extra_repr(self):
        return 'kernel_size={ksize}, stride={stride}, padding={padding}'.format(
            **self.__dict__)
