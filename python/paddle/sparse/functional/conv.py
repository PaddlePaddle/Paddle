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

__all__ = []

from paddle import _C_ops, in_dynamic_mode
from ...fluid.layers.utils import convert_to_list
from paddle.nn.functional.conv import _update_padding_nd
from paddle.fluid.layers.nn import transpose


def conv3d(x,
           weight,
           bias=None,
           stride=1,
           padding=0,
           dilation=1,
           groups=1,
           data_format="NCDHW",
           name=None):
    assert in_dynamic_mode(), "Currently, only support dynamic mode"
    assert bias == None, "Currently, sparse_conv3d does not support bias"
    # Currently, only support 'NDHWC'
    if data_format not in ["NDHWC"]:
        raise ValueError("Attr(data_format) should be 'NDHWC'. Received "
                         "Attr(data_format): {}.".format(data_format))
    if len(x.shape) != 5:
        raise ValueError(
            "Input x should be 5D tensor, but received x with the shape of {}".
            format(x.shape))

    channel_last = (data_format == "NDHWC")
    channel_dim = -1 if channel_last else 1
    if len(x.shape) != 5:
        raise ValueError(
            "Input x should be 5D tensor, but received x with the shape of {}".
            format(x.shape))
    num_channels = x.shape[channel_dim]
    if num_channels < 0:
        raise ValueError(
            "The channel dimension of the input({}) should be defined. "
            "Received: {}.".format(x.shape, num_channels))

    padding, padding_algorithm = _update_padding_nd(padding, channel_last, 3)
    stride = convert_to_list(stride, 3, 'stride')
    dilation = convert_to_list(dilation, 3, 'dilation')
    op_type = "conv3d"

    return _C_ops.final_state_sparse_conv3d(x, weight, padding, dilation,
                                            stride, groups, False)
