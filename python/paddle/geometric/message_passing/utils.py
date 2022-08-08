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

import paddle


def reshape_lhs_rhs(x, y):
    """
    Expand dims to ensure there will be no broadcasting issues with different
    number of dimensions.
    """
    if len(x.shape) == 1:
        x = paddle.reshape(x, [-1, 1])
    if len(y.shape) == 1:
        y = paddle.reshape(y, [-1, 1])

    x_shape = paddle.shape(x)
    y_shape = paddle.shape(y)
    if len(x.shape) != len(y.shape):
        max_ndims = max(len(x.shape), len(y.shape))
        x_pad_ndims = max_ndims - len(x.shape)
        y_pad_ndims = max_ndims - len(y.shape)
        new_x_shape = [
            x_shape[0],
        ] + [
            1,
        ] * x_pad_ndims + list(x_shape[1:])
        new_y_shape = [
            y_shape[0],
        ] + [
            1,
        ] * y_pad_ndims + list(y_shape[1:])
        x = paddle.reshape(x, new_x_shape)
        y = paddle.reshape(y, new_y_shape)

    return x, y
