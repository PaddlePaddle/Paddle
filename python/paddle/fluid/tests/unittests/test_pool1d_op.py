# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
import paddle
import paddle.nn.functional as F
import paddle.fluid as fluid


def adaptive_start_index(index, input_size, output_size):
    return int(np.floor(index * input_size / output_size))


def adaptive_end_index(index, input_size, output_size):
    return int(np.ceil((index + 1) * input_size / output_size))


def max_pool1D_forward_naive(x,
                             ksize,
                             strides,
                             paddings,
                             global_pool=0,
                             ceil_mode=False,
                             exclusive=True,
                             adaptive=False,
                             data_type=np.float64):
    N, C, L = x.shape
    if global_pool == 1:
        ksize = [L]
    if adaptive:
        L_out = ksize[0]
    else:
        L_out = (L - ksize[0] + 2 * paddings[0] + strides[0] - 1
                 ) // strides[0] + 1 if ceil_mode else (
                     L - ksize[0] + 2 * paddings[0]) // strides[0] + 1

    out = np.zeros((N, C, L_out))
    for i in range(L_out):
        if adaptive:
            r_start = adaptive_start_index(i, L, ksize[0])
            r_end = adaptive_end_index(i, L, ksize[0])
        else:
            r_start = np.max((i * strides[0] - paddings[0], 0))
            r_end = np.min((i * strides[0] + ksize[0] - paddings[0], L))
        x_masked = x[:, :, r_start:r_end]

        out[:, :, i] = np.max(x_masked, axis=(2))
    return out


def avg_pool1D_forward_naive(x,
                             ksize,
                             strides,
                             paddings,
                             global_pool=0,
                             ceil_mode=False,
                             exclusive=False,
                             adaptive=False,
                             data_type=np.float64):
    N, C, L = x.shape
    if global_pool == 1:
        ksize = [L]
    if adaptive:
        L_out = ksize[0]
    else:
        L_out = (L - ksize[0] + 2 * paddings[0] + strides[0] - 1
                 ) // strides[0] + 1 if ceil_mode else (
                     L - ksize[0] + 2 * paddings[0]) // strides[0] + 1

    out = np.zeros((N, C, L_out))
    for i in range(L_out):
        if adaptive:
            r_start = adaptive_start_index(i, L, ksize[0])
            r_end = adaptive_end_index(i, L, ksize[0])
        else:
            r_start = np.max((i * strides[0] - paddings[0], 0))
            r_end = np.min((i * strides[0] + ksize[0] - paddings[0], L))
        x_masked = x[:, :, r_start:r_end]

        field_size = (r_end - r_start) \
            if (exclusive or adaptive) else (ksize[0])
        if data_type == np.int8 or data_type == np.uint8:
            out[:, :, i] = (np.rint(
                np.sum(x_masked, axis=(2, 3)) / field_size)).astype(data_type)
        else:
            out[:, :, i] = (np.sum(x_masked, axis=(2)) /
                            field_size).astype(data_type)
    return out


def pool1D_forward_naive(x,
                         ksize,
                         strides,
                         paddings,
                         global_pool=0,
                         ceil_mode=False,
                         exclusive=True,
                         adaptive=False,
                         data_format='NCHW',
                         pool_type="max",
                         padding_algorithm="EXPLICIT"):

    # update paddings
    def _get_padding_with_SAME(input_shape, pool_size, pool_stride):
        padding = []
        for input_size, filter_size, stride_size in zip(input_shape, pool_size,
                                                        pool_stride):
            out_size = int((input_size + stride_size - 1) / stride_size)
            pad_sum = np.max((
                (out_size - 1) * stride_size + filter_size - input_size, 0))
            pad_0 = int(pad_sum / 2)
            pad_1 = int(pad_sum - pad_0)
            padding.append(pad_0)
            padding.append(pad_1)
        return padding

    if isinstance(padding_algorithm, str):
        padding_algorithm = padding_algorithm.upper()
        if padding_algorithm not in ["SAME", "VALID", "EXPLICIT"]:
            raise ValueError("Unknown Attr(padding_algorithm): '%s'. "
                             "It can only be 'SAME' or 'VALID'." %
                             str(padding_algorithm))

        if padding_algorithm == "VALID":
            paddings = [0, 0]
            if ceil_mode != False:
                raise ValueError(
                    "When Attr(pool_padding) is \"VALID\", Attr(ceil_mode)"
                    " must be False. "
                    "Received ceil_mode: True.")
        elif padding_algorithm == "SAME":
            input_data_shape = [x.shape[-1]]
            paddings = _get_padding_with_SAME(input_data_shape, ksize, strides)

    assert len(paddings) == 2 or len(paddings) == 1
    is_sys = True if len(paddings) == 1 else False

    N = x.shape[0]
    C, L = [x.shape[1], x.shape[2]]

    if global_pool == 1:
        ksize = [L]
        paddings = [0 for _ in range(len(paddings))]

    pad_l_left = paddings[0] if is_sys else paddings[0]
    pad_l_right = paddings[1] if is_sys else paddings[1]

    if adaptive:
        L_out = ksize[0]
    else:
        L_out = (L - ksize[0] + pad_l_left + pad_l_right + strides[0] - 1) // strides[0] + 1 \
            if ceil_mode else (L - ksize[0] + pad_l_left + pad_l_right) // strides[0] + 1

    out = np.zeros((N, C, L_out))
    for i in range(L_out):
        if adaptive:
            in_l_start = adaptive_start_index(i, L, ksize[0])
            in_l_end = adaptive_end_index(i, L, ksize[0])
        else:
            in_l_start = np.max((i * strides[0] - pad_l_left, 0))
            in_l_end = np.min((i * strides[0] + ksize[0] - pad_l_right, L))

        x_masked = x[:, :, in_l_start:in_l_end]
        if pool_type == 'avg':
            field_size = ((in_l_end - in_l_start) * (in_l_end - in_l_start)) \
                if (exclusive or adaptive) else (ksize[0])
            out[:, :, i] = np.sum(x_masked, axis=(2)) / field_size
        elif pool_type == 'max':
            out[:, :, i] = np.max(x_masked, axis=(2))

    return out


def test_np_pd_pool1d():
    data = np.random.random((2, 4, 32)).astype('float32')
    with fluid.dygraph.guard():
        # datapd = fluid.layers.assign(data)
        datapd = fluid.dygraph.to_variable(data)

        res_pd = F.avg_pool1d(
            datapd, kernel_size=3, stride=2, padding=1, ceil_mode=False)

        res_np = pool1D_forward_naive(
            data,
            ksize=[3],
            paddings=[1, 1],
            strides=[2],
            ceil_mode=False,
            pool_type='avg',
            exclusive=False)

    np.testing.assert_allclose(res_np, res_pd.numpy())
    print("=> unittest avgpool1d success!")


def test_np_pd_avg_pool1d():
    data = np.random.random((2, 4, 32)).astype('float32')
    with fluid.dygraph.guard():
        # datapd = fluid.layers.assign(data)
        datapd = fluid.dygraph.to_variable(data)

        res_pd = F.avg_pool1d(
            datapd, kernel_size=3, stride=2, padding=1, ceil_mode=True)

        res_np = avg_pool1D_forward_naive(
            data, ksize=[3], paddings=[1], strides=[2], ceil_mode=True)

    np.testing.assert_allclose(res_np, res_pd.numpy())
    print("=> unittest avgpool1d success!")


def test_np_pd_max_pool1d():
    data = np.random.random((2, 4, 32)).astype('float32')
    with fluid.dygraph.guard():
        # datapd = fluid.layers.assign(data)
        datapd = fluid.dygraph.to_variable(data)

        res_pd = F.max_pool1d(
            datapd, kernel_size=3, stride=2, padding=0, ceil_mode=False)

        res_np = max_pool1D_forward_naive(
            data, ksize=[3], paddings=[0], strides=[2], ceil_mode=False)

    np.testing.assert_allclose(res_np, res_pd.numpy())
    print("=> unittest avgpool1d success!")


def test_np_pd_adaptive_avg_pool1d():
    data = np.random.random((1, 2, 16)).astype('float32')
    with fluid.dygraph.guard():
        datapd = fluid.dygraph.to_variable(data)

        res_pd = F.adaptive_avg_pool1d(datapd, 6)

        res_np = avg_pool1D_forward_naive(
            data, ksize=[6], adaptive=True, paddings=[0], strides=[0])

    np.testing.assert_allclose(res_np, res_pd.numpy())
    print("=> unittest adaptive_avg_pool1d success!")


def test_np_pd_adaptive_max_pool1d():
    data = np.random.random((1, 2, 16)).astype('float32')
    with fluid.dygraph.guard():
        datapd = fluid.dygraph.to_variable(data)

        res_pd = F.adaptive_max_pool1d(datapd, 6)

        res_np = max_pool1D_forward_naive(
            data, ksize=[6], adaptive=True, paddings=[0], strides=[0])

    np.testing.assert_allclose(res_np, res_pd.numpy())
    print("=> unittest adaptive_avg_pool1d success!")


if __name__ == '__main__':
    test_np_pd_pool1d()
    test_np_pd_avg_pool1d()
    test_np_pd_max_pool1d()
    test_np_pd_adaptive_avg_pool1d()
    test_np_pd_adaptive_max_pool1d()
