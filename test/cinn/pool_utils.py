#!/usr/bin/env python3

# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

import math
import sys

import numpy as np


def pool2d(np_data, attrs, dtype="float32"):
    pool_type = "max"
    ceil_mode = False
    exclusive = True
    data_format = "NCHW"
    for key in attrs.attr_store:
        if key == "kernel_size":
            kernel_size = attrs.get_attr("kernel_size")
        elif key == "stride_size":
            stride_size = attrs.get_attr("stride_size")
        elif key == "padding_size":
            padding_size = attrs.get_attr("padding_size")
        elif key == "pool_type":
            pool_type = attrs.get_attr("pool_type")
        elif key == "ceil_mode":
            ceil_mode = attrs.get_attr("ceil_mode")
        elif key == "exclusive":
            exclusive = attrs.get_attr("exclusive")
        elif key == "data_format":
            data_format = attrs.get_attr("data_format")
        else:
            raise ValueError(f"attr_store {key} is not supported")

    if data_format == "NCHW":
        in_n, in_c, in_h, in_w = in_shape = np_data.shape
        height_axis = 2
        width_axis = 3
    elif data_format == "NHWC":
        in_n, in_h, in_w, in_c = in_shape = np_data.shape
        height_axis = 1
        width_axis = 2
    else:
        raise ValueError(f"data_format {data_format} is not supported")

    if isinstance(kernel_size, int):
        k_h = k_w = kernel_size
    else:
        k_h, k_w = kernel_size
    if isinstance(stride_size, int):
        s_h = s_w = stride_size
    else:
        s_h, s_w = stride_size
    if isinstance(padding_size, int):
        pt = pl = pb = pr = padding_size
    else:
        pt, pl, pb, pr = padding_size

    out_shape = list(in_shape)
    if ceil_mode:
        out_shape[height_axis] = int(
            math.ceil(float(in_shape[height_axis] - k_h + pt + pb) / s_h) + 1
        )
        out_shape[width_axis] = int(
            math.ceil(float(in_shape[width_axis] - k_w + pl + pr) / s_w) + 1
        )
    else:
        out_shape[height_axis] = int(
            math.floor(float(in_shape[height_axis] - k_h + pt + pb) / s_h) + 1
        )
        out_shape[width_axis] = int(
            math.floor(float(in_shape[width_axis] - k_w + pl + pr) / s_w) + 1
        )

    fill_value = 0
    if exclusive and pool_type == 'max':
        fill_value = sys.float_info.min

    if data_format == "NCHW":
        pad_np = np.full(
            shape=(in_n, in_c, in_h + pt + pb, in_w + pl + pr),
            fill_value=fill_value,
            dtype=dtype,
        )
        no_zero = (
            range(in_n),
            range(in_c),
            range(pt, in_h + pt),
            range(pl, in_w + pl),
        )
    else:
        pad_np = np.full(
            shape=(in_n, in_h + pt + pb, in_w + pl + pr, in_c),
            fill_value=fill_value,
            dtype=dtype,
        )
        no_zero = (
            range(in_n),
            range(pt, in_h + pt),
            range(pl, in_w + pl),
            range(in_c),
        )

    pad_np[np.ix_(*no_zero)] = np_data
    ret_np = np.zeros(shape=out_shape).astype(dtype)
    if pool_type == 'avg':
        for i in range(out_shape[height_axis]):
            for j in range(out_shape[width_axis]):
                if exclusive:
                    pad_exclusive = pad_np.copy()
                    pad_exclusive[np.ix_(*no_zero)] = 1
                    if data_format == "NCHW":
                        pad_count = np.sum(
                            pad_exclusive[
                                :,
                                :,
                                i * s_h : i * s_h + k_h,
                                j * s_w : j * s_w + k_w,
                            ]
                            == 1,
                            axis=(height_axis, width_axis),
                        )
                        ret_np[:, :, i, j] = np.sum(
                            pad_np[
                                :,
                                :,
                                i * s_h : i * s_h + k_h,
                                j * s_w : j * s_w + k_w,
                            ],
                            axis=(height_axis, width_axis),
                        ) / np.maximum(pad_count, 1)
                    else:
                        pad_count = np.sum(
                            pad_exclusive[
                                :,
                                i * s_h : i * s_h + k_h,
                                j * s_w : j * s_w + k_w,
                                :,
                            ]
                            == 1,
                            axis=(height_axis, width_axis),
                        )
                        ret_np[:, i, j, :] = np.sum(
                            pad_np[
                                :,
                                i * s_h : i * s_h + k_h,
                                j * s_w : j * s_w + k_w,
                                :,
                            ],
                            axis=(height_axis, width_axis),
                        ) / np.maximum(pad_count, 1)
                else:
                    if data_format == "NCHW":
                        ret_np[:, :, i, j] = np.mean(
                            pad_np[
                                :,
                                :,
                                i * s_h : i * s_h + k_h,
                                j * s_w : j * s_w + k_w,
                            ],
                            axis=(height_axis, width_axis),
                        )
                    else:
                        ret_np[:, i, j, :] = np.mean(
                            pad_np[
                                :,
                                i * s_h : i * s_h + k_h,
                                j * s_w : j * s_w + k_w,
                                :,
                            ],
                            axis=(height_axis, width_axis),
                        )
    elif pool_type == 'max':
        for i in range(out_shape[height_axis]):
            for j in range(out_shape[width_axis]):
                if data_format == "NCHW":
                    ret_np[:, :, i, j] = np.max(
                        pad_np[
                            :,
                            :,
                            i * s_h : i * s_h + k_h,
                            j * s_w : j * s_w + k_w,
                        ],
                        axis=(height_axis, width_axis),
                    )
                else:
                    ret_np[:, i, j, :] = np.max(
                        pad_np[
                            :,
                            i * s_h : i * s_h + k_h,
                            j * s_w : j * s_w + k_w,
                            :,
                        ],
                        axis=(height_axis, width_axis),
                    )
    else:
        raise ValueError(f"pool type {pool_type} is not supported")

    ret_np = np.maximum(ret_np, fill_value)
    return ret_np, [out_shape]


def pool3d(np_data, attrs, dtype="float32"):
    pool_type = "max"
    ceil_mode = False
    exclusive = True
    data_format = "NCDHW"
    for key in attrs.attr_store:
        if key == "kernel_size":
            kernel_size = attrs.get_attr("kernel_size")
        elif key == "stride_size":
            stride_size = attrs.get_attr("stride_size")
        elif key == "padding_size":
            padding_size = attrs.get_attr("padding_size")
        elif key == "pool_type":
            pool_type = attrs.get_attr("pool_type")
        elif key == "ceil_mode":
            ceil_mode = attrs.get_attr("ceil_mode")
        elif key == "exclusive":
            exclusive = attrs.get_attr("exclusive")
        elif key == "data_format":
            data_format = attrs.get_attr("data_format")
        else:
            raise ValueError(f"attr_store {key} is not supported")

    if data_format == "NCDHW":
        in_n, in_c, in_d, in_h, in_w = in_shape = np_data.shape
        depth_axis = 2
        height_axis = 3
        width_axis = 4
    elif data_format == "NDHWC":
        in_n, in_d, in_h, in_w, in_c = in_shape = np_data.shape
        depth_axis = 1
        height_axis = 2
        width_axis = 3
    else:
        raise ValueError(f"data_format {data_format} is not supported")

    if isinstance(kernel_size, int):
        k_d = k_h = k_w = kernel_size
    else:
        k_d, k_h, k_w = kernel_size
    if isinstance(stride_size, int):
        s_d = s_h = s_w = stride_size
    else:
        s_d, s_h, s_w = stride_size
    if isinstance(padding_size, int):
        pf = pt = pl = pk = pb = pr = padding_size
    else:
        pf, pt, pl, pk, pb, pr = padding_size

    out_shape = list(in_shape)
    if ceil_mode:
        out_shape[depth_axis] = int(
            math.ceil(float(in_shape[depth_axis] - k_d + pf + pk) / s_d) + 1
        )
        out_shape[height_axis] = int(
            math.ceil(float(in_shape[height_axis] - k_h + pt + pb) / s_h) + 1
        )
        out_shape[width_axis] = int(
            math.ceil(float(in_shape[width_axis] - k_w + pl + pr) / s_w) + 1
        )
    else:
        out_shape[depth_axis] = int(
            math.floor(float(in_shape[depth_axis] - k_d + pf + pk) / s_d) + 1
        )
        out_shape[height_axis] = int(
            math.floor(float(in_shape[height_axis] - k_h + pt + pb) / s_h) + 1
        )
        out_shape[width_axis] = int(
            math.floor(float(in_shape[width_axis] - k_w + pl + pr) / s_w) + 1
        )

    fill_value = 0
    if exclusive and pool_type == 'max':
        fill_value = sys.float_info.min

    if data_format == "NCDHW":
        pad_np = np.full(
            shape=(in_n, in_c, in_d + pf + pk, in_h + pt + pb, in_w + pl + pr),
            fill_value=fill_value,
            dtype=dtype,
        )
        no_zero = (
            range(in_n),
            range(in_c),
            range(pf, in_d + pf),
            range(pt, in_h + pt),
            range(pl, in_w + pl),
        )
    else:
        pad_np = np.full(
            shape=(in_n, in_d + pf + pk, in_h + pt + pb, in_w + pl + pr, in_c),
            fill_value=fill_value,
            dtype=dtype,
        )
        no_zero = (
            range(in_n),
            range(pf, in_d + pf),
            range(pt, in_h + pt),
            range(pl, in_w + pl),
            range(in_c),
        )

    pad_np[np.ix_(*no_zero)] = np_data
    ret_np = np.zeros(shape=out_shape).astype(dtype)
    if pool_type == 'avg':
        for i in range(out_shape[depth_axis]):
            for j in range(out_shape[height_axis]):
                for k in range(out_shape[width_axis]):
                    if exclusive:
                        pad_exclusive = pad_np.copy()
                        pad_exclusive[np.ix_(*no_zero)] = 1
                        if data_format == "NCDHW":
                            pad_count = np.sum(
                                pad_exclusive[
                                    :,
                                    :,
                                    i * s_d : i * s_d + k_d,
                                    j * s_h : j * s_h + k_h,
                                    k * s_w : k * s_w + k_w,
                                ]
                                == 1,
                                axis=(depth_axis, height_axis, width_axis),
                            )
                            ret_np[:, :, i, j, k] = np.sum(
                                pad_np[
                                    :,
                                    :,
                                    i * s_d : i * s_d + k_d,
                                    j * s_h : j * s_h + k_h,
                                    k * s_w : k * s_w + k_w,
                                ],
                                axis=(depth_axis, height_axis, width_axis),
                            ) / np.maximum(pad_count, 1)
                        else:
                            pad_count = np.sum(
                                pad_exclusive[
                                    :,
                                    i * s_d : i * s_d + k_d,
                                    j * s_h : j * s_h + k_h,
                                    k * s_w : k * s_w + k_w,
                                    :,
                                ]
                                == 1,
                                axis=(depth_axis, height_axis, width_axis),
                            )
                            ret_np[:, i, j, k, :] = np.sum(
                                pad_np[
                                    :,
                                    i * s_d : i * s_d + k_d,
                                    j * s_h : j * s_h + k_h,
                                    k * s_w : k * s_w + k_w,
                                    :,
                                ],
                                axis=(depth_axis, height_axis, width_axis),
                            ) / np.maximum(pad_count, 1)
                    else:
                        if data_format == "NCDHW":
                            ret_np[:, :, i, j, k] = np.mean(
                                pad_np[
                                    :,
                                    :,
                                    i * s_d : i * s_d + k_d,
                                    j * s_h : j * s_h + k_h,
                                    k * s_w : k * s_w + k_w,
                                ],
                                axis=(depth_axis, height_axis, width_axis),
                            )
                        else:
                            ret_np[:, i, j, k, :] = np.mean(
                                pad_np[
                                    :,
                                    i * s_d : i * s_d + k_d,
                                    j * s_h : j * s_h + k_h,
                                    k * s_w : k * s_w + k_w,
                                    :,
                                ],
                                axis=(depth_axis, height_axis, width_axis),
                            )
    elif pool_type == 'max':
        for i in range(out_shape[depth_axis]):
            for j in range(out_shape[height_axis]):
                for k in range(out_shape[width_axis]):
                    if data_format == "NCDHW":
                        ret_np[:, :, i, j, k] = np.max(
                            pad_np[
                                :,
                                :,
                                i * s_d : i * s_d + k_d,
                                j * s_h : j * s_h + k_h,
                                k * s_w : k * s_w + k_w,
                            ],
                            axis=(depth_axis, height_axis, width_axis),
                        )
                    else:
                        ret_np[:, i, j, k, :] = np.max(
                            pad_np[
                                :,
                                i * s_d : i * s_d + k_d,
                                j * s_h : j * s_h + k_h,
                                k * s_w : k * s_w + k_w,
                                :,
                            ],
                            axis=(depth_axis, height_axis, width_axis),
                        )
    else:
        raise ValueError(f"pool type {pool_type} is not supported")

    ret_np = np.maximum(ret_np, fill_value)
    return ret_np, [out_shape]


def pool1d(np_data, attrs, dtype="float32"):
    pool_type = "max"
    ceil_mode = False
    exclusive = True
    data_format = "NCW"
    for key in attrs.attr_store:
        if key == "kernel_size":
            kernel_size = attrs.get_attr("kernel_size")
        elif key == "stride_size":
            stride_size = attrs.get_attr("stride_size")
        elif key == "padding_size":
            padding_size = attrs.get_attr("padding_size")
        elif key == "pool_type":
            pool_type = attrs.get_attr("pool_type")
        elif key == "ceil_mode":
            ceil_mode = attrs.get_attr("ceil_mode")
        elif key == "exclusive":
            exclusive = attrs.get_attr("exclusive")
        elif key == "data_format":
            data_format = attrs.get_attr("data_format")
        else:
            raise ValueError(f"attr_store {key} is not supported")

    if data_format == "NCW":
        in_n, in_c, in_w = in_shape = np_data.shape
        width_axis = 2
    elif data_format == "NWC":
        in_n, in_w, in_c = in_shape = np_data.shape
        width_axis = 1
    else:
        raise ValueError(f"data_format {data_format} is not supported")

    if isinstance(kernel_size, int):
        k_w = kernel_size
    else:
        (k_w,) = kernel_size
    if isinstance(stride_size, int):
        s_w = stride_size
    else:
        (s_w,) = stride_size
    if isinstance(padding_size, int):
        pl = pr = padding_size
    else:
        pl, pr = padding_size

    out_shape = list(in_shape)
    if ceil_mode:
        out_shape[width_axis] = int(
            math.ceil(float(in_shape[width_axis] - k_w + pl + pr) / s_w) + 1
        )
    else:
        out_shape[width_axis] = int(
            math.floor(float(in_shape[width_axis] - k_w + pl + pr) / s_w) + 1
        )

    fill_value = 0
    if exclusive and pool_type == 'max':
        fill_value = sys.float_info.min

    if data_format == "NCW":
        pad_np = np.full(
            shape=(in_n, in_c, in_w + pl + pr),
            fill_value=fill_value,
            dtype=dtype,
        )
        no_zero = (range(in_n), range(in_c), range(pl, in_w + pl))
    else:
        pad_np = np.full(
            shape=(in_n, in_w + pl + pr, in_c),
            fill_value=fill_value,
            dtype=dtype,
        )
        no_zero = (range(in_n), range(pl, in_w + pl), range(in_c))

    pad_np[np.ix_(*no_zero)] = np_data
    ret_np = np.zeros(shape=out_shape).astype(dtype)
    if pool_type == 'avg':
        for i in range(out_shape[width_axis]):
            if exclusive:
                pad_exclusive = pad_np.copy()
                pad_exclusive[np.ix_(*no_zero)] = 1
                if data_format == "NCW":
                    pad_count = np.sum(
                        pad_exclusive[:, :, i * s_w : i * s_w + k_w] == 1,
                        axis=width_axis,
                    )
                    ret_np[:, :, i] = np.sum(
                        pad_np[:, :, i * s_w : i * s_w + k_w], axis=width_axis
                    ) / np.maximum(pad_count, 1)
                else:
                    pad_count = np.sum(
                        pad_exclusive[:, i * s_w : i * s_w + k_w, :] == 1,
                        axis=width_axis,
                    )
                    ret_np[:, i, :] = np.sum(
                        pad_np[:, i * s_w : i * s_w + k_w, :], axis=width_axis
                    ) / np.maximum(pad_count, 1)
            else:
                if data_format == "NCW":
                    ret_np[:, :, i] = np.mean(
                        pad_np[:, :, i * s_w : i * s_w + k_w], axis=width_axis
                    )
                else:
                    ret_np[:, i, :] = np.mean(
                        pad_np[:, i * s_w : i * s_w + k_w, :], axis=width_axis
                    )
    elif pool_type == 'max':
        for k in range(out_shape[width_axis]):
            if data_format == "NCW":
                ret_np[:, :, k] = np.max(
                    pad_np[:, :, k * s_w : k * s_w + k_w], axis=width_axis
                )
            else:
                ret_np[:, k, :] = np.max(
                    pad_np[:, k * s_w : k * s_w + k_w, :], axis=width_axis
                )
    else:
        raise ValueError(f"pool type {pool_type} is not supported")

    ret_np = np.maximum(ret_np, fill_value)
    return ret_np, [out_shape]
