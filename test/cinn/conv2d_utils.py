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

import numpy as np

import paddle
from paddle import static


def conv2d_native(inputs_data, input_shape, filter_size, attrs, is_depthwise):
    main_program = static.Program()
    paddle.enable_static()

    with static.program_guard(main_program, static.Program()):
        padding = [0, 0]
        stride = [1, 1]
        dilation = [1, 1]
        data_format = "NCHW"
        groups = 1
        for key in attrs.attr_store:
            if key == "stride":
                stride = attrs.get_attr("stride")
            elif key == "padding":
                padding = attrs.get_attr("padding")
            elif key == "dilation":
                dilation = attrs.get_attr("dilation")
            elif key == "groups":
                groups = attrs.get_attr("groups")
            elif key == "data_format":
                data_format = attrs.get_attr("data_format")
            else:
                raise ValueError(f"attr_store {key} is not supported")

        img = static.data(name='img', shape=input_shape[1:], dtype='float32')
        if is_depthwise:
            if data_format == "NCHW":
                cin_index = 1
            else:
                cin_index = 3
            filter_size_new = [
                filter_size[1] * input_shape[cin_index],
                filter_size[0] // groups,
                filter_size[2],
                filter_size[3],
            ]
        else:
            filter_size_new = filter_size
        param = paddle.nn.initializer.NumpyArrayInitializer(
            np.array(inputs_data[1]).reshape(filter_size_new).astype("float32")
        )
        # filter: (c_out, c_in // group, kernel_h, kernel_w)
        filter_hw = list(filter_size_new[2:4])
        if data_format == "NHWC":
            filter_hw = list(filter_size_new[1:3])
        if isinstance(stride, int):
            stride = [stride.copy(), stride.copy()]
        if isinstance(padding, int):
            padding = [padding.copy(), padding.copy()]
        if isinstance(dilation, int):
            dilation = [dilation.copy(), dilation.copy()]

        c_index = 1 if data_format == "NCHW" else 3
        res = paddle.nn.Conv2D(
            in_channels=input_shape[c_index],
            out_channels=filter_size_new[0],
            kernel_size=filter_hw,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            data_format=data_format,
            weight_attr=param,
        )(img)
        exe = static.Executor(paddle.CPUPlace())
        exe.run(static.default_startup_program())

        x = np.array(inputs_data[0]).reshape(input_shape).astype("float32")
        output = exe.run(feed={"img": x}, fetch_list=[res])
        output = np.array(output)
        print("output's shape is:", output.shape)

    res_shape = output.shape[1:]

    return output, [res_shape]
