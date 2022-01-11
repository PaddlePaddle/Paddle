"""
Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
This module provides an example for paddle.nn.Conv2D.
"""
import sys
import atheris_no_libfuzzer as atheris
import paddle
import os
from fuzz_util import Mutator, IgnoredErrors

# Switches for logging and ignoring errors.
LOGGING = os.getenv('PD_FUZZ_LOGGING') == '1'
IGNORE_ERRS = os.getenv('IGNORE_ERRS') == '1'


def TestOneInput(input_bytes):
    m = Mutator(input_bytes, LOGGING)

    data_format = m.pick(['NCHW', 'NHWC'], 'data_format')
    in_channels = m.int_range(1, 100, 'in_channels')
    out_channels = m.int_range(1, 100, 'out_channels')
    kernel_size_type = m.pick(['int', 'list'], 'kernel_size_type')
    if kernel_size_type == 'int':
        kernel_size = m.int_range(1, 100, 'kernel_size')
    else:
        kernel_size = m.int_list(2, 1, 100, 'kernel_size')
    stride_type = m.pick(['int', 'list'], 'stride_type')
    if stride_type == 'int':
        stride = m.int_range(1, 10, 'stride')
    else:
        stride = m.int_list(2, 1, 10, 'stride')

    # `padding` could be one of the following forms.
    # 1. A string in ['VALID', 'SAME'].
    # 2. An int.
    # 3. A list[int] or tuple[int]. [pad_height, pad_weight].
    # 4. A list[int] or tuple[int]. [pad_height_top, pad_height_bottom, pad_width_left, pad_width_right].
    # 5. A list or tuple of pairs of ints. [
    #       [0, 0], [0, 0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]
    #    ]
    #    or [
    #       [0, 0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0, 0]
    #    ] according to `data_format`.
    padding_type = m.pick(['int', 'list', 'str'], 'padding_type')
    if padding_type == 'int':
        padding = m.int_range(0, 10, 'padding')
    elif padding_type == 'list':
        padding_list_type = m.pick(['4_list', '4_int', '2_int'], 'padding_list_type')
        if padding_list_type == '4_list':
            if data_format == 'NCHW':
                padding_height_top = m.int_range(-10, 10, 'padding_height_top')
                padding_height_bottom = m.int_range(-10, 10, 'padding_height_bottom')
                padding_width_left = m.int_range(-10, 10, 'padding_width_left')
                padding_width_right = m.int_range(-10, 10, 'padding_width_right')
                padding = [[0, 0], [0, 0], [padding_height_top, padding_height_bottom],
                           [padding_width_left, padding_width_right]]
            else:
                padding_height_top = m.int_range(-10, 10, 'padding_height_top')
                padding_height_bottom = m.int_range(-10, 10, 'padding_height_bottom')
                padding_width_left = m.int_range(-10, 10, 'padding_width_left')
                padding_width_right = m.int_range(-10, 10, 'padding_width_right')
                padding = [[0, 0], [padding_height_top, padding_height_bottom],
                           [padding_width_left, padding_width_right], [0, 0]]
        elif padding_list_type == '4_int':
            padding_height_top = m.int_range(-10, 10, 'padding_height_top')
            padding_height_bottom = m.int_range(-10, 10, 'padding_height_bottom')
            padding_width_left = m.int_range(-10, 10, 'padding_width_left')
            padding_width_right = m.int_range(-10, 10, 'padding_width_right')
            padding = [padding_height_top, padding_height_bottom, padding_width_left, padding_width_right]
        else:
            padding_height = m.int_range(-10, 10, 'padding_height')
            padding_width = m.int_range(-10, 10, 'padding_width')
            padding = [padding_height, padding_width]
    else:
        padding = m.pick(['VALID', 'SAME'], 'padding')

    dilation_type = m.pick(['int', 'list'], 'dilation_type')
    if dilation_type == 'int':
        dilation = m.int_range(-10, 10, 'dilation')
    else:
        dilation = m.int_list(2, -10, 10, 'dilation')

    groups = m.int_range(1, 10, 'groups')
    padding_mode = m.pick(['zeros', 'reflect', 'replicate', 'circular'], 'padding_mode')

    has_weight = m.bool('has_weight')
    weight_attr = m.param_attr() if has_weight else None
    has_bias = m.bool('has_bias')
    bias_bool = m.bool('bias_bool')
    if bias_bool:
        bias_attr = False
    else:
        bias_attr = m.param_attr() if has_bias else None

    x_dim1 = m.int_range(0, 5, 'x_dim1')
    x_dim2 = m.int_range(0, 5, 'x_dim2')
    x_dim3 = m.int_range(0, 5, 'x_dim3')
    x_dim4 = m.int_range(0, 5, 'x_dim4')
    x_val = m.float_list(x_dim1 * x_dim2 * x_dim3 * x_dim4, -10.0, 1000.0, 'x_val')
    x = m.tensor(x_val, 4, [x_dim1, x_dim2, x_dim3, x_dim4])

    if IGNORE_ERRS:
        try:
            conv = paddle.nn.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, groups=groups,
                                    padding_mode=padding_mode, weight_attr=weight_attr, bias_attr=bias_attr,
                                    data_format=data_format)
            conv(x)
        except IgnoredErrors:
            pass
    else:
        conv = paddle.nn.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, dilation=dilation, groups=groups,
                                padding_mode=padding_mode, weight_attr=weight_attr, bias_attr=bias_attr,
                                data_format=data_format)
        conv(x)


def main():
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
