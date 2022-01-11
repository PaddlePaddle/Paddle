"""
Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
This module provides an example for paddle.static.nn.conv3d.
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

    input_tensor, rank = m.tensor_with_diff_shape(min_val=0.0, max_val=30.0, min_dim=0, max_dim=10, max_rank=4)
    groups = m.int_range(0, 10, 'groups')
    num_filters = m.int_range(0, 5, 'num_filters')
    filter_size = (m.int_range(0, 5, 'filter_size'), m.int_range(0, 5, 'filter_size'), m.int_range(0, 5, 'filter_size'))
    stride = m.int_range(0, 20, 'stride')
    padding = m.int_range(0, 20, 'padding')
    dilation = m.int_range(0, 20, 'dilation')
    use_cudnn = m.bool('use_cudnn')
    act_list = ['tanh', 'softmax', 'sigmoid', 'relu']
    act = act_list[m.int_range(0, len(act_list) - 1, 'act_list')]

    if IGNORE_ERRS:
        try:
            paddle.static.nn.conv3d(input=input_tensor, num_filters=num_filters, filter_size=filter_size,
                                    stride=stride, padding=padding, dilation=dilation, groups=groups, param_attr=None,
                                    bias_attr=None, use_cudnn=use_cudnn, act=act)
        except IgnoredErrors:
            pass
    else:
        paddle.static.nn.conv3d(input=input_tensor, num_filters=num_filters, filter_size=filter_size, stride=stride,
                                padding=padding, dilation=dilation, groups=groups, param_attr=None, bias_attr=None,
                                use_cudnn=use_cudnn, act=act)


def main():
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
