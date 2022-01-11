"""
Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
This module provides an example for paddle.nn.Upsample.
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
    size = []
    for i in range(2):
        size.append(m.int_range(0, 20, 'size'))

    scale_factor = []
    for i in range(2):
        scale_factor.append(m.float_range(0.001, 20.0, 'scale_factor'))

    is_size = m.bool('is_size')
    align_corners = m.bool('align_corners')
    align_mode = m.int_range(0, 20, 'align_mode')

    if IGNORE_ERRS:
        try:
            if is_size:
                upsample_out = paddle.nn.Upsample(size=size, align_corners=align_corners, align_mode=align_mode)
                output = upsample_out(input_tensor)
            else:
                upsample_out = paddle.nn.Upsample(scale_factor=scale_factor, align_corners=align_corners,
                                                  align_mode=align_mode)
                output = upsample_out(input_tensor)
        except IgnoredErrors:
            pass
    else:
        if is_size:
            upsample_out = paddle.nn.Upsample(size=size, align_corners=align_corners, align_mode=align_mode)
            output = upsample_out(input_tensor)
        else:
            upsample_out = paddle.nn.Upsample(scale_factor=scale_factor, align_corners=align_corners,
                                              align_mode=align_mode)
            output = upsample_out(input_tensor)


def main():
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
