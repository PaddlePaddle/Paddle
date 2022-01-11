"""
Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
This module provides an example for paddle.nn.functional.interpolate.
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

    x, x_rank = m.tensor_with_diff_shape(min_val=0.0, max_val=30.0, min_dim=0, max_dim=10, max_rank=5)
    size, size_rank = m.tensor_with_diff_shape(min_val=0.0, max_val=30.0, min_dim=0, max_dim=10, max_rank=5)
    scale_factor, sf_rank = m.tensor_with_diff_shape(min_val=0.0, max_val=30.0, min_dim=0, max_dim=10, max_rank=5)

    mode_list = ["bilinear", "trilinear", "nearest", "bicubic", "linear", "area"]
    mode = mode_list[m.int_range(0, len(mode_list) - 1, 'mode')]
    align_corners = m.bool('align_corners')
    align_mode = m.int_range(-10, 10, 'align_mode')

    if IGNORE_ERRS:
        try:
            paddle.nn.functional.interpolate(x, size=size, scale_factor=scale_factor, mode=mode,
                                             align_corners=align_corners, align_mode=align_mode)
        except IgnoredErrors:
            pass
    else:
        paddle.nn.functional.interpolate(x, size=size, scale_factor=scale_factor, mode=mode,
                                         align_corners=align_corners,
                                         align_mode=align_mode)


def main():
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
