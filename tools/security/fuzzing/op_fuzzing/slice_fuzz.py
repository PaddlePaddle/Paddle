"""
Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
This module provides an example for paddle.slice.
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

    x_dim1 = m.int_range(0, 20, 'x_dim1')
    x_dim2 = m.int_range(0, 20, 'x_dim2')
    x_dim3 = m.int_range(0, 20, 'x_dim3')
    x_val = m.float_list(x_dim1 * x_dim2 * x_dim3, -10.0, 100.0, 'x_val')
    x = m.tensor(x_val, 3, [x_dim1, x_dim2, x_dim3])

    axis_num = m.int_range(0, 10, 'axis_num')
    # Try negative axis
    axis = m.int_list(axis_num, -1, 30, 'axis')

    starts_dim1 = m.int_range(0, 20, 'starts_dim1')
    starts_val = m.int_list(starts_dim1, -10, 100, 'starts_val')
    starts = m.tensor(starts_val, 1, [starts_dim1], dtype='int32')

    ends_dim1 = m.int_range(0, 20, 'ends_dim1')
    ends_val = m.int_list(ends_dim1, -10, 100, 'ends_val')
    ends = m.tensor(ends_val, 1, [ends_dim1], dtype='int32')

    if IGNORE_ERRS:
        try:
            paddle.slice(x, axis, starts, ends)
        except IgnoredErrors:
            pass
    else:
        paddle.slice(x, axis, starts, ends)


def main():
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
