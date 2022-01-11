"""
Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
This module provides an example for paddle.matmul.
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

    transpose_x = m.bool('transpose_x')
    transpose_y = m.bool('transpose_y')

    # Try rank 0-3 random tensors.
    x, _ = m.tensor_with_diff_shape(max_rank=3)
    y, _ = m.tensor_with_diff_shape(max_rank=3)

    if IGNORE_ERRS:
        try:
            paddle.matmul(x, y, transpose_x, transpose_y)
        except IgnoredErrors:
            pass
    else:
        paddle.matmul(x, y, transpose_x, transpose_y)


def main():
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
