"""
Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
This module provides an example for paddle.nn.functional.diag_embed.
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

    input_tensor, rank = m.tensor_with_diff_shape(min_val=0.0, max_val=30.0, min_dim=0, max_dim=10, max_rank=5)
    offset = m.int_range(-20, 20, 'offset')
    dim1 = m.int_range(0, rank, 'o_dim1')
    dim2 = m.int_range(0, rank, 'o_dim2')

    if dim1 == dim2:
        dim1 -= 1

    if IGNORE_ERRS:
        try:
            paddle.nn.functional.diag_embed(input_tensor, offset=offset, dim1=dim1, dim2=dim2)
        except IgnoredErrors:
            pass
    else:
        paddle.nn.functional.diag_embed(input_tensor, offset=offset, dim1=dim1, dim2=dim2)


def main():
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
