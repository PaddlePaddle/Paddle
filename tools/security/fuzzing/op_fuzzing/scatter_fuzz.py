"""
Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
This module provides an example for paddle.scatter.
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

    overwrite = m.bool('overwrite')

    # `x` is N-D Tensor and `updates` shape should be the same as `x` shape.
    # The N-D N is not necessary to be very big. The experience tell us up to 5 is enough to cover
    # the path. Code logic is not effected by big N or very big N.
    x_dims = m.int_range(1, 3, 'x_dims')
    if x_dims == 1:
        x_dim1 = m.int_range(0, 20, 'x_dim1')
        x_val = m.float_list(x_dim1, -10.0, 1000.0, 'x_val')
        x = m.tensor(x_val, 1, [x_dim1])
        updates_val = m.float_list(x_dim1, -10.0, 1000.0, 'updates_val')
        updates = m.tensor(updates_val, 1, [x_dim1])
    elif x_dims == 2:
        x_dim1 = m.int_range(0, 20, 'x_dim1')
        x_dim2 = m.int_range(0, 20, 'x_dim2')
        x_val = m.float_list(x_dim1 * x_dim2, -10.0, 1000.0, 'x_val')
        x = m.tensor(x_val, 2, [x_dim1, x_dim2])
        updates_val = m.float_list(x_dim1 * x_dim2, -10.0, 1000.0,
                                   'updates_val')
        updates = m.tensor(updates_val, 2, [x_dim1, x_dim2])
    else:
        x_dim1 = m.int_range(0, 20, 'x_dim1')
        x_dim2 = m.int_range(0, 20, 'x_dim2')
        x_dim3 = m.int_range(0, 20, 'x_dim3')
        x_val = m.float_list(x_dim1 * x_dim2 * x_dim3, -10.0, 1000.0, 'x_val')
        x = m.tensor(x_val, 3, [x_dim1, x_dim2, x_dim3])
        updates_val = m.float_list(x_dim1 * x_dim2 * x_dim3, -10.0, 1000.0,
                                   'updates_val')
        updates = m.tensor(updates_val, 3, [x_dim1, x_dim2, x_dim3])

    index_val = m.int_list(x_dim1, 0, 1000, 'index_val')
    index = m.tensor(index_val, 1, [x_dim1], dtype='int32')

    if IGNORE_ERRS:
        try:
            paddle.scatter(x, index, updates, overwrite)
        except IgnoredErrors:
            pass
    else:
        paddle.scatter(x, index, updates, overwrite)


def main():
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
