"""
Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
This module provides an example for paddle.nn.SpectralNorm.
"""
import sys
import atheris_no_libfuzzer as atheris
import paddle
import numpy as np
import os
from fuzz_util import Mutator, IgnoredErrors

# Switches for logging and ignoring errors.
LOGGING = os.getenv('PD_FUZZ_LOGGING') == '1'
IGNORE_ERRS = os.getenv('IGNORE_ERRS') == '1'


def TestOneInput(input_bytes):
    m = Mutator(input_bytes, LOGGING)

    weight_shape_len = m.int_range(2, 5, 'weight_shape_len')
    weight_shape = m.int_list(weight_shape_len, 0, 10, 'weight_shape')

    dim = m.int_range(0, 1, 'dim')

    power_iters = m.int_range(0, 100, 'power_iters')
    eps = m.float_range(-10.0, 100.0, 'eps')

    x_dim1 = m.int_range(0, 10, 'x_dim1')
    x_dim2 = m.int_range(0, 10, 'x_dim2')
    x_dim3 = m.int_range(0, 10, 'x_dim3')
    x_dim4 = m.int_range(0, 10, 'x_dim4')

    if dim == 0:
        x_dim1 = weight_shape[dim]
    elif dim == 1:
        x_dim2 = weight_shape[dim]
    x_val = m.float_list(x_dim1 * x_dim2 * x_dim3 * x_dim4, -10.0, 1000.0, 'x_val')
    x = m.tensor(x_val, 4, [x_dim1, x_dim2, x_dim3, x_dim4])

    # Filter invalid inputs. The conditions can be found in `InferShape` and `Compute`.
    # Setting input constraints can improve the fuzzing performance.
    # Although this need some code auditing work.
    x_dims = [x_dim1, x_dim2, x_dim3, x_dim4]
    w = 1
    for i in range(4):
        if i != dim:
            w *= x_dims[i]

    if np.prod(weight_shape) / weight_shape[dim] != w:
        return

    if IGNORE_ERRS:
        try:
            spectral_norm = paddle.nn.SpectralNorm(weight_shape, dim=dim, power_iters=power_iters, eps=eps)
            spectral_norm(x)
        except IgnoredErrors:
            pass
    else:
        spectral_norm = paddle.nn.SpectralNorm(weight_shape, dim=dim, power_iters=power_iters, eps=eps)
        spectral_norm(x)


def main():
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
