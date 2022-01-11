"""
Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
This module provides an example for paddle.nn.SpectralNorm.
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

    enc_input, rank = m.tensor_with_diff_shape(min_val=0.0, max_val=30.0, min_dim=0, max_dim=10, max_rank=3)

    weight_shape = []
    weight_shape_rank = m.int_range(0, 10, 'weight_shape_rank')
    for i in range(weight_shape_rank):
        weight_shape.append(m.int_range(0, 10, 'weight_shape'))
    dim = m.int_range(0, 10, 'dim')
    power_iters = m.int_range(0, 10, 'power_iters')
    eps = m.float_range(0.0, 10.0, 'eps')

    if IGNORE_ERRS:
        try:
            spectral_norm = paddle.nn.SpectralNorm(weight_shape=weight_shape, dim=dim, power_iters=power_iters, eps=eps)
            decoder = spectral_norm(enc_input)
        except IgnoredErrors:
            pass
    else:
        spectral_norm = paddle.nn.SpectralNorm(weight_shape=weight_shape, dim=dim, power_iters=power_iters, eps=eps)
        decoder = spectral_norm(enc_input)


def main():
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
