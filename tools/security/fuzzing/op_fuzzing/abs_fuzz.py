"""
Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
This module provides an example for paddle.abs.
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

    # Prepare mutated inputs
    x_dim1 = m.int_range(0, 20, 'x_dim1')
    x_dim2 = m.int_range(0, 20, 'x_dim2')
    x_dim3 = m.int_range(0, 20, 'x_dim3')
    x_val = m.float_list(x_dim1 * x_dim2 * x_dim3, -10.0, 1000.0, 'x_val')
    x = m.tensor(x_val, 3, [x_dim1, x_dim2, x_dim3])

    # # Generate tensor by dict
    # x_map = {
    #     'name': 'x',
    #     'dtype': 'float32',
    #     'np_type': 'float32',
    #     'min_val': -10.0,
    #     'max_val': 1000.0,
    #     'dims': {
    #         'dim1': {
    #             'min': 0,
    #             'max': 20,
    #         },
    #         'dim2': {
    #             'min': 0,
    #             'max': 20,
    #         }
    #     }
    # }
    # x = m.tensor_from_dict(x_map)

    if IGNORE_ERRS:
        # Run with catching IgnoredErrors.
        try:
            paddle.abs(x)
        except IgnoredErrors:
            pass
    else:
        # This may encounter lots of Error.
        # Some of them is raised by Paddle in sanitizing and should be ignored.
        # This mode can help refining the inputs limitations.
        paddle.abs(x)


def main():
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
