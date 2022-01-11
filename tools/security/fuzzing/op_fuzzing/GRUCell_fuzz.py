"""
Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
This module provides an example for paddle.nn.GRUCell.
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
    x_val = m.float_list(x_dim1 * x_dim2 * x_dim3, -10.0, 1000.0, 'x_val')
    x = m.tensor(x_val, 3, [x_dim1, x_dim2, x_dim3])

    y_dim1 = m.int_range(0, 20, 'y_dim1')
    y_dim2 = m.int_range(0, 20, 'y_dim2')
    y_dim3 = m.int_range(0, 20, 'y_dim3')
    y_val = m.float_list(y_dim1 * y_dim2 * y_dim3, -10.0, 1000.0, 'y_val')
    y = m.tensor(y_val, 3, [y_dim1, y_dim2, y_dim3])

    input_size = m.int_range(-1, 100, 'input_size')
    hidden_size = m.int_range(-1, 100, 'hidden_size')
    weight_ih_attr = m.param_attr()
    weight_hh_attr = m.param_attr()
    bias_ih_attr = m.param_attr()
    bias_hh_attr = m.param_attr()

    if IGNORE_ERRS:
        try:
            cell = paddle.nn.GRUCell(input_size=input_size, hidden_size=hidden_size, weight_ih_attr=weight_ih_attr,
                                     weight_hh_attr=weight_hh_attr, bias_ih_attr=bias_ih_attr,
                                     bias_hh_attr=bias_hh_attr)
            cell(x, y)
        except IgnoredErrors:
            pass
    else:
        cell = paddle.nn.GRUCell(input_size=input_size, hidden_size=hidden_size, weight_ih_attr=weight_ih_attr,
                                 weight_hh_attr=weight_hh_attr, bias_ih_attr=bias_ih_attr, bias_hh_attr=bias_hh_attr)
        cell(x, y)


def main():
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
