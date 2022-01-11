"""
Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
This module provides an example for paddle.nn.RNN.
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

    # Prepare attributes for `SimpleRNNCell`.
    input_size = m.int_range(0, 20, 'input_size')
    hidden_size = m.int_range(1, 20, 'hidden_size')
    # Can be `tanh` or `relu`. Defaults to `tanh`.
    activation = m.pick(['tanh', 'relu'], 'activation')

    has_weight_ih_attr = m.bool('has_weight_ih_attr')
    weight_ih_attr = m.param_attr() if has_weight_ih_attr else None

    has_weight_hh_attr = m.bool('has_weight_hh_attr')
    weight_hh_attr = m.param_attr() if has_weight_hh_attr else None

    has_bias_ih_attr = m.bool('has_bias_ih_attr')
    bias_ih_attr = m.param_attr() if has_bias_ih_attr else None

    has_bias_hh_attr = m.bool('has_bias_hh_attr')
    bias_hh_attr = m.param_attr() if has_bias_hh_attr else None

    is_reverse = m.bool('is_reverse')
    time_major = m.bool('time_major')

    x_map = {
        'name': 'x',
        'dtype': 'float32',
        'np_type': 'float32',
        'min_val': -10.0,
        'max_val': 1000.0,
        'dims': {
            'dim1': {
                'min': 0,
                'max': 20,
            },
            'dim2': {
                'min': 0,
                'max': 20,
            }
        }
    }
    x = m.tensor_from_dict(x_map)

    y_map = {
        'name': 'y',
        'dtype': 'float32',
        'np_type': 'float32',
        'min_val': -10.0,
        'max_val': 1000.0,
        'dims': {
            'dim1': {
                'min': 0,
                'max': 20,
            },
            'dim2': {
                'min': 0,
                'max': 20,
            }
        }
    }
    y = m.tensor_from_dict(y_map)

    batch_size = m.int_range(0, 20, 'batch_size')
    sequence_length = m.int_list(batch_size, -10, 100, 'sequence_length')

    if IGNORE_ERRS:
        try:
            # Create a `SimpleRNNCell` for `RNN`.
            cell = paddle.nn.SimpleRNNCell(input_size=input_size, hidden_size=hidden_size, activation=activation,
                                           weight_ih_attr=weight_ih_attr, weight_hh_attr=weight_hh_attr,
                                           bias_ih_attr=bias_ih_attr, bias_hh_attr=bias_hh_attr)
            rnn = paddle.nn.RNN(cell, is_reverse, time_major)
            rnn(x, y, sequence_length)
        except IgnoredErrors:
            pass
    else:
        cell = paddle.nn.SimpleRNNCell(input_size=input_size, hidden_size=hidden_size, activation=activation,
                                       weight_ih_attr=weight_ih_attr, weight_hh_attr=weight_hh_attr,
                                       bias_ih_attr=bias_ih_attr, bias_hh_attr=bias_hh_attr)
        rnn = paddle.nn.RNN(cell, is_reverse, time_major)
        rnn(x, y, sequence_length)


def main():
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
