"""
Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
This module provides an example for paddle.nn.SimpleRNNCell.
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

    enc_input, en_rank = m.tensor_with_diff_shape(min_val=0.0, max_val=30.0, min_dim=0, max_dim=10, max_rank=4)
    label, la_rank = m.tensor_with_diff_shape(min_val=0.0, max_val=30.0, min_dim=0, max_dim=10, max_rank=4)
    input_size = m.int_range(0, 20, 'input_size')
    hidden_size = m.int_range(0, 20, 'hidden_size')
    activation_list = ['tanh', 'relu']
    activation = activation_list[m.int_range(0, len(activation_list) - 1, 'activation_list')]

    if IGNORE_ERRS:
        try:
            m = paddle.nn.SimpleRNNCell(input_size=input_size, hidden_size=hidden_size, activation=activation,
                                        weight_ih_attr=None, weight_hh_attr=None, bias_ih_attr=None, bias_hh_attr=None)
            decoder = m(enc_input, label)
        except IgnoredErrors:
            pass
    else:
        m = paddle.nn.SimpleRNNCell(input_size=input_size, hidden_size=hidden_size, activation=activation,
                                    weight_ih_attr=None, weight_hh_attr=None, bias_ih_attr=None, bias_hh_attr=None)
        decoder = m(enc_input, label)


def main():
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
