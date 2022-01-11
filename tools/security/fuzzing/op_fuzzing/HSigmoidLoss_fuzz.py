"""
Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
This module provides an example for paddle.nn.HSigmoidLoss.
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

    feature_size = m.int_range(0, 100, 'feature_size')
    num_classes = m.int_range(2, 100, 'num_classes')

    weight_attr = m.param_attr()
    bias_type = m.pick(['ParamAttr', 'bool'], 'bias_type')
    if bias_type == 'ParamAttr':
        bias_attr = m.param_attr()
    else:
        bias_attr = m.bool('bias_attr')

    is_custom = m.bool('is_custom')
    is_sparse = m.bool('is_sparse')

    # Shape is [N, D]
    x_dim1 = m.int_range(0, 20, 'x_dim1')
    x_dim2 = m.int_range(0, 20, 'x_dim2')
    x_val = m.float_list(x_dim1 * x_dim2, -10.0, 1000.0, 'x_val')
    x = m.tensor(x_val, 2, [x_dim1, x_dim2])

    label_len = m.int_range(0, 10, 'label_len')
    label = m.int_list(label_len, 0, 10, 'label')
    label = paddle.to_tensor(label)

    if IGNORE_ERRS:
        try:
            hsl = paddle.nn.HSigmoidLoss(feature_size=feature_size, num_classes=num_classes, weight_attr=weight_attr,
                                         bias_attr=bias_attr, is_custom=is_custom, is_sparse=is_sparse)
            hsl(x, label)
        except IgnoredErrors:
            pass
    else:
        hsl = paddle.nn.HSigmoidLoss(feature_size=feature_size, num_classes=num_classes, weight_attr=weight_attr,
                                     bias_attr=bias_attr, is_custom=is_custom, is_sparse=is_sparse)
        hsl(x, label)


def main():
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
