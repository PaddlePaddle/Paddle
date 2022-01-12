"""
Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
This module provides an example for paddle.static.auc.
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

    input_tensor, rank = m.tensor_with_diff_shape(
        min_val=0.0, max_val=30.0, min_dim=0, max_dim=10, max_rank=4)
    label, la_rank = m.tensor_with_diff_shape(
        min_val=0.0, max_val=30.0, min_dim=0, max_dim=10, max_rank=4)
    num_thresholds = m.int_range(0, 20, 'num_thresholds')
    topk = m.int_range(0, 20, 'topk')
    slide_steps = m.int_range(0, 20, 'slide_steps')

    if IGNORE_ERRS:
        try:
            paddle.static.auc(input_tensor,
                              label,
                              num_thresholds=num_thresholds,
                              topk=topk,
                              slide_steps=slide_steps)
        except IgnoredErrors:
            pass
    else:
        paddle.static.auc(input_tensor,
                          label,
                          num_thresholds=num_thresholds,
                          topk=topk,
                          slide_steps=slide_steps)


def main():
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
