"""
Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
This module provides an example for paddle.nn.TransformerEncoderLayer.
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

    enc_input, enc_rank = m.tensor_with_diff_shape(
        min_val=0,
        max_val=30,
        min_dim=0,
        max_dim=10,
        max_rank=3,
        dtype='int32',
        np_type='int32')
    attn_mask, att_rank = m.tensor_with_diff_shape(
        min_val=0,
        max_val=30,
        min_dim=0,
        max_dim=10,
        max_rank=3,
        dtype='int32',
        np_type='int32')
    d_model = m.int_range(0, 30, 'd_model')
    nhead = m.int_range(0, 30, 'nhead')
    dim_feedforward = m.int_range(0, 30, 'dim_feedforward')
    dropout = m.float_range(0.0, 30.0, 'dropout')
    attn_dropout = m.float_range(0.0, 30.0, 'attn_dropout')
    act_dropout = m.float_range(0.0, 30.0, 'act_dropout')
    normalize_before = m.bool('normalize_before')
    bias_attr = m.bool('bias_attr')

    if IGNORE_ERRS:
        try:
            encoder_layer = paddle.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                attn_dropout=attn_dropout,
                act_dropout=act_dropout,
                normalize_before=normalize_before,
                bias_attr=bias_attr)
            output = encoder_layer(enc_input, attn_mask)
        except IgnoredErrors:
            pass
    else:
        encoder_layer = paddle.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            attn_dropout=attn_dropout,
            act_dropout=act_dropout,
            normalize_before=normalize_before,
            bias_attr=bias_attr)
        output = encoder_layer(enc_input, attn_mask)


def main():
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
