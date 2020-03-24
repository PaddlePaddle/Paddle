# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import six
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import contextlib

import numpy as np
import paddle
import paddle.fluid as fluid

from utils.configure import PDConfig
from utils.check import check_gpu, check_version

# include task-specific libs
import reader
from transformer import InferTransformer, position_encoding_init
from model import Input


def post_process_seq(seq, bos_idx, eos_idx, output_bos=False,
                     output_eos=False):
    """
    Post-process the decoded sequence.
    """
    eos_pos = len(seq) - 1
    for i, idx in enumerate(seq):
        if idx == eos_idx:
            eos_pos = i
            break
    seq = [
        idx for idx in seq[:eos_pos + 1]
        if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)
    ]
    return seq


def do_predict(args):
    @contextlib.contextmanager
    def null_guard():
        yield

    guard = fluid.dygraph.guard() if args.eager_run else null_guard()

    # define the data generator
    processor = reader.DataProcessor(
        fpattern=args.predict_file,
        src_vocab_fpath=args.src_vocab_fpath,
        trg_vocab_fpath=args.trg_vocab_fpath,
        token_delimiter=args.token_delimiter,
        use_token_batch=False,
        batch_size=args.batch_size,
        device_count=1,
        pool_size=args.pool_size,
        sort_type=reader.SortType.NONE,
        shuffle=False,
        shuffle_batch=False,
        start_mark=args.special_token[0],
        end_mark=args.special_token[1],
        unk_mark=args.special_token[2],
        max_length=args.max_length,
        n_head=args.n_head)
    batch_generator = processor.data_generator(phase="predict")
    args.src_vocab_size, args.trg_vocab_size, args.bos_idx, args.eos_idx, \
        args.unk_idx = processor.get_vocab_summary()
    trg_idx2word = reader.DataProcessor.load_dict(
        dict_path=args.trg_vocab_fpath, reverse=True)

    with guard:
        # define data loader
        test_loader = batch_generator

        # define model
        inputs = [
            Input(
                [None, None], "int64", name="src_word"),
            Input(
                [None, None], "int64", name="src_pos"),
            Input(
                [None, args.n_head, None, None],
                "float32",
                name="src_slf_attn_bias"),
            Input(
                [None, args.n_head, None, None],
                "float32",
                name="trg_src_attn_bias"),
        ]
        transformer = InferTransformer(
            args.src_vocab_size,
            args.trg_vocab_size,
            args.max_length + 1,
            args.n_layer,
            args.n_head,
            args.d_key,
            args.d_value,
            args.d_model,
            args.d_inner_hid,
            args.prepostprocess_dropout,
            args.attention_dropout,
            args.relu_dropout,
            args.preprocess_cmd,
            args.postprocess_cmd,
            args.weight_sharing,
            args.bos_idx,
            args.eos_idx,
            beam_size=args.beam_size,
            max_out_len=args.max_out_len)
        transformer.prepare(inputs=inputs)

        # load the trained model
        assert args.init_from_params, (
            "Please set init_from_params to load the infer model.")
        transformer.load(os.path.join(args.init_from_params, "transformer"))

        f = open(args.output_file, "wb")
        for input_data in test_loader():
            (src_word, src_pos, src_slf_attn_bias, trg_word,
             trg_src_attn_bias) = input_data
            finished_seq = transformer.test(inputs=(
                src_word, src_pos, src_slf_attn_bias, trg_src_attn_bias))[0]
            finished_seq = np.transpose(finished_seq, [0, 2, 1])
            for ins in finished_seq:
                for beam_idx, beam in enumerate(ins):
                    if beam_idx >= args.n_best: break
                    id_list = post_process_seq(beam, args.bos_idx,
                                               args.eos_idx)
                    word_list = [trg_idx2word[id] for id in id_list]
                    sequence = b" ".join(word_list) + b"\n"
                    f.write(sequence)
            break


if __name__ == "__main__":
    args = PDConfig(yaml_file="./transformer.yaml")
    args.build()
    args.Print()
    check_gpu(args.use_cuda)
    check_version()

    do_predict(args)
