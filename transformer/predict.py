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
from functools import partial

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.io import DataLoader
from paddle.fluid.layers.utils import flatten

from utils.configure import PDConfig
from utils.check import check_gpu, check_version

from model import Input, set_device
from reader import prepare_infer_input, Seq2SeqDataset, Seq2SeqBatchSampler
from transformer import InferTransformer, position_encoding_init


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
    device = set_device("gpu" if args.use_cuda else "cpu")
    fluid.enable_dygraph(device) if args.eager_run else None

    inputs = [
        Input([None, None], "int64", name="src_word"),
        Input([None, None], "int64", name="src_pos"),
        Input([None, args.n_head, None, None],
              "float32",
              name="src_slf_attn_bias"),
        Input([None, args.n_head, None, None],
              "float32",
              name="trg_src_attn_bias"),
    ]

    # define data
    dataset = Seq2SeqDataset(fpattern=args.predict_file,
                             src_vocab_fpath=args.src_vocab_fpath,
                             trg_vocab_fpath=args.trg_vocab_fpath,
                             token_delimiter=args.token_delimiter,
                             start_mark=args.special_token[0],
                             end_mark=args.special_token[1],
                             unk_mark=args.special_token[2])
    args.src_vocab_size, args.trg_vocab_size, args.bos_idx, args.eos_idx, \
        args.unk_idx = dataset.get_vocab_summary()
    trg_idx2word = Seq2SeqDataset.load_dict(dict_path=args.trg_vocab_fpath,
                                            reverse=True)
    batch_sampler = Seq2SeqBatchSampler(dataset=dataset,
                                        use_token_batch=False,
                                        batch_size=args.batch_size,
                                        max_length=args.max_length)
    data_loader = DataLoader(dataset=dataset,
                             batch_sampler=batch_sampler,
                             places=device,
                             feed_list=[x.forward() for x in inputs],
                             collate_fn=partial(prepare_infer_input,
                                                src_pad_idx=args.eos_idx,
                                                n_head=args.n_head),
                             num_workers=0,
                             return_list=True)

    # define model
    transformer = InferTransformer(args.src_vocab_size,
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

    # TODO: use model.predict when support variant length
    f = open(args.output_file, "wb")
    for data in data_loader():
        finished_seq = transformer.test(inputs=flatten(data))[0]
        finished_seq = np.transpose(finished_seq, [0, 2, 1])
        for ins in finished_seq:
            for beam_idx, beam in enumerate(ins):
                if beam_idx >= args.n_best: break
                id_list = post_process_seq(beam, args.bos_idx,
                                           args.eos_idx)
                word_list = [trg_idx2word[id] for id in id_list]
                sequence = b" ".join(word_list) + b"\n"
                f.write(sequence)


if __name__ == "__main__":
    args = PDConfig(yaml_file="./transformer.yaml")
    args.build()
    args.Print()
    check_gpu(args.use_cuda)
    check_version()

    do_predict(args)
