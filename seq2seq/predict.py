# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import io
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
from functools import partial

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.layers.utils import flatten
from paddle.fluid.io import DataLoader

from model import Input, set_device
from args import parse_args
from seq2seq_base import BaseInferModel
from seq2seq_attn import AttentionInferModel, AttentionGreedyInferModel
from reader import Seq2SeqDataset, Seq2SeqBatchSampler, SortType, prepare_infer_input


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
    device = set_device("gpu" if args.use_gpu else "cpu")
    fluid.enable_dygraph(device) if args.eager_run else None

    # define model
    inputs = [
        Input(
            [None, None], "int64", name="src_word"),
        Input(
            [None], "int64", name="src_length"),
    ]

    # def dataloader
    dataset = Seq2SeqDataset(
        fpattern=args.infer_file,
        src_vocab_fpath=args.vocab_prefix + "." + args.src_lang,
        trg_vocab_fpath=args.vocab_prefix + "." + args.tar_lang,
        token_delimiter=None,
        start_mark="<s>",
        end_mark="</s>",
        unk_mark="<unk>")
    trg_idx2word = Seq2SeqDataset.load_dict(
        dict_path=args.vocab_prefix + "." + args.tar_lang, reverse=True)
    (args.src_vocab_size, args.trg_vocab_size, bos_id, eos_id,
     unk_id) = dataset.get_vocab_summary()
    batch_sampler = Seq2SeqBatchSampler(
        dataset=dataset, use_token_batch=False, batch_size=args.batch_size)
    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        places=device,
        feed_list=None
        if fluid.in_dygraph_mode() else [x.forward() for x in inputs],
        collate_fn=partial(
            prepare_infer_input, bos_id=bos_id, eos_id=eos_id, pad_id=eos_id),
        num_workers=0,
        return_list=True)

    # model_maker = AttentionInferModel if args.attention else BaseInferModel
    model_maker = AttentionGreedyInferModel if args.attention else BaseInferModel
    model = model_maker(
        args.src_vocab_size,
        args.tar_vocab_size,
        args.hidden_size,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        bos_id=bos_id,
        eos_id=eos_id,
        beam_size=args.beam_size,
        max_out_len=256)

    model.prepare(inputs=inputs)

    # load the trained model
    assert args.reload_model, (
        "Please set reload_model to load the infer model.")
    model.load(args.reload_model)

    # TODO(guosheng): use model.predict when support variant length
    with io.open(args.infer_output_file, 'w', encoding='utf-8') as f:
        for data in data_loader():
            finished_seq = model.test(inputs=flatten(data))[0]
            finished_seq = finished_seq[:, :, np.newaxis] if len(
                finished_seq.shape == 2) else finished_seq
            finished_seq = np.transpose(finished_seq, [0, 2, 1])
            for ins in finished_seq:
                for beam_idx, beam in enumerate(ins):
                    id_list = post_process_seq(beam, bos_id, eos_id)
                    word_list = [trg_idx2word[id] for id in id_list]
                    sequence = " ".join(word_list) + "\n"
                    f.write(sequence)
                    break


if __name__ == "__main__":
    args = parse_args()
    do_predict(args)
