# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import transformer_model
import numpy as np
from parallel_executor_test_base import TestParallelExecutorBase, DeviceType
import unittest
import paddle
import paddle.fluid.core as core
import paddle.dataset.wmt16 as wmt16
import os
from feed_data_reader import FeedDataReader

os.environ['CPU_NUM'] = str(4)


class ModelHyperParams:
    # Dictionary size for source and target language. This model directly uses
    # paddle.dataset.wmt16 in which <bos>, <eos> and <unk> token has
    # alreay been added, but the <pad> token is not added. Transformer requires
    # sequences in a mini-batch are padded to have the same length. A <pad> token is
    # added into the original dictionary in paddle.dateset.wmt16.

    # size of source word dictionary.
    src_vocab_size = 10000
    # index for <pad> token in source language.
    src_pad_idx = src_vocab_size

    # size of target word dictionay
    trg_vocab_size = 10000
    # index for <pad> token in target language.
    trg_pad_idx = trg_vocab_size

    # position value corresponding to the <pad> token.
    pos_pad_idx = 0

    # max length of sequences. It should plus 1 to include position
    # padding token for position encoding.
    max_length = 50

    # the dimension for word embeddings, which is also the last dimension of
    # the input and output of multi-head attention, position-wise feed-forward
    # networks, encoder and decoder.

    d_model = 512
    # size of the hidden layer in position-wise feed-forward networks.
    d_inner_hid = 1024
    # the dimension that keys are projected to for dot-product attention.
    d_key = 64
    # the dimension that values are projected to for dot-product attention.
    d_value = 64
    # number of head used in multi-head attention.
    n_head = 8
    # number of sub-layers to be stacked in the encoder and decoder.
    # NOTE(zcd): the origin number of layer is 6, to make this unit test faster,
    # we should reduce the layer number to 4.
    n_layer = 4
    # dropout rate used by all dropout layers.
    dropout = 0.1


def prepare_batch_input(insts, src_pad_idx, trg_pad_idx, n_head):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias. Then, convert the numpy
    data to tensors and return a dict mapping names to tensors.
    """

    def __pad_batch_data(
        insts,
        pad_idx,
        is_target=False,
        return_pos=True,
        return_attn_bias=True,
        return_max_len=True,
    ):
        """
        Pad the instances to the max sequence length in batch, and generate the
        corresponding position data and attention bias.
        """
        return_list = []
        max_len = max(len(inst) for inst in insts)
        inst_data = np.array(
            [inst + [pad_idx] * (max_len - len(inst)) for inst in insts]
        )
        return_list += [inst_data.astype("int64").reshape([-1, 1])]
        if return_pos:
            inst_pos = np.array(
                [
                    [
                        pos_i + 1 if w_i != pad_idx else 0
                        for pos_i, w_i in enumerate(inst)
                    ]
                    for inst in inst_data
                ]
            )

            return_list += [inst_pos.astype("int64").reshape([-1, 1])]
        if return_attn_bias:
            if is_target:
                # This is used to avoid attention on paddings and subsequent
                # words.
                slf_attn_bias_data = np.ones(
                    (inst_data.shape[0], max_len, max_len)
                )
                slf_attn_bias_data = np.triu(slf_attn_bias_data, 1).reshape(
                    [-1, 1, max_len, max_len]
                )
                slf_attn_bias_data = np.tile(
                    slf_attn_bias_data, [1, n_head, 1, 1]
                ) * [-1e9]
            else:
                # This is used to avoid attention on paddings.
                slf_attn_bias_data = np.array(
                    [
                        [0] * len(inst) + [-1e9] * (max_len - len(inst))
                        for inst in insts
                    ]
                )
                slf_attn_bias_data = np.tile(
                    slf_attn_bias_data.reshape([-1, 1, 1, max_len]),
                    [1, n_head, max_len, 1],
                )
            return_list += [slf_attn_bias_data.astype("float32")]
        if return_max_len:
            return_list += [max_len]
        return return_list if len(return_list) > 1 else return_list[0]

    src_word, src_pos, src_slf_attn_bias, src_max_len = __pad_batch_data(
        [inst[0] for inst in insts], src_pad_idx, is_target=False
    )
    trg_word, trg_pos, trg_slf_attn_bias, trg_max_len = __pad_batch_data(
        [inst[1] for inst in insts], trg_pad_idx, is_target=True
    )
    trg_src_attn_bias = np.tile(
        src_slf_attn_bias[:, :, ::src_max_len, :], [1, 1, trg_max_len, 1]
    ).astype("float32")
    lbl_word = __pad_batch_data(
        [inst[2] for inst in insts], trg_pad_idx, False, False, False, False
    )
    lbl_weight = (lbl_word != trg_pad_idx).astype("float32").reshape([-1, 1])

    return [
        src_word,
        src_pos,
        trg_word,
        trg_pos,
        src_slf_attn_bias,
        trg_slf_attn_bias,
        trg_src_attn_bias,
        lbl_word,
        lbl_weight,
    ]


feed_data_reader = None


def transformer(use_feed):
    assert not use_feed, "transfomer doesn't support feed yet"
    return transformer_model.transformer(
        ModelHyperParams.src_vocab_size + 1,
        ModelHyperParams.trg_vocab_size + 1,
        ModelHyperParams.max_length + 1,
        ModelHyperParams.n_layer,
        ModelHyperParams.n_head,
        ModelHyperParams.d_key,
        ModelHyperParams.d_value,
        ModelHyperParams.d_model,
        ModelHyperParams.d_inner_hid,
        ModelHyperParams.dropout,
        ModelHyperParams.src_pad_idx,
        ModelHyperParams.trg_pad_idx,
        ModelHyperParams.pos_pad_idx,
    )


def get_feed_data_reader():
    global feed_data_reader
    if feed_data_reader is not None:
        return feed_data_reader

    reader = paddle.batch(
        wmt16.train(
            ModelHyperParams.src_vocab_size, ModelHyperParams.trg_vocab_size
        ),
        batch_size=transformer_model.batch_size,
    )
    all_batch_tensors = []
    for batch in reader():
        tensors = []
        for tensor in prepare_batch_input(
            batch,
            ModelHyperParams.src_pad_idx,
            ModelHyperParams.trg_pad_idx,
            ModelHyperParams.n_head,
        ):
            tensors.append(np.array(tensor))
        all_batch_tensors.append(tensors)

    def __reader__():
        for t in all_batch_tensors:
            yield t

    feed_data_reader = FeedDataReader(
        feed_list=transformer_model.build_inputs(
            ModelHyperParams.max_length + 1, ModelHyperParams.n_head
        ),
        reader=__reader__,
    )

    return feed_data_reader


class TestTransformer(TestParallelExecutorBase):
    def test_main(self):
        if core.is_compiled_with_cuda():
            self.check_network_convergence(
                transformer,
                use_device=DeviceType.CUDA,
                feed_data_reader=get_feed_data_reader(),
            )
            self.check_network_convergence(
                transformer,
                use_device=DeviceType.CUDA,
                enable_sequential_execution=True,
                feed_data_reader=get_feed_data_reader(),
            )
        self.check_network_convergence(
            transformer,
            use_device=DeviceType.CPU,
            iter=2,
            feed_data_reader=get_feed_data_reader(),
        )


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
