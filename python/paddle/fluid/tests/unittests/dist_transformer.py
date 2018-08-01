#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import argparse
import time
import math

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
import os
import sys
import transformer_model
import paddle.dataset.wmt16 as wmt16

# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1

WMT16_RECORDIO_FILE = "/tmp/wmt16.recordio"


class ModelHyperParams(object):
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
    n_layer = 6
    # dropout rate used by all dropout layers.
    dropout = 0.1


def prepare_batch_input(insts, src_pad_idx, trg_pad_idx, n_head):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias. Then, convert the numpy
    data to tensors and return a dict mapping names to tensors.
    """

    def __pad_batch_data(insts,
                         pad_idx,
                         is_target=False,
                         return_pos=True,
                         return_attn_bias=True,
                         return_max_len=True):
        """
        Pad the instances to the max sequence length in batch, and generate the
        corresponding position data and attention bias.
        """
        return_list = []
        max_len = max(len(inst) for inst in insts)
        inst_data = np.array(
            [inst + [pad_idx] * (max_len - len(inst)) for inst in insts])
        return_list += [inst_data.astype("int64").reshape([-1, 1])]
        if return_pos:
            inst_pos = np.array([[
                pos_i + 1 if w_i != pad_idx else 0
                for pos_i, w_i in enumerate(inst)
            ] for inst in inst_data])

            return_list += [inst_pos.astype("int64").reshape([-1, 1])]
        if return_attn_bias:
            if is_target:
                # This is used to avoid attention on paddings and subsequent
                # words.
                slf_attn_bias_data = np.ones((inst_data.shape[0], max_len,
                                              max_len))
                slf_attn_bias_data = np.triu(slf_attn_bias_data, 1).reshape(
                    [-1, 1, max_len, max_len])
                slf_attn_bias_data = np.tile(slf_attn_bias_data,
                                             [1, n_head, 1, 1]) * [-1e9]
            else:
                # This is used to avoid attention on paddings.
                slf_attn_bias_data = np.array([[0] * len(inst) + [-1e9] *
                                               (max_len - len(inst))
                                               for inst in insts])
                slf_attn_bias_data = np.tile(
                    slf_attn_bias_data.reshape([-1, 1, 1, max_len]),
                    [1, n_head, max_len, 1])
            return_list += [slf_attn_bias_data.astype("float32")]
        if return_max_len:
            return_list += [max_len]
        return return_list if len(return_list) > 1 else return_list[0]

    src_word, src_pos, src_slf_attn_bias, src_max_len = __pad_batch_data(
        [inst[0] for inst in insts], src_pad_idx, is_target=False)
    trg_word, trg_pos, trg_slf_attn_bias, trg_max_len = __pad_batch_data(
        [inst[1] for inst in insts], trg_pad_idx, is_target=True)
    trg_src_attn_bias = np.tile(src_slf_attn_bias[:, :, ::src_max_len, :],
                                [1, 1, trg_max_len, 1]).astype("float32")
    lbl_word = __pad_batch_data([inst[2] for inst in insts], trg_pad_idx, False,
                                False, False, False)
    lbl_weight = (lbl_word != trg_pad_idx).astype("float32").reshape([-1, 1])

    return [
        src_word, src_pos, trg_word, trg_pos, src_slf_attn_bias,
        trg_slf_attn_bias, trg_src_attn_bias, lbl_word, lbl_weight
    ]


def transformer(use_feed):
    assert not use_feed, "transfomer doesn't support feed yet"
    return transformer_model.transformer(
        ModelHyperParams.src_vocab_size + 1,
        ModelHyperParams.trg_vocab_size + 1, ModelHyperParams.max_length + 1,
        ModelHyperParams.n_layer, ModelHyperParams.n_head,
        ModelHyperParams.d_key, ModelHyperParams.d_value,
        ModelHyperParams.d_model, ModelHyperParams.d_inner_hid,
        ModelHyperParams.dropout, ModelHyperParams.src_pad_idx,
        ModelHyperParams.trg_pad_idx, ModelHyperParams.pos_pad_idx)


def get_model():
    avg_cost = transformer(use_feed=False)
    optimizer = fluid.optimizer.Adam()
    optimizer.minimize(avg_cost)
    return avg_cost


def get_transpiler(trainer_id, main_program, pserver_endpoints, trainers):
    t = fluid.DistributeTranspiler()
    t.transpile(
        trainer_id=trainer_id,
        program=main_program,
        pservers=pserver_endpoints,
        trainers=trainers)
    return t


class DistTransformer2x2(object):
    def run_pserver(self, pserver_endpoints, trainers, current_endpoint,
                    trainer_id):
        get_model()
        t = get_transpiler(trainer_id,
                           fluid.default_main_program(), pserver_endpoints,
                           trainers)
        pserver_prog = t.get_pserver_program(current_endpoint)
        startup_prog = t.get_startup_program(current_endpoint, pserver_prog)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_prog)
        exe.run(pserver_prog)

    def _wait_ps_ready(self, pid):
        retry_times = 20
        while True:
            assert retry_times >= 0, "wait ps ready failed"
            time.sleep(3)
            print("waiting ps ready: ", pid)
            try:
                # the listen_and_serv_op would touch a file which contains the listen port
                # on the /tmp directory until it was ready to process all the RPC call.
                os.stat("/tmp/paddle.%d.port" % pid)
                return
            except os.error:
                retry_times -= 1

    def run_trainer(self, place, endpoints, trainer_id, trainers, is_dist=True):
        avg_cost = get_model()
        if is_dist:
            t = get_transpiler(trainer_id,
                               fluid.default_main_program(), endpoints,
                               trainers)
            trainer_prog = t.get_trainer_program()
        else:
            trainer_prog = fluid.default_main_program()

        startup_exe = fluid.Executor(place)
        startup_exe.run(fluid.default_startup_program())

        strategy = fluid.ExecutionStrategy()
        strategy.num_threads = 1
        strategy.allow_op_delay = False
        exe = fluid.ParallelExecutor(
            True, loss_name=avg_cost.name, exec_strategy=strategy)

        first_loss, = exe.run(fetch_list=[avg_cost.name])
        print(first_loss)
        for i in xrange(5):
            _ = exe.run(fetch_list=[avg_cost.name])
        last_loss, = exe.run(fetch_list=[avg_cost.name])
        print(last_loss)


def main(role="pserver",
         endpoints="127.0.0.1:9123",
         trainer_id=0,
         current_endpoint="127.0.0.1:9123",
         trainers=1,
         is_dist=True):

    reader = paddle.batch(
        wmt16.train(ModelHyperParams.src_vocab_size,
                    ModelHyperParams.trg_vocab_size),
        batch_size=transformer_model.batch_size)

    with fluid.recordio_writer.create_recordio_writer(
            WMT16_RECORDIO_FILE) as writer:
        for batch in reader():
            for tensor in prepare_batch_input(
                    batch, ModelHyperParams.src_pad_idx,
                    ModelHyperParams.trg_pad_idx, ModelHyperParams.n_head):
                t = fluid.LoDTensor()
                t.set(tensor, fluid.CPUPlace())
                writer.append_tensor(t)
            writer.complete_append_tensor()

    model = DistTransformer2x2()
    if role == "pserver":
        model.run_pserver(endpoints, trainers, current_endpoint, trainer_id)
    else:
        p = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        model.run_trainer(p, endpoints, trainer_id, trainers, is_dist)


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print(
            "Usage: python dist_transformer.py [pserver/trainer] [endpoints] [trainer_id] [current_endpoint] [trainers] [is_dist]"
        )
    role = sys.argv[1]
    endpoints = sys.argv[2]
    trainer_id = int(sys.argv[3])
    current_endpoint = sys.argv[4]
    trainers = int(sys.argv[5])
    is_dist = True if sys.argv[6] == "TRUE" else False
    main(
        role=role,
        endpoints=endpoints,
        trainer_id=trainer_id,
        current_endpoint=current_endpoint,
        trainers=trainers,
        is_dist=is_dist)
