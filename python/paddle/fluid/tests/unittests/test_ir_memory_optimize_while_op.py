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

from __future__ import print_function

import paddle.fluid as fluid
import unittest
from ir_memory_optimize_net_base import TestIrMemOptBase

from paddle.fluid.layers.control_flow import lod_rank_table
from paddle.fluid.layers.control_flow import max_sequence_len
from paddle.fluid.layers.control_flow import lod_tensor_to_array
from paddle.fluid.layers.control_flow import array_to_lod_tensor
from paddle.fluid.layers.control_flow import shrink_memory


def plain_while_op(data, label, dict_dim, emb_dim=128, hid_dim=128):
    label = fluid.layers.cast(label, dtype="float32")
    sent_emb = fluid.layers.embedding(
        input=data, size=[dict_dim, emb_dim], dtype='float32')
    rank_table = lod_rank_table(x=sent_emb)
    sent_emb_array = lod_tensor_to_array(x=sent_emb, table=rank_table)

    seq_len = max_sequence_len(rank_table=rank_table)
    i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
    i.stop_gradient = False

    boot_mem = fluid.layers.fill_constant_batch_size_like(
        input=fluid.layers.array_read(
            array=sent_emb_array, i=i),
        value=0,
        shape=[-1, hid_dim],
        dtype='float32')
    boot_mem.stop_gradient = False

    mem_array = fluid.layers.array_write(x=boot_mem, i=i)

    cond = fluid.layers.less_than(x=i, y=seq_len)
    cond.stop_gradient = False
    while_op = fluid.layers.While(cond=cond)
    out = fluid.layers.create_array(dtype='float32')

    with while_op.block():
        mem = fluid.layers.array_read(array=mem_array, i=i)
        ipt = fluid.layers.array_read(array=sent_emb_array, i=i)

        mem = shrink_memory(x=mem, i=i, table=rank_table)

        hidden = fluid.layers.fc(input=[mem, ipt], size=hid_dim, act='tanh')

        fluid.layers.array_write(x=hidden, i=i, array=out)
        fluid.layers.increment(x=i, in_place=True)
        fluid.layers.array_write(x=hidden, i=i, array=mem_array)
        fluid.layers.less_than(x=i, y=seq_len, cond=cond)

    all_timesteps = array_to_lod_tensor(x=out, table=rank_table)
    last = fluid.layers.sequence_last_step(input=all_timesteps)
    logits = fluid.layers.fc(input=last, size=1, act=None)
    loss = fluid.layers.sigmoid_cross_entropy_with_logits(x=logits, label=label)
    loss = fluid.layers.mean(loss)
    return loss


def dynamic_rnn_with_while(data, label, dict_dim, emb_dim=128, hid_dim=128):
    label = fluid.layers.cast(label, dtype="float32")
    sent_emb = fluid.layers.embedding(
        input=data, size=[dict_dim, emb_dim], dtype='float32')
    rnn = fluid.layers.DynamicRNN()
    with rnn.block():
        in_ = rnn.step_input(sent_emb)
        mem = rnn.memory(shape=[hid_dim], dtype='float32')
        out_ = fluid.layers.fc(input=[in_, mem], size=hid_dim, act='tanh')
        rnn.update_memory(mem, out_)
        rnn.output(out_)

    last = fluid.layers.sequence_last_step(input=rnn())
    logits = fluid.layers.fc(input=last, size=1, act=None)
    label = fluid.layers.data(name='label', shape=[1], dtype='float32')
    loss = fluid.layers.sigmoid_cross_entropy_with_logits(x=logits, label=label)
    loss = fluid.layers.mean(loss)
    return loss


class TestIrMemOptWhile(TestIrMemOptBase):
    def setUp(self):
        self.network = plain_while_op


class TestIrMemOptDynamicRNN(TestIrMemOptBase):
    def setUp(self):
        self.network = dynamic_rnn_with_while


if __name__ == "__main__":
    unittest.main()
