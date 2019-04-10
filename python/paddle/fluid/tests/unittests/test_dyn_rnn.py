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

from __future__ import print_function

import paddle.fluid as fluid
import paddle
import unittest
import numpy

from paddle.fluid.layers.control_flow import lod_rank_table
from paddle.fluid.layers.control_flow import max_sequence_len
from paddle.fluid.layers.control_flow import lod_tensor_to_array
from paddle.fluid.layers.control_flow import array_to_lod_tensor
from paddle.fluid.layers.control_flow import shrink_memory
from fake_reader import fake_imdb_reader


class TestDynRNN(unittest.TestCase):
    def setUp(self):
        self.word_dict_len = 5147
        self.BATCH_SIZE = 2
        reader = fake_imdb_reader(self.word_dict_len, self.BATCH_SIZE * 100)
        self.train_data = paddle.batch(reader, batch_size=self.BATCH_SIZE)

        self.word_dict = paddle.dataset.imdb.word_dict()
        self.BATCH_SIZE = 2
        self.train_data = paddle.batch(
            paddle.dataset.imdb.train(self.word_dict),
            batch_size=self.BATCH_SIZE)

    def test_plain_while_op(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()

        with fluid.program_guard(main_program, startup_program):
            sentence = fluid.layers.data(
                name='word', shape=[1], dtype='int64', lod_level=1)
            sent_emb = fluid.layers.embedding(
                input=sentence, size=[self.word_dict_len, 32], dtype='float32')

            label = fluid.layers.data(name='label', shape=[1], dtype='float32')

            rank_table = lod_rank_table(x=sent_emb)

            sent_emb_array = lod_tensor_to_array(x=sent_emb, table=rank_table)

            seq_len = max_sequence_len(rank_table=rank_table)
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
            i.stop_gradient = False

            boot_mem = fluid.layers.fill_constant_batch_size_like(
                input=fluid.layers.array_read(
                    array=sent_emb_array, i=i),
                value=0,
                shape=[-1, 100],
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

                hidden = fluid.layers.fc(input=[mem, ipt], size=100, act='tanh')

                fluid.layers.array_write(x=hidden, i=i, array=out)
                fluid.layers.increment(x=i, in_place=True)
                fluid.layers.array_write(x=hidden, i=i, array=mem_array)
                fluid.layers.less_than(x=i, y=seq_len, cond=cond)

            all_timesteps = array_to_lod_tensor(x=out, table=rank_table)
            last = fluid.layers.sequence_last_step(input=all_timesteps)
            logits = fluid.layers.fc(input=last, size=1, act=None)
            loss = fluid.layers.sigmoid_cross_entropy_with_logits(
                x=logits, label=label)
            loss = fluid.layers.mean(loss)
            sgd = fluid.optimizer.SGD(1e-4)
            sgd.minimize(loss=loss)
        cpu = fluid.CPUPlace()
        exe = fluid.Executor(cpu)
        exe.run(startup_program)
        feeder = fluid.DataFeeder(feed_list=[sentence, label], place=cpu)

        data = next(self.train_data())
        val = exe.run(main_program, feed=feeder.feed(data),
                      fetch_list=[loss])[0]
        self.assertEqual((1, ), val.shape)
        print(val)
        self.assertFalse(numpy.isnan(val))

    def test_train_dyn_rnn(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            sentence = fluid.layers.data(
                name='word', shape=[1], dtype='int64', lod_level=1)
            sent_emb = fluid.layers.embedding(
                input=sentence, size=[self.word_dict_len, 32], dtype='float32')

            rnn = fluid.layers.DynamicRNN()

            with rnn.block():
                in_ = rnn.step_input(sent_emb)
                mem = rnn.memory(shape=[100], dtype='float32')
                out_ = fluid.layers.fc(input=[in_, mem], size=100, act='tanh')
                rnn.update_memory(mem, out_)
                rnn.output(out_)

            last = fluid.layers.sequence_last_step(input=rnn())
            logits = fluid.layers.fc(input=last, size=1, act=None)
            label = fluid.layers.data(name='label', shape=[1], dtype='float32')
            loss = fluid.layers.sigmoid_cross_entropy_with_logits(
                x=logits, label=label)
            loss = fluid.layers.mean(loss)
            sgd = fluid.optimizer.Adam(1e-3)
            sgd.minimize(loss=loss)

        cpu = fluid.CPUPlace()
        exe = fluid.Executor(cpu)
        exe.run(startup_program)
        feeder = fluid.DataFeeder(feed_list=[sentence, label], place=cpu)
        data = next(self.train_data())
        loss_0 = exe.run(main_program,
                         feed=feeder.feed(data),
                         fetch_list=[loss])[0]
        for _ in range(100):
            val = exe.run(main_program,
                          feed=feeder.feed(data),
                          fetch_list=[loss])[0]
        # loss should be small after 100 mini-batch
        self.assertLess(val[0], loss_0[0])

    # this unit test is just used to the two layer nested dyn_rnn.
    def test_train_nested_dyn_rnn(self):
        word_dict = [i for i in range(30)]

        def fake_reader():
            seq_len, label = [[2, 2]], [0, 1]
            data = []
            for ele in seq_len:
                for j in ele:
                    data.append([numpy.random.randint(30) \
                                 for _ in range(j)])

            while True:
                yield data, label

        train_data = paddle.batch(fake_reader, batch_size=2)

        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            sentence = fluid.layers.data(
                name='word', shape=[1], dtype='int64', lod_level=2)
            label = fluid.layers.data(
                name='label', shape=[1], dtype='float32', lod_level=1)

            rnn = fluid.layers.DynamicRNN()
            with rnn.block():
                in_ = rnn.step_input(sentence)
                assert in_.lod_level == 1, "the lod level of in_ should be 1"
                sent_emb = fluid.layers.embedding(
                    input=in_, size=[len(word_dict), 32], dtype='float32')
                out_ = fluid.layers.fc(input=sent_emb, size=100, act='tanh')

                rnn1 = fluid.layers.DynamicRNN()
                with rnn1.block():
                    in_1 = rnn1.step_input(out_)
                    assert in_1.lod_level == 0, "the lod level of in_1 should be 0"
                    out_1 = fluid.layers.fc(input=[in_1], size=100, act='tanh')
                    rnn1.output(out_1)

                last = fluid.layers.sequence_last_step(input=rnn1())
                rnn.output(last)

            last = rnn()
            logits = fluid.layers.fc(input=last, size=1, act=None)
            loss = fluid.layers.sigmoid_cross_entropy_with_logits(
                x=logits, label=label)
            loss = fluid.layers.mean(loss)
            sgd = fluid.optimizer.SGD(1e-3)
            #sgd = fluid.optimizer.Adam(1e-3)
            sgd.minimize(loss=loss)

        cpu = fluid.CPUPlace()
        exe = fluid.Executor(cpu)
        exe.run(startup_program)
        feeder = fluid.DataFeeder(feed_list=[sentence, label], place=cpu)
        data = next(train_data())
        val = exe.run(main_program, feed=feeder.feed(data),
                      fetch_list=[loss])[0]

        for _ in range(100):
            val = exe.run(main_program,
                          feed=feeder.feed(data),
                          fetch_list=[loss])[0]
            print(val)

    # this unit test is just used to the two layer nested dyn_rnn.
    def test_train_nested_dyn_rnn2(self):
        word_dict = [i for i in range(30)]

        def fake_reader():
            seq_len, label = [[2, 2]], [0, 1]
            data = []
            for ele in seq_len:
                for j in ele:
                    data.append([numpy.random.randint(30) \
                                 for _ in range(j)])

            while True:
                yield data, label

        train_data = paddle.batch(fake_reader, batch_size=2)
        hidden_size = 32
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            sentence = fluid.layers.data(
                name='word', shape=[1], dtype='int64', lod_level=2)
            label = fluid.layers.data(
                name='label', shape=[1], dtype='float32', lod_level=1)

            rnn = fluid.layers.DynamicRNN()
            with rnn.block():
                in_ = rnn.step_input(sentence)
                sent_emb = fluid.layers.embedding(
                    input=in_,
                    size=[len(word_dict), hidden_size],
                    dtype='float32')
                input_forward_proj = fluid.layers.fc(input=sent_emb,
                                                     size=hidden_size * 4,
                                                     act=None,
                                                     bias_attr=False)
                forward, _ = fluid.layers.dynamic_lstm(
                    input=input_forward_proj,
                    size=hidden_size * 4,
                    use_peepholes=False)

                rnn1 = fluid.layers.DynamicRNN()
                with rnn1.block():
                    in_1 = rnn1.step_input(forward)
                    out_1 = fluid.layers.fc(input=[in_1], size=100, act='tanh')
                    rnn1.output(out_1)

                last = fluid.layers.sequence_last_step(input=rnn1())
                rnn.output(last)

            last = rnn()
            logits = fluid.layers.fc(input=last, size=1, act=None)
            loss = fluid.layers.sigmoid_cross_entropy_with_logits(
                x=logits, label=label)
            loss = fluid.layers.mean(loss)
            sgd = fluid.optimizer.SGD(1e-3)
            #sgd = fluid.optimizer.Adam(1e-3)
            sgd.minimize(loss=loss)

        cpu = fluid.CPUPlace()
        exe = fluid.Executor(cpu)
        exe.run(startup_program)
        feeder = fluid.DataFeeder(feed_list=[sentence, label], place=cpu)
        data = next(train_data())
        val = exe.run(main_program, feed=feeder.feed(data),
                      fetch_list=[loss])[0]

        for _ in range(100):
            val = exe.run(main_program,
                          feed=feeder.feed(data),
                          fetch_list=[loss])[0]


if __name__ == '__main__':
    unittest.main()
