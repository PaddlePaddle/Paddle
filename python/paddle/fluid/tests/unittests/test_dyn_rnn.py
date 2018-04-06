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

import paddle.fluid as fluid
import paddle
import unittest
import numpy


class TestDynRNN(unittest.TestCase):
    def setUp(self):
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
                input=sentence, size=[len(self.word_dict), 32], dtype='float32')

            label = fluid.layers.data(name='label', shape=[1], dtype='float32')

            rank_table = fluid.layers.lod_rank_table(x=sent_emb)

            sent_emb_array = fluid.layers.lod_tensor_to_array(
                x=sent_emb, table=rank_table)

            seq_len = fluid.layers.max_sequence_len(rank_table=rank_table)
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

                mem = fluid.layers.shrink_memory(x=mem, i=i, table=rank_table)

                hidden = fluid.layers.fc(input=[mem, ipt], size=100, act='tanh')

                fluid.layers.array_write(x=hidden, i=i, array=out)
                fluid.layers.increment(x=i, in_place=True)
                fluid.layers.array_write(x=hidden, i=i, array=mem_array)
                fluid.layers.less_than(x=i, y=seq_len, cond=cond)

            all_timesteps = fluid.layers.array_to_lod_tensor(
                x=out, table=rank_table)
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
                input=sentence, size=[len(self.word_dict), 32], dtype='float32')

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
        for _ in xrange(100):
            val = exe.run(main_program,
                          feed=feeder.feed(data),
                          fetch_list=[loss])[0]
        # loss should be small after 100 mini-batch
        self.assertLess(val[0], loss_0[0])


if __name__ == '__main__':
    unittest.main()
