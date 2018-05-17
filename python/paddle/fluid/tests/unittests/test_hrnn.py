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
from paddle.fluid import debugger

class TestNestedRNN(unittest.TestCase):
    def setUp(self):
        self.dict_dim = 20
        self.word_dim = 1
        self.hidden_dim = 1
        self.label_dim = 3
        #self.data = [
        #    [[[1, 3, 2], [4, 5, 2]], 0],
        #    [[[0, 0], [0, 2], [2, 5], [0, 1, 2]], 1],
        #]
        self.data = [
            [[[1, 2], [4]], 0],
            [[[11], [15]], 1],
        ]

    def rnn_data(self):
        for d in self.data:
            seq = []
            for subseq in d[0]:
                seq += subseq
            yield seq, d[1]

    def hrnn_data(self):
        for d in self.data:
            yield d

    def rnn(self):
        data = fluid.layers.data(
                        name='word', shape=[1], dtype='int64', lod_level=1)
        emb = fluid.layers.embedding(
            input=data, size=[self.dict_dim, self.word_dim])

        rnn = fluid.layers.DynamicRNN()

        with rnn.block():
            y = rnn.step_input(emb)
            fluid.layers.Print(y, message="y")
            mem = rnn.memory(shape=[self.hidden_dim])
            out = fluid.layers.fc(input=[y, mem],
                                  size=self.hidden_dim,
                                  act='tanh')
            rnn.update_memory(mem, out)
            rnn.output(out)

        rep = fluid.layers.sequence_last_step(input=rnn())
        prob = fluid.layers.fc(input=rep, size=self.label_dim, act='softmax')

        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

        loss = fluid.layers.cross_entropy(prob, label)
        loss = fluid.layers.mean(loss)
        sgd = fluid.optimizer.Adam(1e-3)
        sgd.minimize(loss=loss)
        return [data, label], [loss]

    def hrnn(self):
        data = fluid.layers.data(
                        name='word', shape=[1], dtype='int64', lod_level=2)
        fluid.layers.Print(data, message="raw_data")

        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

        emb = fluid.layers.embedding(
            input=data, size=[self.dict_dim, self.word_dim])

        rnn = fluid.layers.DynamicRNN()
        rnn_inner = fluid.layers.DynamicRNN()

        fluid.layers.Print(emb, print_phase='forward', message='emb')
        with rnn.block():
            y = rnn.step_input(emb)
            mem = rnn.memory(shape=[self.hidden_dim], prefix='outer_')
            fluid.layers.Print(y, print_phase='forward', message='y')
            inner_flag = False # change this to True to insert the inner-RNN
            if inner_flag:
                # rnn_inner = fluid.layers.DynamicRNN()
                with rnn_inner.block():
                    y_inner = rnn_inner.step_input(y)
                    mem_inner = rnn_inner.memory(init=mem, shape=[self.hidden_dim], prefix='inner_')
                    out_inner = fluid.layers.fc(input=[y_inner, mem_inner],
                                  size=self.hidden_dim,
                                  act='tanh')
                    rnn_inner.update_memory(mem_inner, out_inner)
                    rnn_inner.output(out_inner)
                y_inner_out = rnn_inner()
            else:
                y_inner_out = y

            fluid.layers.Print(y_inner_out, print_phase='forward', message='y_inner_out')
            y = fluid.layers.sequence_last_step(input=y_inner_out)
            # will have linking error if not feeding mem into fc
            out = fluid.layers.fc(input=[y, mem],
                                  size=self.hidden_dim,
                                  act='tanh')
            rnn.update_memory(mem, out)
            rnn.output(out)
        out = rnn()
        rep = fluid.layers.sequence_last_step(input=out)
        prob = fluid.layers.fc(input=rep, size=self.label_dim, act='softmax')
        loss = fluid.layers.cross_entropy(prob, label)
        loss = fluid.layers.mean(loss)
        fluid.layers.Print(loss, print_phase='forward', message='loss')
        sgd = fluid.optimizer.Adam(1e-3)
        sgd.minimize(loss=loss)
        return [data, label], [loss]

    def test_hrnn(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()

        with fluid.program_guard(main_program, startup_program):
            inputs, outputs = self.hrnn()
        # print(main_program)

        cpu = fluid.CPUPlace()
        exe = fluid.Executor(cpu)
        exe.run(startup_program)

        feeder = fluid.DataFeeder(feed_list=inputs, place=cpu)
        dataset = paddle.batch(self.hrnn_data, batch_size=2)
        for data in dataset():
            # GRAD, tmp
            loss_np = exe.run(main_program,
                              feed=feeder.feed(data),
                              fetch_list=outputs)[0]

    def test_rnn(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()

        with fluid.program_guard(main_program, startup_program):
            inputs, outputs = self.rnn()

        cpu = fluid.CPUPlace()
        exe = fluid.Executor(cpu)
        exe.run(startup_program)
        feeder = fluid.DataFeeder(feed_list=inputs, place=cpu)
        dataset = paddle.batch(self.rnn_data, batch_size=2)
        for data in dataset():
            loss_np = exe.run(main_program,
                              feed=feeder.feed(data),
                              fetch_list=outputs)[0]

if __name__ == '__main__':
    unittest.main()
