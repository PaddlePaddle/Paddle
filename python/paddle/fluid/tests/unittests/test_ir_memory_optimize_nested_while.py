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

import os, sys
import unittest
import numpy
import numpy as np
import time
import math

import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler


# nested while need lod_level=2 data
def nested_while_op(data, label, dict_dim, emb_dim=128, hid_dim=128):
    rnn = fluid.layers.DynamicRNN()
    with rnn.block():
        in_ = rnn.step_input(data)
        sent_emb = fluid.layers.embedding(
            input=in_, size=[dict_dim, emb_dim], dtype='float32')
        input_forward_proj = fluid.layers.fc(input=sent_emb,
                                             size=emb_dim,
                                             act=None,
                                             bias_attr=False)
        forward, _ = fluid.layers.dynamic_lstm(
            input=input_forward_proj, size=hid_dim, use_peepholes=False)

        rnn1 = fluid.layers.DynamicRNN()
        with rnn1.block():
            in_1 = rnn1.step_input(forward)
            out_1 = fluid.layers.fc(input=[in_1], size=hid_dim, act='tanh')
            rnn1.output(out_1)

        last = fluid.layers.sequence_last_step(input=rnn1())
        rnn.output(last)

    last = rnn()
    logits = fluid.layers.fc(input=last, size=1, act=None)
    loss = fluid.layers.sigmoid_cross_entropy_with_logits(x=logits, label=label)
    loss = fluid.layers.mean(loss)
    return loss


class BuildNestedWhileIrMemOptBase(unittest.TestCase):
    def custom_reader(self):
        seq_len, label = [[2, 2]], [0, 1]
        data = []
        for ele in seq_len:
            for j in ele:
                data.append([numpy.random.randint(30) \
                                for _ in range(j)])
        while True:
            yield data, label

    def check_network_convergence(self,
                                  network,
                                  use_cuda=True,
                                  memory_opt=True,
                                  iter=5):
        if use_cuda and not core.is_compiled_with_cuda():
            print('Skip use_cuda=True because Paddle is not compiled with cuda')
            return

        if os.name == 'nt':
            print(
                'Skip use_parallel_executor=True because Paddle comes without parallel support on windows'
            )
            return
        fluid.default_startup_program().random_seed = 100
        fluid.default_main_program().random_seed = 100
        batch_size = 2

        word_dict = [i for i in range(30)]
        train_reader = paddle.batch(self.custom_reader, batch_size=2)

        data = fluid.layers.data(
            name='word', shape=[1], dtype='int64', lod_level=2)
        label = fluid.layers.data(
            name='label', shape=[1], dtype='float32', lod_level=1)
        cost = network(data, label, len(word_dict))
        optimizer = fluid.optimizer.Adam(learning_rate=0.001)
        optimizer.minimize(cost)
        if memory_opt:
            fluid.memory_optimize(fluid.default_main_program())

        # execution
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        feeder = fluid.DataFeeder(feed_list=[data, label], place=place)
        reader = feeder.decorate_reader(train_reader, multi_devices=True)
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        train_cp = compiler.CompiledProgram(fluid.default_main_program())
        train_cp = train_cp.with_data_parallel(loss_name=cost.name)
        fetch_list = [cost.name]

        begin = time.time()
        first_loss, last_loss = None, None
        step_id = 0
        for data in reader():
            ret = exe.run(train_cp, feed=data, fetch_list=fetch_list)
            print(ret)
            step_id += 1
            if step_id == 1:
                first_loss = ret[0]
            if step_id == iter:
                last_loss = ret[0]
                break
        end = time.time()

        print("%.4f Instance per second" % (
            (batch_size * iter) / (end - begin)))

        print(first_loss, last_loss)
        avg_last_loss_val = np.array(last_loss).mean()
        avg_first_loss_val = np.array(first_loss).mean()
        if math.isnan(float(avg_last_loss_val)) or math.isnan(
                float(avg_first_loss_val)):
            sys.exit("got NaN loss, training failed.")
        return first_loss, last_loss


class TestIrMemOptBase(BuildNestedWhileIrMemOptBase):
    def setUp(self):
        self.network = None

    def test_network(self):
        if self.network is None or not core.is_compiled_with_cuda():
            return

        baseline_first_loss, baseline_last_loss = None, None
        for use_cuda in [True]:
            for use_python_mem_opt in [True, False]:
                print(
                    'network: {}, use_cuda: {}, use_python_mem_opt: {}, use_ir_mem_opt : {}'.
                    format(self.network.__name__, use_cuda, use_python_mem_opt,
                           not use_python_mem_opt))
                with fluid.program_guard(fluid.Program(), fluid.Program()):
                    with fluid.scope_guard(core.Scope()):
                        if use_cuda is True and use_python_mem_opt is True:
                            baseline_first_loss, baseline_last_loss = self.check_network_convergence(
                                self.network,
                                use_cuda=use_cuda,
                                memory_opt=use_python_mem_opt)
                        else:
                            cur_first_loss, cur_last_loss = self.check_network_convergence(
                                self.network,
                                use_cuda=use_cuda,
                                memory_opt=use_python_mem_opt)

                            self.assertAlmostEquals(
                                np.mean(baseline_last_loss),
                                np.mean(cur_last_loss),
                                delta=1e-2)
                            self.assertAlmostEquals(
                                np.mean(baseline_first_loss),
                                np.mean(cur_first_loss),
                                delta=1e-2)


class TestNestedWhileIrMemOpt(TestIrMemOptBase):
    def setUp(self):
        self.network = nested_while_op


if __name__ == "__main__":
    unittest.main()
