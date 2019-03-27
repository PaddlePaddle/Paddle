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

import paddle.fluid as fluid
import unittest
import numpy as np
import six


class StaticRNNTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.seq_len = 16
        self.other_dims = (32, 16)
        self.reverse = False

    def test_main(self):
        seq_shape = (self.seq_len, -1) + tuple(self.other_dims)

        other_dim_size = np.product(np.array(self.other_dims))

        seq_in = fluid.layers.data(
            shape=seq_shape, dtype='float32', name='seq_in')
        seq_in_init = fluid.layers.data(
            shape=(-1, ) + tuple(self.other_dims),
            dtype='float32',
            name='seq_in_init')

        rnn = fluid.layers.StaticRNN(name='rnn')

        with rnn.step():
            pre_mem = rnn.memory(init=seq_in_init)
            x_t = rnn.step_input(seq_in)

            tmp = x_t
            for _ in six.moves.range(4):
                tmp = fluid.layers.fc(tmp, size=32, act='sigmoid')

            tmp = fluid.layers.fc(tmp, size=other_dim_size, act='relu')

            tmp = fluid.layers.reshape(tmp, shape=seq_in_init.shape)

            tmp = fluid.layers.elementwise_add(pre_mem, tmp, act='tanh')

            rnn.update_memory(pre_mem, tmp)

            rnn.step_output(tmp)

        rnn_output = rnn()

        tmp = rnn_output
        for _ in six.moves.range(4):
            tmp = fluid.layers.fc(tmp, size=32, num_flatten_dims=2, act='relu')

        loss = fluid.layers.mean(tmp)

        optimizer = fluid.optimizer.Adam(learning_rate=1e-3)

        optimizer.minimize(loss)

        exe = fluid.Executor(fluid.CUDAPlace(0))

        exe.run(fluid.default_startup_program())

        actual_seq_shape = (self.seq_len, self.batch_size
                            ) + tuple(self.other_dims)
        actual_seq_init_shape = (self.batch_size, ) + tuple(self.other_dims)

        for idx in six.moves.range(100):
            enable_gc = True if idx % 10 != 0 else False
            fluid.core._set_eager_deletion_mode(0.0 if enable_gc else -1.0, 1.0,
                                                True)
            feed = {
                seq_in.name:
                np.random.random(size=actual_seq_shape).astype('float32'),
                seq_in_init.name:
                np.random.random(size=actual_seq_init_shape).astype('float32')
            }
            fetch_loss, = exe.run(fluid.default_main_program(),
                                  feed=feed,
                                  fetch_list=[loss])


if __name__ == '__main__':
    unittest.main()
