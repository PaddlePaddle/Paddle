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
import math
import unittest
import numpy
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
from paddle.fluid.initializer import init_on_cpu


def cosine_decay(learning_rate, step_each_epoch, epochs=120):
    """Applies cosine decay to the learning rate.
    lr = 0.05 * (math.cos(epoch * (math.pi / 120)) + 1)
    """
    global_step = _decay_step_counter()
    epoch = fluid.layers.floor(global_step / step_each_epoch)
    lr = learning_rate / 2.
    decayed_lr = lr * (fluid.layers.cos(epoch * (math.pi / epochs)) + 1)
    return decayed_lr


def simple_fc():
    img = fluid.layers.data(name='img', shape=[784])
    hidden = img
    for i in xrange(2):
        hidden = fluid.layers.fc(hidden, size=200, act='sigmoid')
        hidden = fluid.layers.dropout(hidden, dropout_prob=0.1, seed=1)
        # hidden = fluid.layers.batch_norm(
        #     hidden,
        #     moving_mean_name='moving_mean.{0}'.format(i),
        #     moving_variance_name='moving_var.{0}'.format(i))
    loss = hidden
    loss = fluid.layers.mean(loss)
    adam = fluid.optimizer.Adam(
        learning_rate=cosine_decay(
            1e-2, step_each_epoch=100),
        regularization=fluid.regularizer.L2Decay(0.5))
    adam.minimize(loss)
    return loss


class FCDataRandom(object):
    def __init__(self, batch_size=64):
        self.random_state = numpy.random.RandomState(seed=1)
        self.batch_size = batch_size

    def next(self):
        return {
            'img': self.random_state.uniform(
                low=-1, high=1, size=(self.batch_size, 784)).astype('float32')
        }


def create_unittest(network_func, data_random):
    class __cls__(unittest.TestCase):
        def test_main(self):
            startup = fluid.Program()
            startup.random_seed = 1
            main = fluid.Program()
            with fluid.program_guard(main, startup):
                loss = network_func()

            mem_opt_main = main.clone()
            fluid.memory_optimize(mem_opt_main)
            place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            exe.run(startup)
            data = data_random()
            pe = fluid.ParallelExecutor(
                use_cuda=True,
                loss_name=loss.name,
                main_program=main,
                num_threads=10)
            for i in xrange(1000):
                pe.run(fetch_list=[], feed=next(data))
            loss_value = numpy.array(
                pe.run(fetch_list=[loss.name], feed=next(data))[0])
            del pe
            data = data_random()

            exe.run(startup)
            pe = fluid.ParallelExecutor(
                use_cuda=True,
                loss_name=loss.name,
                main_program=mem_opt_main,
                num_threads=10)
            for i in xrange(1000):
                pe.run(fetch_list=[], feed=next(data))

            loss_value_mem_opted = numpy.array(
                pe.run(fetch_list=[loss.name], feed=next(data))[0])
            self.assertAlmostEqual(loss_value[0], loss_value_mem_opted[0])

    return __cls__


TestSimpleFC = create_unittest(simple_fc, FCDataRandom)

if __name__ == '__main__':
    unittest.main()
