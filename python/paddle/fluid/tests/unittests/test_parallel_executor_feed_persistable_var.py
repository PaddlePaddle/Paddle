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

import os
import unittest
from functools import partial

import numpy
from simple_nets import init_data, simple_fc_net

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core


class TestFeedPersistableVar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['CPU_NUM'] = str(4)
        batch_size = 4
        cls.img, cls.label = init_data(
            batch_size, img_shape=[784], label_range=9
        )
        cls.feed_dict = {
            'image': cls.img,
            'label': cls.label,
            'learning_rate': numpy.array([1.0]).astype("float32"),
        }

    def optimizer(self):
        learning_rate = paddle.static.create_global_var(
            name="learning_rate",
            shape=[1],
            value=1.0,
            dtype='float32',
            persistable=True,
        )
        optimizer = fluid.optimizer.SGD(learning_rate=learning_rate)
        return optimizer

    def check_feed_persistable_var(self, feed_dict, use_cuda=False):
        if use_cuda and not core.is_compiled_with_cuda():
            return
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)

        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            loss = simple_fc_net()

            optimizer = self.optimizer()
            optimizer.minimize(loss)

            exe.run(program=startup)
            compiled_prog = fluid.compiler.CompiledProgram(
                main
            ).with_data_parallel(loss_name=loss.name)

            exe.run(program=compiled_prog, feed=feed_dict)

    def test_feed_persistable_var(self):
        self.check_feed_persistable_var(self.feed_dict)
        self.check_feed_persistable_var(self.feed_dict, use_cuda=True)

        self.feed_dict['learning_rate'] = numpy.array([1.0, 1.0]).astype(
            "float32"
        )
        self.check_feed_persistable_var(self.feed_dict, use_cuda=True)

        self.feed_dict['learning_rate'] = numpy.array([1.0, 1.0]).astype(
            "float32"
        )
        run = partial(self.check_feed_persistable_var, self.feed_dict)
        self.assertRaises(RuntimeError, run)

        self.feed_dict['image'] = self.img[0, :]
        self.feed_dict['label'] = self.label[0, :]
        run = partial(self.check_feed_persistable_var, self.feed_dict)
        self.assertRaises(RuntimeError, run)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
