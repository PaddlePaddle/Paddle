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

from __future__ import print_function
import os
import paddle.fluid as fluid
import paddle
import numpy as np
import unittest
import six


def simple_fc_net():
    pyreader = fluid.layers.py_reader(
        capacity=2,
        shapes=([-1, 784], [-1, 1]),
        dtypes=("float32", "int64"),
        use_double_buffer=True)
    img, label = fluid.layers.read_file(pyreader)
    hidden = img
    for _ in range(4):
        hidden = fluid.layers.fc(
            hidden,
            size=200,
            act='tanh',
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=1.0)))
    prediction = fluid.layers.fc(hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    loss = fluid.layers.mean(loss)
    return loss, pyreader


class TestReaderEmptyBatch(unittest.TestCase):
    def fake_data(self):
        def reader():
            for n in six.moves.xrange(self.total_samples):
                yield np.ones(shape=[784]) * n, n

        return reader

    def setUp(self):
        self.batch_size = 2  # batch size for each device
        self.use_cuda = fluid.core.is_compiled_with_cuda()
        self.num_devices = fluid.core.get_cuda_device_count(
        ) if self.use_cuda else 1
        self.total_samples = self.batch_size * (self.num_devices - 1)

    def test(self):
        if not self.use_cuda:
            return
        main_prog = fluid.Program()
        startup_prog = fluid.Program()
        with fluid.program_guard(main_prog, startup_prog):
            loss, pyreader = simple_fc_net()

            pyreader.decorate_paddle_reader(
                paddle.batch(
                    self.fake_data(),
                    batch_size=self.batch_size,
                    num_devices=fluid.core.get_cuda_device_count()))

            optimizer = fluid.optimizer.Adam()
            optimizer.minimize(loss)
            place = fluid.CUDAPlace(0) if self.use_cuda else fluid.CPUPlace()

            startup_exe = fluid.Executor(place)
            startup_exe.run(startup_prog)

            exe = fluid.ParallelExecutor(self.use_cuda, loss_name=loss.name)
            pyreader.start()
            with self.assertRaises(fluid.core.EOFException):
                exe.run(fetch_list=[])
            pyreader.reset()


if __name__ == "__main__":
    unittest.main()
