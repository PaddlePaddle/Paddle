#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from simple_nets import simple_fc_net, simple_fc_net_with_inputs

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers


class TestFetchLoDTensorArray(unittest.TestCase):
    def build_program(self, main_program, startup_program):
        with fluid.unique_name.guard():
            with fluid.program_guard(main_program, startup_program):
                i = layers.zeros(shape=[1], dtype='int64')
                img = fluid.data(name='image', shape=[-1, 784], dtype='float32')
                label = fluid.data(name='label', shape=[-1, 1], dtype='int64')
                loss = simple_fc_net_with_inputs(img, label, class_num=10)
                loss = simple_fc_net()
                opt = fluid.optimizer.SGD(learning_rate=0.001)
                opt.minimize(loss)

                array = paddle.tensor.array_write(x=img, i=i)
                i = paddle.increment(i)
                paddle.tensor.array_write(x=label, i=i, array=array)
                i = paddle.increment(i)
                paddle.tensor.array_write(x=loss, i=i, array=array)

                return loss, array

    def check_network(self, use_cuda=True):
        os.environ["CPU_NUM"] = str(2)
        main_program = fluid.Program()
        startup_program = fluid.Program()

        loss, array = self.build_program(main_program, startup_program)

        batch_size = 32
        image = np.random.normal(size=(batch_size, 784)).astype('float32')
        label = np.random.randint(0, 10, (batch_size, 1), dtype="int64")

        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        feed_dict = {'image': image, 'label': label}

        build_strategy = fluid.BuildStrategy()
        binary = fluid.CompiledProgram(main_program).with_data_parallel(
            loss_name=loss.name, build_strategy=build_strategy
        )

        device_num = fluid.core.get_cuda_device_count() if use_cuda else 2
        for _ in range(3):
            loss_v, array_v = exe.run(
                binary,
                feed=feed_dict,
                fetch_list=[loss, array],
                return_merged=False,
            )
            self.assertEqual(np.array(loss_v).shape, (device_num, 1))
            self.assertEqual(
                np.array(array_v[0][0]).shape, (batch_size / device_num, 784)
            )
            self.assertEqual(
                np.array(array_v[0][1]).shape, (batch_size / device_num, 1)
            )
            self.assertEqual(np.array(array_v[0][2]).shape, (1,))

        for _ in range(3):
            loss_v, array_v = exe.run(
                binary,
                feed=feed_dict,
                fetch_list=[loss, array],
                return_merged=True,
            )
            self.assertEqual(np.array(loss_v).shape, (device_num,))
            self.assertEqual(np.array(array_v[0]).shape, (batch_size, 784))
            self.assertEqual(np.array(array_v[1]).shape, (batch_size, 1))
            np.testing.assert_allclose(loss_v, array_v[2], rtol=1e-05)

    def test_fetch_lod_tensor_array(self):
        if fluid.core.is_compiled_with_cuda():
            self.check_network(use_cuda=True)
        self.check_network(use_cuda=False)

    def test_fetch_unmerged_parallel_graph(self):
        fluid.core.globals()['FLAGS_enable_parallel_graph'] = True
        if fluid.core.is_compiled_with_cuda():
            self.check_network(use_cuda=True)
        self.check_network(use_cuda=False)
        fluid.core.globals()['FLAGS_enable_parallel_graph'] = False


if __name__ == '__main__':
    unittest.main()
