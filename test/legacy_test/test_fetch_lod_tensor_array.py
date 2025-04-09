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

import unittest

import numpy as np
from simple_nets import simple_fc_net, simple_fc_net_with_inputs

import paddle
from paddle import base


class TestFetchDenseTensorArray(unittest.TestCase):
    def build_program(self, main_program, startup_program):
        with base.unique_name.guard():
            with base.program_guard(main_program, startup_program):
                i = paddle.zeros(shape=[1], dtype='int64')
                img = paddle.static.data(
                    name='image', shape=[-1, 784], dtype='float32'
                )
                label = paddle.static.data(
                    name='label', shape=[-1, 1], dtype='int64'
                )
                loss = simple_fc_net_with_inputs(img, label, class_num=10)
                loss = simple_fc_net()
                opt = paddle.optimizer.SGD(learning_rate=0.001)
                opt.minimize(loss)

                array = paddle.tensor.array_write(x=img, i=i)
                i = paddle.increment(i)
                paddle.tensor.array_write(x=label, i=i, array=array)
                i = paddle.increment(i)
                paddle.tensor.array_write(x=loss, i=i, array=array)

                return loss, array

    def check_network(self, use_cuda=True):
        main_program = base.Program()
        startup_program = base.Program()

        loss, array = self.build_program(main_program, startup_program)

        batch_size = 32
        image = np.random.normal(size=(batch_size, 784)).astype('float32')
        label = np.random.randint(0, 10, (batch_size, 1), dtype="int64")

        place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
        exe = base.Executor(place)
        exe.run(startup_program)
        feed_dict = {'image': image, 'label': label}

        if not paddle.base.framework.use_pir_api():
            build_strategy = base.BuildStrategy()
            binary = base.CompiledProgram(
                main_program, build_strategy=build_strategy
            )
        else:
            binary = main_program

        for _ in range(3):
            loss_v, array_v = exe.run(
                binary, feed=feed_dict, fetch_list=[loss, array]
            )
            self.assertEqual(loss_v.shape, ())
            self.assertEqual(array_v[0].shape, (batch_size, 784))
            self.assertEqual(array_v[1].shape, (batch_size, 1))
            self.assertEqual(array_v[2].shape, ())
            np.testing.assert_allclose(loss_v, array_v[2], rtol=1e-05)

    def test_fetch_dense_tensor_array(self):
        if base.core.is_compiled_with_cuda():
            self.check_network(use_cuda=True)
        self.check_network(use_cuda=False)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
