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

import os

os.environ['CPU_NUM'] = str(1)
import paddle.fluid as fluid
from paddle.fluid import compiler
import paddle
import numpy as np
import unittest


class TestReaderReset(unittest.TestCase):

    def prepare_data(self):

        def fake_data_generator():
            for n in range(self.total_ins_num):
                yield np.ones(self.ins_shape) * n, n

        return fake_data_generator

    def setUp(self):
        self.use_cuda = fluid.core.is_compiled_with_cuda()
        self.ins_shape = [3]
        self.batch_size = 5
        self.batch_num = 20
        self.total_ins_num = self.batch_size * self.batch_num
        self.test_pass_num = 100
        self.prepare_data()

    def main(self, with_double_buffer):
        main_prog = fluid.Program()
        startup_prog = fluid.Program()

        with fluid.program_guard(main_prog, startup_prog):
            image = fluid.layers.data(name='image',
                                      shape=self.ins_shape,
                                      dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            data_reader_handle = fluid.io.PyReader(
                feed_list=[image, label],
                capacity=16,
                iterable=False,
                use_double_buffer=with_double_buffer)
            fetch_list = [image.name, label.name]

        place = fluid.CUDAPlace(0) if self.use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_prog)

        data_reader_handle.decorate_sample_list_generator(
            paddle.batch(self.prepare_data(), batch_size=self.batch_size))

        train_cp = compiler.CompiledProgram(main_prog).with_data_parallel(
            places=[place])

        batch_id = 0
        pass_count = 0
        while pass_count < self.test_pass_num:
            data_reader_handle.start()
            try:
                while True:
                    data_val, label_val = exe.run(train_cp,
                                                  fetch_list=fetch_list,
                                                  return_numpy=True)
                    ins_num = data_val.shape[0]
                    broadcasted_label = np.ones((
                        ins_num, ) + tuple(self.ins_shape)) * label_val.reshape(
                            (ins_num, 1))
                    self.assertEqual(data_val.all(), broadcasted_label.all())
                    batch_id += 1
            except fluid.core.EOFException:
                data_reader_handle.reset()
                pass_count += 1
                self.assertEqual(pass_count * self.batch_num, batch_id)

        self.assertEqual(pass_count, self.test_pass_num)

    def test_all(self):
        self.main(with_double_buffer=False)
        self.main(with_double_buffer=True)


if __name__ == '__main__':
    unittest.main()
