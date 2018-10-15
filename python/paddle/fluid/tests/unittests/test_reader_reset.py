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


class TestReaderReset(unittest.TestCase):
    def prepare_data(self):
        def fake_data_generator():
            for n in range(self.total_ins_num):
                yield np.ones(self.ins_shape) * n, n

        # Prepare data
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            reader = paddle.batch(fake_data_generator, batch_size=1)
            feeder = fluid.DataFeeder(
                feed_list=[
                    fluid.layers.data(
                        name='data', shape=[3], dtype='float32'),
                    fluid.layers.data(
                        name='label', shape=[1], dtype='int64'),
                ],
                place=fluid.CPUPlace())
            fluid.recordio_writer.convert_reader_to_recordio_file(
                self.data_file_name, reader, feeder)

    def setUp(self):
        # set parallel threads to fit 20 batches in line 49
        os.environ['CPU_NUM'] = str(20)
        self.use_cuda = fluid.core.is_compiled_with_cuda()
        self.data_file_name = './reader_reset_test.recordio'
        self.ins_shape = [3]
        self.batch_size = 5
        self.total_ins_num = self.batch_size * 20
        self.test_pass_num = 100
        self.prepare_data()

    def main(self, with_double_buffer):
        main_prog = fluid.Program()
        startup_prog = fluid.Program()

        with fluid.program_guard(main_prog, startup_prog):
            data_reader_handle = fluid.layers.io.open_files(
                filenames=[self.data_file_name],
                shapes=[[-1] + self.ins_shape, [-1, 1]],
                lod_levels=[0, 0],
                dtypes=['float32', 'int64'],
                thread_num=1,
                pass_num=1)
            data_reader = fluid.layers.io.batch(data_reader_handle,
                                                self.batch_size)
            if with_double_buffer:
                data_reader = fluid.layers.double_buffer(data_reader)
            image, label = fluid.layers.read_file(data_reader)
            fetch_list = [image.name, label.name]

        place = fluid.CUDAPlace(0) if self.use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_prog)

        build_strategy = fluid.BuildStrategy()
        if with_double_buffer:
            build_strategy.enable_data_balance = True
        exec_strategy = fluid.ExecutionStrategy()
        parallel_exe = fluid.ParallelExecutor(
            use_cuda=self.use_cuda,
            main_program=main_prog,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)

        data_appeared = [False] * self.total_ins_num
        pass_count = 0
        while (True):
            try:
                data_val, label_val = parallel_exe.run(fetch_list,
                                                       return_numpy=True)
                ins_num = data_val.shape[0]
                broadcasted_label = np.ones((ins_num, ) + tuple(
                    self.ins_shape)) * label_val.reshape((ins_num, 1))
                self.assertEqual(data_val.all(), broadcasted_label.all())
                for l in label_val:
                    self.assertFalse(data_appeared[l[0]])
                    data_appeared[l[0]] = True

            except fluid.core.EOFException:
                pass_count += 1
                if with_double_buffer:
                    data_appeared = data_appeared[:-parallel_exe.device_count *
                                                  self.batch_size]
                for i in data_appeared:
                    self.assertTrue(i)
                if pass_count < self.test_pass_num:
                    data_appeared = [False] * self.total_ins_num
                    data_reader_handle.reset()
                else:
                    break

    def test_all(self):
        self.main(with_double_buffer=False)
        self.main(with_double_buffer=True)


if __name__ == '__main__':
    unittest.main()
