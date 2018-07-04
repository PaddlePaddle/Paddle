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

import unittest
import paddle.fluid as fluid
import paddle.v2 as paddle
import numpy as np


class TestDataBalance(unittest.TestCase):
    def prepare_data(self):
        def fake_data_generator():
            for n in xrange(self.total_ins_num):
                yield np.ones((3, 4)) * n, n

        # Prepare data
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            reader = paddle.batch(
                fake_data_generator, batch_size=self.batch_size)
            feeder = fluid.DataFeeder(
                feed_list=[
                    fluid.layers.data(
                        name='image', shape=[3, 4], dtype='float32'),
                    fluid.layers.data(
                        name='label', shape=[1], dtype='int64'),
                ],
                place=fluid.CPUPlace())
            self.num_batches = fluid.recordio_writer.convert_reader_to_recordio_file(
                self.data_file_name, reader, feeder)

    def prepare_lod_data(self):
        def fake_data_generator():
            for n in xrange(1, self.total_ins_num + 1):
                d1 = (np.ones((n, 3)) * n).astype('float32')
                d2 = (np.array(n).reshape((1, 1))).astype('int32')
                yield d1, d2

        # Prepare lod data
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            with fluid.recordio_writer.create_recordio_writer(
                    filename=self.lod_data_file_name) as writer:
                eof = False
                generator = fake_data_generator()
                while (not eof):
                    data_batch = [
                        np.array([]).reshape((0, 3)), np.array([]).reshape(
                            (0, 1))
                    ]
                    lod = [0]
                    for _ in xrange(self.batch_size):
                        try:
                            ins = generator.next()
                        except StopIteration:
                            eof = True
                            break
                        for i, d in enumerate(ins):
                            data_batch[i] = np.concatenate(
                                (data_batch[i], d), axis=0)
                        lod.append(lod[-1] + ins[0].shape[0])
                    if data_batch[0].shape[0] > 0:
                        for i, d in enumerate(data_batch):
                            t = fluid.LoDTensor()
                            t.set(data_batch[i], fluid.CPUPlace())
                            if i == 0:
                                t.set_lod([lod])
                            writer.append_tensor(t)
                        writer.complete_append_tensor()

    def setUp(self):
        self.use_cuda = fluid.core.is_compiled_with_cuda()
        self.data_file_name = './data_balance_test.recordio'
        self.lod_data_file_name = './data_balance_with_lod_test.recordio'
        self.total_ins_num = 50
        self.batch_size = 10
        self.prepare_data()
        self.prepare_lod_data()

    def main(self):
        main_prog = fluid.Program()
        startup_prog = fluid.Program()
        with fluid.program_guard(main_prog, startup_prog):
            data_reader = fluid.layers.io.open_files(
                filenames=[self.data_file_name],
                shapes=[[-1, 3, 4], [-1, 1]],
                lod_levels=[0, 0],
                dtypes=['float32', 'int64'])
            if self.use_cuda:
                data_reader = fluid.layers.double_buffer(data_reader)
            image, label = fluid.layers.read_file(data_reader)

            place = fluid.CUDAPlace(0) if self.use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_prog)

            parallel_exe = fluid.ParallelExecutor(
                use_cuda=self.use_cuda, main_program=main_prog)

            if (parallel_exe.device_count > self.batch_size):
                print("WARNING: Unittest TestDataBalance skipped. \
                    For the result is not correct when device count \
                    is larger than batch size.")
                exit(0)
            fetch_list = [image.name, label.name]

            data_appeared = [False] * self.total_ins_num
            while (True):
                try:
                    image_val, label_val = parallel_exe.run(fetch_list,
                                                            return_numpy=True)
                except fluid.core.EOFException:
                    break
                ins_num = image_val.shape[0]
                broadcasted_label = np.ones(
                    (ins_num, 3, 4)) * label_val.reshape((ins_num, 1, 1))
                self.assertEqual(image_val.all(), broadcasted_label.all())
                for l in label_val:
                    self.assertFalse(data_appeared[l[0]])
                    data_appeared[l[0]] = True
            for i in data_appeared:
                self.assertTrue(i)

    def main_lod(self):
        main_prog = fluid.Program()
        startup_prog = fluid.Program()
        with fluid.program_guard(main_prog, startup_prog):
            data_reader = fluid.layers.io.open_files(
                filenames=[self.lod_data_file_name],
                shapes=[[-1, 3], [-1, 1]],
                lod_levels=[1, 0],
                dtypes=['float32', 'int32'],
                thread_num=1)
            ins, label = fluid.layers.read_file(data_reader)

            place = fluid.CUDAPlace(0) if self.use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_prog)

            parallel_exe = fluid.ParallelExecutor(
                use_cuda=self.use_cuda, main_program=main_prog)

            if (parallel_exe.device_count > self.batch_size):
                print("WARNING: Unittest TestDataBalance skipped. \
                    For the result is not correct when device count \
                    is larger than batch size.")
                exit(0)
            fetch_list = [ins.name, label.name]

            data_appeared = [False] * self.total_ins_num
            while (True):
                try:
                    ins_tensor, label_tensor = parallel_exe.run(
                        fetch_list, return_numpy=False)
                except fluid.core.EOFException:
                    break

                ins_val = np.array(ins_tensor)
                label_val = np.array(label_tensor)
                ins_lod = ins_tensor.lod()[0]
                self.assertEqual(ins_val.shape[1], 3)
                self.assertEqual(label_val.shape[1], 1)
                self.assertEqual(len(ins_lod) - 1, label_val.shape[0])
                for i in range(0, len(ins_lod) - 1):
                    ins_elem = ins_val[ins_lod[i]:ins_lod[i + 1]][:]
                    label_elem = label_val[i][0]
                    self.assertEqual(ins_elem.all(), label_elem.all())
                    self.assertFalse(data_appeared[int(label_elem - 1)])
                    data_appeared[int(label_elem - 1)] = True

            for i in data_appeared:
                self.assertTrue(i)

    def test_all(self):
        self.main()
        self.main_lod()
