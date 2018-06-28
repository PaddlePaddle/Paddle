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

import unittest

import paddle.fluid as fluid
import paddle
import paddle.dataset.mnist as mnist
from shutil import copyfile


class TestMultipleReader(unittest.TestCase):
    def setUp(self):
        self.batch_size = 64
        # Convert mnist to recordio file
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            reader = paddle.batch(mnist.train(), batch_size=self.batch_size)
            feeder = fluid.DataFeeder(
                feed_list=[  # order is image and label
                    fluid.layers.data(
                        name='image', shape=[784]),
                    fluid.layers.data(
                        name='label', shape=[1], dtype='int64'),
                ],
                place=fluid.CPUPlace())
            self.num_batch = fluid.recordio_writer.convert_reader_to_recordio_file(
                './mnist_0.recordio', reader, feeder)
        copyfile('./mnist_0.recordio', './mnist_1.recordio')
        copyfile('./mnist_0.recordio', './mnist_2.recordio')

    def main(self, thread_num):
        file_list = [
            './mnist_0.recordio', './mnist_1.recordio', './mnist_2.recordio'
        ]
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            data_files = fluid.layers.open_files(
                filenames=file_list,
                thread_num=thread_num,
                shapes=[(-1, 784), (-1, 1)],
                lod_levels=[0, 0],
                dtypes=['float32', 'int64'])
            img, label = fluid.layers.read_file(data_files)

            if fluid.core.is_compiled_with_cuda():
                place = fluid.CUDAPlace(0)
            else:
                place = fluid.CPUPlace()

            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            batch_count = 0
            while True:
                try:
                    img_val, = exe.run(fetch_list=[img])
                except fluid.core.EnforceNotMet as ex:
                    self.assertIn("There is no next data.", ex.message)
                    break
                batch_count += 1
                self.assertLessEqual(img_val.shape[0], self.batch_size)
            self.assertEqual(batch_count, self.num_batch * 3)

    def test_main(self):
        self.main(thread_num=3)  # thread number equals to file number
        self.main(thread_num=10)  # thread number is larger than file number
        self.main(thread_num=2)  # thread number is less than file number
