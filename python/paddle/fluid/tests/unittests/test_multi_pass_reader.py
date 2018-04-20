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


class TestMultipleReader(unittest.TestCase):
    def setUp(self):
        self.batch_size = 64
        self.pass_num = 3
        # Convert mnist to recordio file
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            data_file = paddle.batch(mnist.train(), batch_size=self.batch_size)
            feeder = fluid.DataFeeder(
                feed_list=[
                    fluid.layers.data(
                        name='image', shape=[784]),
                    fluid.layers.data(
                        name='label', shape=[1], dtype='int64'),
                ],
                place=fluid.CPUPlace())
            self.num_batch = fluid.recordio_writer.convert_reader_to_recordio_file(
                './mnist.recordio', data_file, feeder)

    def test_main(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            data_file = fluid.layers.open_recordio_file(
                filename='./mnist.recordio',
                shapes=[(-1, 784), (-1, 1)],
                lod_levels=[0, 0],
                dtypes=['float32', 'int64'],
                pass_num=self.pass_num)
            img, label = fluid.layers.read_file(data_file)

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
            self.assertEqual(batch_count, self.num_batch * self.pass_num)
