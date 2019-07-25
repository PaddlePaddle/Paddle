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

import paddle
import paddle.fluid as fluid
import numpy
import unittest

EPOCH_NUM = 3
ITER_NUM = 5
BATCH_SIZE = 2
IMAGE_SIZE = [1, 1024, 1024]

dataset = [[numpy.random.uniform(
    low=0, high=255, size=IMAGE_SIZE), [1]]
           for i in range(ITER_NUM * BATCH_SIZE)]


def random_image(channel, height, width):
    def reader():
        for i in range(ITER_NUM * BATCH_SIZE):
            yield dataset[i]

    return reader


class TestPipeReader(unittest.TestCase):
    def test_pipe_reader(self):
        img = fluid.layers.data(name='image', shape=IMAGE_SIZE)
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        reader = fluid.reader.PipeReader([img, label])
        user_reader = random_image(*IMAGE_SIZE)
        reader.decorate_sample_generator(user_reader, batch_size=BATCH_SIZE)

        executor = fluid.Executor(fluid.CPUPlace())
        for epoch in range(EPOCH_NUM):
            reader.start()
            for it in range(ITER_NUM):
                img_t, label_t = executor.run(fluid.default_main_program(),
                                              fetch_list=[img, label],
                                              return_numpy=False)
                real = numpy.array(img_t)
                expect = numpy.array([
                    dataset[i][0]
                    for i in range(it * BATCH_SIZE, (it + 1) * BATCH_SIZE)
                ])
                assert numpy.abs(real - expect).all()
            reader.reset()


if __name__ == '__main__':
    unittest.main()
