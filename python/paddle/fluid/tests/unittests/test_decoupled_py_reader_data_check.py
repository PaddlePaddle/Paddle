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

import paddle.fluid as fluid
import paddle
import numpy as np
import unittest
import six


class TestClass(unittest.TestCase):
    def setUp(self):
        self.use_double_buffer = True
        self.use_py_reader = True

    def test_reader_data(self):
        img_shape = [28, 31]
        label_shape = [1]
        batch_size = 32
        batch_num = 10

        def fake_reader():
            for _ in six.moves.range(batch_size * batch_num):
                img = np.random.random(size=img_shape).astype('float32')
                label = np.random.random_integers(
                    low=0, high=9, size=label_shape).astype('int64')
                yield img, label

        reader = fluid.io.cache(fake_reader)
        batch_reader = fluid.io.batch(reader, batch_size=batch_size)

        places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        for p in places:
            main_prog = fluid.Program()
            startup_prog = fluid.Program()
            with fluid.program_guard(main_prog, startup_prog):
                img = fluid.layers.data(
                    shape=img_shape, dtype='float32', name='image')
                label = fluid.layers.data(
                    shape=label_shape, dtype='int64', name='label')

                feeder = fluid.DataFeeder(feed_list=[img, label], place=p)

                use_double_buffer = self.use_double_buffer
                if p._type() != fluid.CPUPlace()._type(
                ) and not use_double_buffer:
                    use_double_buffer = True

                if self.use_py_reader:
                    py_reader = fluid.io.PyReader(
                        feed_list=[img, label],
                        capacity=4,
                        iterable=True,
                        use_double_buffer=use_double_buffer)
                    py_reader.decorate_sample_list_generator(
                        batch_reader, places=p)
                else:
                    py_reader = fluid.io.DataLoader.from_generator(
                        feed_list=[img, label],
                        capacity=4,
                        iterable=True,
                        use_double_buffer=use_double_buffer
                    ).set_sample_list_generator(
                        batch_reader, places=p)

                for break_beforehand in [True, False]:
                    for epoch_id in six.moves.range(10):
                        gen = batch_reader()
                        batch_id = 0
                        for d in py_reader():
                            feed = feeder.feed(next(gen))
                            I1, L1 = feed['image'], feed['label']
                            I2, L2 = d[0]['image'], d[0]['label']

                            I1 = np.array(I1)
                            I2 = np.array(I2)
                            L1 = np.array(L1)
                            L2 = np.array(L2)

                            self.assertTrue(np.array_equal(I1, I2))
                            self.assertTrue(np.array_equal(L1, L2))

                            batch_id += 1
                            if break_beforehand and batch_id >= int(batch_num /
                                                                    2):
                                break

                        if break_beforehand:
                            self.assertTrue(next(gen, None) is not None)
                        else:
                            self.assertTrue(next(gen, None) is None)


class TestClass2(TestClass):
    def setUp(self):
        self.use_double_buffer = False
        self.use_py_reader = True


class TestClass3(TestClass):
    def setUp(self):
        self.use_double_buffer = True
        self.use_py_reader = False


class TestClass4(TestClass):
    def setUp(self):
        self.use_double_buffer = False
        self.use_py_reader = False


if __name__ == '__main__':
    unittest.main()
