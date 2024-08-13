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

import unittest

import numpy as np

import paddle
from paddle import base


class TestPyReaderCombination(unittest.TestCase):
    def setUp(self):
        self.n1 = 10
        self.n2 = 20
        self.batch_size = 2

    def create_reader(self, batch_num):
        def __impl__():
            for _ in range(batch_num):
                image = np.random.uniform(
                    low=-1, high=1, size=[batch_num, 784]
                ).astype('float32')
                label = np.random.random_integers(
                    low=0, high=9, size=[batch_num, 1]
                ).astype('int64')
                yield image, label

        return __impl__

    def assertFeedVarEqual(self, reader_list_data, py_reader_dict_data):
        image1 = reader_list_data[0]
        label1 = reader_list_data[1]

        image2 = np.array(py_reader_dict_data[0]['image'])
        label2 = np.array(py_reader_dict_data[0]['label'])
        np.testing.assert_array_equal(image1, image2)
        np.testing.assert_array_equal(label1, label2)

    # FIXME(zjl): do not know why Python 35 would raise SIGABRT if not reset reader
    # manually.
    def _reset_iterable_reader(self, py_reader):
        if py_reader.iterable:
            py_reader._loader._reset()

    def main_impl(self, place):
        with base.program_guard(base.Program(), base.Program()):
            image = paddle.static.data(
                name='image', dtype='float32', shape=[-1, 784]
            )
            label = paddle.static.data(
                name='label', dtype='int64', shape=[-1, 1]
            )

            py_reader1 = base.io.PyReader(
                feed_list=[image, label], capacity=16, iterable=True
            )
            py_reader2 = base.io.PyReader(
                feed_list=[image, label], capacity=16, iterable=True
            )

            reader1 = paddle.reader.cache(self.create_reader(self.n1))
            reader2 = paddle.reader.cache(self.create_reader(self.n2))
            py_reader1.decorate_batch_generator(reader1, places=place)
            py_reader2.decorate_batch_generator(reader2, places=place)

            for _ in range(10):
                max_num = min(self.n1, self.n2)
                batch_num = 0
                for (
                    reader_np1,
                    py_reader_dict1,
                    reader_np2,
                    py_reader_dict2,
                ) in zip(reader1(), py_reader1(), reader2(), py_reader2()):
                    self.assertFeedVarEqual(reader_np1, py_reader_dict1)
                    self.assertFeedVarEqual(reader_np2, py_reader_dict2)
                    batch_num += 1

                self.assertEqual(batch_num, max_num)

            self._reset_iterable_reader(py_reader1)
            self._reset_iterable_reader(py_reader2)

    def get_places(self):
        if base.is_compiled_with_cuda():
            return [base.CUDAPlace(0), base.CPUPlace()]
        else:
            return [base.CPUPlace()]

    def test_main(self):
        for p in self.get_places():
            self.main_impl(p)


class TestPyReaderCombination2(TestPyReaderCombination):
    def setUp(self):
        self.n1 = 20
        self.n2 = 10
        self.batch_size = 2


class TestPyReaderCombination3(TestPyReaderCombination):
    def setUp(self):
        self.n1 = 10
        self.n2 = 10
        self.batch_size = 2


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
