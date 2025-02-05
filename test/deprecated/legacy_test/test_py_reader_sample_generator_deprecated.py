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

import math
import os
import unittest

import numpy as np

import paddle
from paddle import base

os.environ['CPU_NUM'] = '1'


def random_reader(sample_num):
    def __impl__():
        for _ in range(sample_num):
            yield np.random.random(size=[784]).astype(
                'float32'
            ), np.random.random_integers(low=0, high=9, size=[1]).astype(
                'int64'
            )

    return paddle.reader.cache(__impl__)


class TestCaseBase(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.epoch_num = 2
        self.sample_num = 165

    def generate_all_data(self, reader):
        ret = []
        for d in reader():
            slots = [[], []]
            for item in d:
                slots[0].append(item[0])
                slots[1].append(item[1])
            slots = [np.array(slot) for slot in slots]
            ret.append(slots)
        return ret

    def run_main(self, reader, use_sample_generator, iterable, drop_last):
        image = paddle.static.data(
            name='image', dtype='float32', shape=[-1, 784]
        )
        label = paddle.static.data(name='label', dtype='int64', shape=[-1, 1])
        py_reader = base.io.PyReader(
            feed_list=[image, label],
            capacity=16,
            iterable=iterable,
            use_double_buffer=False,
        )

        batch_reader = paddle.batch(reader, self.batch_size, drop_last)
        all_datas = self.generate_all_data(batch_reader)

        if not use_sample_generator:
            py_reader.decorate_sample_list_generator(
                batch_reader, places=base.cpu_places()
            )
        else:
            py_reader.decorate_sample_generator(
                reader, self.batch_size, drop_last, places=base.cpu_places()
            )

        if drop_last:
            batch_num = int(self.sample_num / self.batch_size)
        else:
            batch_num = math.ceil(float(self.sample_num) / self.batch_size)

        exe = base.Executor(base.CPUPlace())
        exe.run(base.default_startup_program())
        for _ in range(self.epoch_num):
            if py_reader.iterable:
                step = 0
                for data in py_reader():
                    img, lbl = exe.run(feed=data, fetch_list=[image, label])
                    self.assertArrayEqual(img, all_datas[step][0])
                    self.assertArrayEqual(lbl, all_datas[step][1])
                    step += 1
                self.assertEqual(step, len(all_datas))
            else:
                step = 0
                try:
                    py_reader.start()
                    while True:
                        img, lbl = exe.run(fetch_list=[image, label])
                        self.assertArrayEqual(img, all_datas[step][0])
                        self.assertArrayEqual(lbl, all_datas[step][1])
                        step += 1
                except base.core.EOFException:
                    py_reader.reset()
                    self.assertEqual(step, len(all_datas))
                    break

    def assertArrayEqual(self, arr1, arr2):
        self.assertEqual(arr1.shape, arr2.shape)
        self.assertTrue((arr1 == arr2).all())

    def test_main(self):
        reader = random_reader(self.sample_num)
        for use_sample_generator in [False, True]:
            for iterable in [False]:
                for drop_last in [False, True]:
                    with base.program_guard(base.Program(), base.Program()):
                        self.run_main(
                            reader, use_sample_generator, iterable, drop_last
                        )


class TestCase1(TestCaseBase):
    def setUp(self):
        self.batch_size = 32
        self.epoch_num = 10
        self.sample_num = 160


class TestCase2(TestCaseBase):
    def setUp(self):
        self.batch_size = 32
        self.epoch_num = 2
        self.sample_num = 200


class TestCase3(TestCaseBase):
    def setUp(self):
        self.batch_size = 32
        self.epoch_num = 2
        self.sample_num = 159


if __name__ == '__main__':
    unittest.main()
