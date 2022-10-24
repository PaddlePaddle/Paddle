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
import unittest


class TestCifar10(unittest.TestCase):
    def test_main(self):
        reader = paddle.dataset.cifar.train10(cycle=False)
        sample_num = 0
        for _ in reader():
            sample_num += 1

        cyclic_reader = paddle.dataset.cifar.train10(cycle=True)
        read_num = 0
        for data in cyclic_reader():
            read_num += 1
            self.assertEquals(len(data), 2)
            if read_num == sample_num * 2:
                break

        self.assertEquals(read_num, sample_num * 2)


if __name__ == '__main__':
    unittest.main()
