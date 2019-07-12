#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np
import paddle.fluid as fluid
import os


def data_generator():
    data = [0, 1, 2, 3]
    for val in data:
        yield val


class TestDistributedReader(unittest.TestCase):
    def test_distributed_reader(self):
        trainer_num = 4
        os.environ['PADDLE_TRAINER_ID'] = str(1)
        os.environ['PADDLE_TRAINERS_NUM'] = str(trainer_num)

        reader = fluid.contrib.reader.distributed_batch_reader(data_generator)
        data = next(reader())
        assert data == 1

        os.unsetenv('PADDLE_TRAINER_ID')
        os.unsetenv('PADDLE_TRAINERS_NUM')


if __name__ == '__main__':
    unittest.main()
