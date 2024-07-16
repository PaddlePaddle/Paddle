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


class TestPyReader(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.epoch_num = 2
        self.sample_num = 10

    def test_returnlist(self):
        def reader_creator_random_image(height, width):
            def reader():
                for i in range(self.sample_num):
                    yield np.random.uniform(
                        low=0, high=255, size=[height, width]
                    ),

            return reader

        for return_list in [True, False]:
            with base.program_guard(base.Program(), base.Program()):
                image = paddle.static.data(
                    name='image', shape=[-1, 784, 784], dtype='float32'
                )
                reader = base.io.PyReader(
                    feed_list=[image],
                    capacity=4,
                    iterable=True,
                    return_list=return_list,
                )

                user_defined_reader = reader_creator_random_image(784, 784)
                reader.decorate_sample_list_generator(
                    paddle.batch(
                        user_defined_reader, batch_size=self.batch_size
                    ),
                    base.core.CPUPlace(),
                )
                # definition of network is omitted
                executor = base.Executor(base.core.CPUPlace())

                for _ in range(self.epoch_num):
                    for data in reader():
                        if return_list:
                            executor.run(feed={"image": data[0][0]})
                        else:
                            executor.run(feed=data)

            with base.dygraph.guard():
                batch_py_reader = base.io.PyReader(capacity=2)
                user_defined_reader = reader_creator_random_image(784, 784)
                batch_py_reader.decorate_sample_generator(
                    user_defined_reader,
                    batch_size=self.batch_size,
                    places=base.core.CPUPlace(),
                )

                for epoch in range(self.epoch_num):
                    for _, data in enumerate(batch_py_reader()):
                        # empty network
                        pass


if __name__ == "__main__":
    unittest.main()
