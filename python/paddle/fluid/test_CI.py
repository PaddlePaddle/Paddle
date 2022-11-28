#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# test for new CI check, should not be matched.

import numpy as np

import paddle
import paddle.fluid as fluid

EPOCH_NUM = 3
ITER_NUM = 5
BATCH_SIZE = 10


def network(image, label):
    # User-defined network, here is an example of softmax regression.
    predict = fluid.layers.fc(input=image, size=10, act='softmax')
    return fluid.layers.cross_entropy(input=predict, label=label)


def reader_creator_random_image(height, width):
    def reader():
        for i in range(ITER_NUM):
            fake_image = np.random.uniform(
                low=0, high=255, size=[height, width]
            )
            fake_label = np.ones([1])
            yield fake_image, fake_label

    return reader


image = fluid.data(name='image', shape=[None, 784, 784], dtype='float32')
label = fluid.data(name='label', shape=[None, 1], dtype='int64')
reader = fluid.io.PyReader(
    feed_list=[image, label], capacity=4, iterable=True, return_list=False
)

user_defined_reader = reader_creator_random_image(784, 784)
reader.decorate_sample_list_generator(
    paddle.batch(user_defined_reader, batch_size=BATCH_SIZE),
    fluid.core.CPUPlace(),
)

loss = network(image, label)
executor = fluid.Executor(fluid.CPUPlace())
executor.run(fluid.default_startup_program())

for _ in range(EPOCH_NUM):
    for data in reader():
        executor.run(feed=data, fetch_list=[loss])
