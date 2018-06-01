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

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.dataset.flowers as flowers

batch_size = 1
data_shape = [3, 224, 224]

with fluid.program_guard(fluid.Program(), fluid.Program()):
    reader = paddle.batch(flowers.train(), batch_size=batch_size)
    feeder = fluid.DataFeeder(
        feed_list=[
            fluid.layers.data(
                name='data', shape=data_shape, dtype='float32'),
            fluid.layers.data(
                name='label', shape=[1], dtype='int64'),
        ],
        place=fluid.CPUPlace())
    fluid.recordio_writer.convert_reader_to_recordio_file(
        './flowers_1.recordio', reader, feeder)
