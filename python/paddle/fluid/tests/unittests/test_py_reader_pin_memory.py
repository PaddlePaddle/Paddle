# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import numpy as np
from threading import Thread


def user_reader(inputs):

    def _reader():
        for d in inputs:
            yield d

    return _reader


def batch_feeder(batch_reader, pin_memory=False, img_dtype="float32"):

    def _feeder():
        for batch_data in batch_reader():
            sample_batch = []
            label_batch = []
            for sample, label in batch_data:
                sample_batch.append(sample)
                label_batch.append([label])
            tensor = core.LoDTensor()
            label = core.LoDTensor()
            place = core.CUDAPinnedPlace() if pin_memory else core.CPUPlace()
            tensor.set(np.array(sample_batch, dtype=img_dtype), place)
            label.set(np.array(label_batch, dtype="int64"), place)
            yield [tensor, label]

    return _feeder


class TestPyReader(unittest.TestCase):

    def setUp(self):
        self.capacity = 10
        self.shapes = [(-1, 3, 2, 1), (-1, 1)]
        self.lod_levels = [0, 0]
        self.dtypes = ['float32', 'int64']

    def test_pin_memory_pyreader(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            place = fluid.CUDAPlace(
                0) if fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()
            executor = fluid.Executor(place)

            data_file = fluid.layers.py_reader(capacity=self.capacity,
                                               dtypes=self.dtypes,
                                               lod_levels=self.lod_levels,
                                               shapes=self.shapes)
            # feed_queue = data_file.queue
            read_out_data = fluid.layers.read_file(data_file)

            self.inputs = []
            for _ in range(10):
                sample = np.random.uniform(low=0, high=1,
                                           size=[3, 2, 1]).astype("float32")
                label = np.random.randint(low=0, high=10, dtype="int64")
                self.inputs.append((sample, label))

            self.input_tensors = []
            for d, l in batch_feeder(
                    paddle.batch(user_reader(self.inputs), batch_size=2),
                    pin_memory=True
                    if fluid.core.is_compiled_with_cuda() else False)():
                ta = fluid.LoDTensorArray()
                ta.append(d)
                ta.append(l)
                self.input_tensors.append(ta)

            self.batched_inputs = []
            for batch in paddle.batch(user_reader(self.inputs), batch_size=2)():
                feed_d = []
                feed_l = []
                for d, l in batch:
                    feed_d.append(d)
                    feed_l.append([l])
                self.batched_inputs.append([feed_d, feed_l])

            data_file.decorate_tensor_provider(
                batch_feeder(paddle.batch(user_reader(self.inputs),
                                          batch_size=2),
                             pin_memory=True
                             if fluid.core.is_compiled_with_cuda() else False))

            executor.run(fluid.default_startup_program())
            self.outputs = []

            data_file.start()
            for _ in self.input_tensors:
                self.outputs.append(
                    executor.run(fetch_list=list(read_out_data)))
            data_file.reset()
            self.validate()

    def validate(self):
        self.assertEqual(len(self.batched_inputs), len(self.outputs))
        for in_data_list, out_data_list in zip(self.batched_inputs,
                                               self.outputs):
            self.assertEqual(len(in_data_list), len(out_data_list))
            in_data_list_np = [
                np.array(in_lod_tensor) for in_lod_tensor in in_data_list
            ]
            for in_data, out_data in zip(in_data_list_np, out_data_list):
                self.assertTrue((in_data == out_data).all())


if __name__ == '__main__':
    unittest.main()
