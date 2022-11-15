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
from paddle.fluid.framework import Variable


def cyclic_reader(reader):
    def __reader__():
        while True:
            for data in reader():
                yield data

    return __reader__


class FeedDataReader:
    def __init__(self, feed_list, reader):
        self._feed_list = []
        for var in feed_list:
            if isinstance(var, Variable):
                self._feed_list.append(var.name)
            else:
                self._feed_list.append(var)

        self._reader = cyclic_reader(reader)
        self._iter = self._reader()

    def _feed_executor(self):
        next_data = next(self._iter)
        feed_data = dict()
        assert len(self._feed_list) == len(next_data)
        for key, value in zip(self._feed_list, next_data):
            feed_data[key] = value
        return feed_data

    def _feed_parallel_executor(self, device_num):
        feed_data = []
        for _ in range(device_num):
            feed_data.append(self._feed_executor())

        return feed_data

    def get_next(self, exe, program):
        result = []
        assert isinstance(exe, fluid.Executor), "exe must be Executor"
        use_cuda = isinstance(exe.place, fluid.CUDAPlace)
        if isinstance(program, fluid.CompiledProgram):
            if program._is_data_parallel:
                use_executor = False
                if program._places is None:
                    device_num = (
                        len(fluid.cuda_places())
                        if use_cuda
                        else len(fluid.cpu_places())
                    )
                else:
                    device_num = len(program._places)
            else:
                use_executor = True
                device_num = 1
        else:
            use_executor = True
            device_num = 1

        if use_executor:
            return self._feed_executor()
        else:
            return self._feed_parallel_executor(device_num)
