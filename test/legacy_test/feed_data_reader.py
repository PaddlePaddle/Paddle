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

from paddle import base
from paddle.base.framework import Variable


def cyclic_reader(reader):
    def __reader__():
        while True:
            yield from reader()

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
        feed_data = {}
        assert len(self._feed_list) == len(next_data)
        for key, value in zip(self._feed_list, next_data):
            feed_data[key] = value
        return feed_data

    def get_next(self, exe, program):
        assert isinstance(exe, base.Executor), "exe must be Executor"
        return self._feed_executor()
