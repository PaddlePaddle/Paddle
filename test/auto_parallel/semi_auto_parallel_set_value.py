# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os

import numpy as np

import paddle
import paddle.distributed as dist


class TestSemiAutoParallelSetValue:
    def __init__(self):
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def test_set_value_from_numpy(self):
        p = paddle.rand(shape=[10, 10])
        p = dist.shard_tensor(p, self._mesh, [dist.Shard(0)])
        q = np.arange(0, 100).reshape(10, 10).astype("float32")
        p.set_value(q)
        np.testing.assert_equal(p.numpy(), q)

    def test_set_value_from_tensor(self):
        p = paddle.rand(shape=[10, 10])
        p = dist.shard_tensor(p, self._mesh, [dist.Shard(0)])
        q = paddle.arange(0, 100, dtype="float32").reshape(shape=[10, 10])
        p.set_value(q)
        np.testing.assert_equal(p.numpy(), q.numpy())

    def run_test_case(self):
        self.test_set_value_from_tensor()
        self.test_set_value_from_numpy()


if __name__ == '__main__':
    TestSemiAutoParallelSetValue().run_test_case()
