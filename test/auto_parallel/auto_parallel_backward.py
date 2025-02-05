# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.distributed as dist


class TestBackwardAutoParallel:
    def init_data(self):
        self.mesh = dist.ProcessMesh([0], dim_names=['d0'])

        self.x = paddle.to_tensor([[1]])
        self.y = paddle.to_tensor([[1]])
        self.z = paddle.to_tensor([[1]])

        self.x.stop_gradient = False
        self.y.stop_gradient = False
        self.z.stop_gradient = False

        self.z = dist.shard_tensor(self.z, self.mesh, [dist.Replicate()])

    def run_test_case1(self):
        self.init_data()
        o = self.x * self.y
        o = o + self.z
        o = o.sum()
        o.backward()

    def run_test_case2(self):
        self.init_data()
        o = self.x + self.y
        o = o - self.z
        o = o.sum()
        o.backward()


# python -m paddle.distributed.launch --device=0 auto_parallel_backward.py
if __name__ == '__main__':
    TestBackwardAutoParallel().run_test_case1()
    TestBackwardAutoParallel().run_test_case2()
