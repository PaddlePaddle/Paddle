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

from semi_auto_parallel_util import SemiAutoParallelTestBase

import paddle
import paddle.distributed as dist


class TestExpandApiForSemiAutoParallel(SemiAutoParallelTestBase):
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def check_dim_mapping(self, output, expected_dim_mapping):
        assert (
            output.dist_attr.dims_mapping == expected_dim_mapping
        ), f"{output.dist_attr.dims_mapping}  vs {expected_dim_mapping}"

    def test_expand_shard(self):
        x_shape = ([10, 1, 8],)
        x_specs = ([None, None, 'x'],)
        _, output = self.runfunc_and_check(
            x_shape,
            x_specs,
            op_func=paddle.expand,
            with_backward=True,
            shape=[-1, 8, -1],
        )
        self.check_dim_mapping(output, [-1, -1, 'x'])

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        self.test_expand_shard()


if __name__ == '__main__':
    TestExpandApiForSemiAutoParallel().run_test_case()
