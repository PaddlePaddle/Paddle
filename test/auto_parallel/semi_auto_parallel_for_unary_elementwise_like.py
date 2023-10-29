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


from semi_auto_parallel_util import SemiAutoParallelTestBase

import paddle
import paddle.distributed as dist


class TestElementwiseLikeApiForSemiAutoParallel(SemiAutoParallelTestBase):
    def __init__(self):
        super().__init__()

    def check_specs_unchanged(self, input, output):
        pass

    def test_pow_shard(self):
        x_shape = [16, 32]
        x_specs = ['x', None]
        inputs, outputs = self.runfunc_and_check(
            x_shape, x_specs, op_func=paddle.pow, with_backward=True, y=2
        )
        self.check_specs_unchanged(inputs, outputs)

    def test_cast_shard(self):
        x_shape = [16, 32]
        x_specs = ['x', None]
        inputs, outputs = self.runfunc_and_check(
            x_shape,
            x_specs,
            op_func=paddle.cast,
            with_backward=True,
            dtype="float64",
        )
        self.check_specs_unchanged(inputs, outputs)

    def test_full_like_shard(self):
        x_shape = [16, 32]
        x_specs = ['x', None]
        inputs, outputs = self.runfunc_and_check(
            x_shape,
            x_specs,
            op_func=paddle.full_like,
            with_backward=False,
            fill_value=1.0,
        )
        self.check_specs_unchanged(inputs, outputs)

    def test_scale_shard(self):
        x_shape = [16, 32]
        x_specs = ['x', None]
        inputs, outputs = self.runfunc_and_check(
            x_shape,
            x_specs,
            op_func=paddle.scale,
            with_backward=True,
            scale=2.0,
        )
        self.check_specs_unchanged(inputs, outputs)

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        self.test_pow_shard()
        self.test_full_like_shard()
        self.test_cast_shard()
        self.test_scale_shard()


if __name__ == '__main__':
    TestElementwiseLikeApiForSemiAutoParallel().run_test_case()
