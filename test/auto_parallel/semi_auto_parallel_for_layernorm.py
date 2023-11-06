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


class TestSplitAndConcatSemiAutoParallel(SemiAutoParallelTestBase):
    def __init__(self):
        super().__init__()

    def check_dim_mapping(self, inputs, output, expected_dim_mapping):
        for t in inputs:
            assert (
                t.dist_attr.dim_mapping == expected_dim_mapping
            ), f"{t.dist_attr.dim_mapping}  vs {expected_dim_mapping}"
        assert (
            output.dist_attr.dim_mapping == expected_dim_mapping
        ), f"{output.dist_attr.dim_mapping}  vs {expected_dim_mapping}"

    def test_concat_forward(self):
        shapes = [[16, 4, 4], [64, 4, 4]]
        specs = [[None, None, 'x'], [None, None, 'x']]
        inputs, outputs = self.runfunc_and_check(
            inputs_shape=shapes,
            inputs_specs=specs,
            op_func=paddle.concat,
            with_backward=False,
            axis=0,
        )
        self.check_dim_mapping(inputs, outputs, [-1, -1, 0])

    def test_concat_forward_reshard(self):
        shapes = [[16, 4, 4], [64, 4, 4]]
        specs = [['x', None, None], [None, None, 'x']]
        inputs, outputs = self.runfunc_and_check(
            inputs_shape=shapes,
            inputs_specs=specs,
            op_func=paddle.concat,
            with_backward=False,
            axis=0,
        )
        self.check_dim_mapping(inputs, outputs, [-1, -1, 0])

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        self.test_concat_forward()
        # all to all is not supported yet for cpu
        if self._backend == "gpu":
            self.test_concat_forward_reshard()


if __name__ == '__main__':
    TestSplitAndConcatSemiAutoParallel().run_test_case()
