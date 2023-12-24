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

import numpy as np
from semi_auto_parallel_util import SemiAutoParallelTestBase

import paddle
import paddle.distributed as dist


def layer_norm(input, weights, bias, normalized_shape):
    return paddle.nn.functional.layer_norm(
        input, normalized_shape, weight=weights, bias=bias
    )


class TestLayerNormSemiAutoParallel(SemiAutoParallelTestBase):
    def __init__(self):
        super().__init__()

    def check_tensor_eq(self, a, b):
        np1 = a.numpy()
        np2 = b.numpy()
        np.testing.assert_allclose(np1, np2, rtol=1e-04, verbose=True)

    def check_placements(self, output, expected_placements):
        assert (
            output.placements == expected_placements
        ), f"{output.placements}  vs {expected_placements}"

    def test_layernorm_forward(self):
        shapes = ([16, 4, 4], [16], [16])
        specs = (['x', None, None], [None], [None])
        inputs, outputs = self.runfunc_and_check(
            inputs_shape=shapes,
            inputs_specs=specs,
            op_func=layer_norm,
            with_backward=True,
            normalized_shape=[4, 4],
        )
        self.check_placements(outputs, [dist.Shard(0)])

    def test_layernorm_reshard(self):
        shapes = ([16, 4, 4], [16], [16])
        specs = ([None, None, 'x'], [None], [None])
        inputs, outputs = self.runfunc_and_check(
            inputs_shape=shapes,
            inputs_specs=specs,
            op_func=layer_norm,
            with_backward=True,
            normalized_shape=[4, 4],
        )
        self.check_placements(outputs, [dist.Replicate()])

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        self.test_layernorm_forward()
        # all to all is not supported yet for cpu
        if self._backend == "gpu":
            self.test_layernorm_reshard()


if __name__ == '__main__':
    TestLayerNormSemiAutoParallel().run_test_case()
