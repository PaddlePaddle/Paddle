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
from paddle.nn.functional.flash_attention import flash_attention


class TestFlashAttentionSemiAutoParallel(SemiAutoParallelTestBase):
    def __init__(self):
        super().__init__()

    def check_placements(self, output, expected_placements):
        assert (
            output.placements == expected_placements
        ), f"{output.placements}  vs {expected_placements}"

    def test_flash_att_forward(self, is_gqa=False):
        if is_gqa:
            shapes = ([2, 256, 8, 128], [2, 256, 2, 128], [2, 256, 2, 128])
        else:
            shapes = ([2, 256, 2, 128], [2, 256, 2, 128], [2, 256, 2, 128])
        specs = (
            ['x', None, None, None],
            ["x", None, None, None],
            ['x', None, None, None],
        )
        inputs, outputs = self.runfunc_and_check(
            inputs_shape=shapes,
            inputs_specs=specs,
            op_func=flash_attention,
            with_backward=True,
            causal=True,
        )
        self.check_placements(outputs[0], [dist.Shard(0)])

    def test_flash_att_forward_reshard(self, is_gqa=False):
        if is_gqa:
            shapes = ([2, 256, 8, 128], [2, 256, 2, 128], [2, 256, 2, 128])
        else:
            shapes = ([2, 256, 2, 128], [2, 256, 2, 128], [2, 256, 2, 128])
        specs = (
            ['x', None, None, None],
            [None, None, None, 'x'],
            ['x', None, None, None],
        )
        inputs, outputs = self.runfunc_and_check(
            inputs_shape=shapes,
            inputs_specs=specs,
            op_func=flash_attention,
            with_backward=True,
            causal=True,
        )
        self.check_placements(outputs[0], [dist.Shard(0)])

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        # flash attention is not supported yet for cpu
        if self._backend == "gpu":
            cuda_version_main = int(paddle.version.cuda().split(".")[0])
            device_prop_main = paddle.device.cuda.get_device_capability()[0]
            if cuda_version_main >= 11 and device_prop_main >= 8:
                self.test_flash_att_forward()
                self.test_flash_att_forward(is_gqa=True)
                self.test_flash_att_forward_reshard()
                self.test_flash_att_forward_reshard(is_gqa=True)


if __name__ == '__main__':
    TestFlashAttentionSemiAutoParallel().run_test_case()
