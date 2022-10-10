# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from test_collective_multi_nodes import TestCollectiveAPIRunnerBase, runtime_main


class TestDygrapgHybridDP(TestCollectiveAPIRunnerBase):

    def __init__(self):
        pass

    def check_pass(self, *args, **kwargs):
        from common import init_parallel_env
        import paddle
        hcg = init_parallel_env("DP16-MP1-PP1-SH1-O1", 2)
        import numpy as np
        dp_group = hcg.get_data_parallel_group()
        np.random.seed(1024)
        data = np.random.random((10 * dp_group.nranks, 100)).reshape(
            (dp_group.nranks, -1, 100))
        data_part = paddle.to_tensor(data[dp_group.rank])
        paddle.distributed.collective.all_reduce(data_part)
        data_reduced = data_part
        data_sumed = np.sum(data, axis=0)
        assert np.allclose(data_sumed,
                           data_reduced.numpy(),
                           rtol=1e-8,
                           atol=1e-8)


if __name__ == "__main__":
    runtime_main(TestDygrapgHybridDP, "dp")
