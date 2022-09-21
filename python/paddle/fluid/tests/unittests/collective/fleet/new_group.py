# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import os
import paddle


class TestNewGroupAPI(object):

    def __init__(self):
        paddle.distributed.init_parallel_env()
        d1 = np.array([1, 2, 3])
        d2 = np.array([2, 3, 4])
        self.tensor1 = paddle.to_tensor(d1)
        self.tensor2 = paddle.to_tensor(d2)

    def test_all(self):
        gp = paddle.distributed.new_group([0, 1])
        print("gp info:", gp)
        print("test new group api ok")

        tmp = np.array([0, 0, 0])
        result = paddle.to_tensor(tmp)
        paddle.distributed.scatter(result, [self.tensor2, self.tensor1],
                                   src=0,
                                   group=gp,
                                   sync_op=True)
        if gp.rank == 0:
            assert np.array_equal(result, self.tensor2)
        elif gp.rank == 1:
            assert np.array_equal(result, self.tensor1)
        print("test scatter api ok")

        paddle.distributed.broadcast(result, src=1, group=gp, sync_op=True)
        assert np.array_equal(result, self.tensor1)
        print("test broadcast api ok")

        paddle.distributed.reduce(result, dst=0, group=gp, sync_op=True)
        if gp.rank == 0:
            assert np.array_equal(result, paddle.add(self.tensor1,
                                                     self.tensor1))
        elif gp.rank == 1:
            assert np.array_equal(result, self.tensor1)
        print("test reduce api ok")

        paddle.distributed.all_reduce(result, sync_op=True)
        assert np.array_equal(
            result,
            paddle.add(paddle.add(self.tensor1, self.tensor1), self.tensor1))
        print("test all_reduce api ok")

        paddle.distributed.wait(result, gp, use_calc_stream=True)
        paddle.distributed.wait(result, gp, use_calc_stream=False)
        print("test wait api ok")

        result = []
        paddle.distributed.all_gather(result,
                                      self.tensor1,
                                      group=gp,
                                      sync_op=True)
        assert np.array_equal(result[0], self.tensor1)
        assert np.array_equal(result[1], self.tensor1)
        print("test all_gather api ok")

        paddle.distributed.barrier(group=gp)
        print("test barrier api ok")

        return


if __name__ == "__main__":
    gpt = TestNewGroupAPI()
    gpt.test_all()
