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

import paddle.fluid as fluid

fluid.core._set_fuse_parameter_group_size(3)
fluid.core._set_fuse_parameter_memory_size(131072)

import unittest
import seresnext_net
from seresnext_test_base import TestResnetBase, DeviceType
from functools import partial


class TestResnetWithFuseAllReduceGPU(TestResnetBase):

    def test_seresnext_with_fused_all_reduce(self):
        # NOTE(zcd): In order to make the program faster,
        # this unit test remove drop_out and batch_norm.
        check_func = partial(self.check_network_convergence,
                             optimizer=seresnext_net.optimizer,
                             fuse_all_reduce_ops=True)
        self._compare_result_with_origin_model(check_func,
                                               use_device=DeviceType.CUDA,
                                               delta2=1e-2)


if __name__ == '__main__':
    unittest.main()
