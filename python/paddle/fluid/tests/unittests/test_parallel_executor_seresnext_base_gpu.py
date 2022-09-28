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

import unittest
import seresnext_net
from seresnext_test_base import TestResnetBase, DeviceType
from functools import partial


class TestResnetGPU(TestResnetBase):

    def test_seresnext_with_learning_rate_decay(self):
        # NOTE(zcd): This test is compare the result of use parallel_executor
        # and executor, and the result of drop_out op and batch_norm op in
        # this two executor have diff, so the two ops should be removed
        # from the model.
        check_func = partial(self.check_network_convergence,
                             optimizer=seresnext_net.optimizer,
                             use_parallel_executor=False)
        self._compare_result_with_origin_model(check_func,
                                               use_device=DeviceType.CUDA,
                                               delta2=1e-5,
                                               compare_separately=False)


if __name__ == '__main__':
    unittest.main()
