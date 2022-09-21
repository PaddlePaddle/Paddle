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

import unittest
import paddle
import test_communication_api_base as test_base


class TestCommunicationStreamSendRecvAPI(test_base.CommunicationTestDistBase):

    def setUp(self):
        super(TestCommunicationStreamSendRecvAPI, self).setUp(num_of_devices=2,
                                                              timeout=120)
        self._default_envs = {
            "backend": "nccl",
            "shape": "(100, 200)",
            "dtype": "float32",
            "seeds": str(self._seeds)
        }
        self._changeable_envs = {
            "sync_op": ["True", "False"],
            "use_calc_stream": ["True", "False"]
        }

    def test_sendrecv_stream(self):
        envs_list = test_base.gen_product_envs_list(self._default_envs,
                                                    self._changeable_envs)
        for envs in envs_list:
            if eval(envs["use_calc_stream"]) and not eval(envs["sync_op"]):
                continue
            self.run_test_case("communication_stream_sendrecv_api_dygraph.py",
                               user_defined_envs=envs)

    def tearDown(self):
        super(TestCommunicationStreamSendRecvAPI, self).tearDown()


if __name__ == '__main__':
    unittest.main()
