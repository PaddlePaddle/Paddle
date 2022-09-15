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
import test_collective_api_base as test_base


class TestCollectiveIsendIrecvAPI(test_base.TestDistBase):

    def _setup_config(self):
        pass

    def test_isend_irecv_nccl_dygraph(self):
        dtypes_to_test = [
            'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8',
            'bool'
        ]
        for dtype in dtypes_to_test:
            self.check_with_place("collective_isend_irecv_api_dygraph.py",
                                  "sendrecv",
                                  "nccl",
                                  static_mode="0",
                                  dtype=dtype)


if __name__ == '__main__':
    unittest.main()
