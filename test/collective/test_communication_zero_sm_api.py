# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import test_communication_api_base as test_base

import paddle
from paddle.base import core

# The individual entry points appeared earlier, but the hierarchical zero-SM
# collectives only work from NCCL 2.30.7 on, so that is what the feature is
# gated on.
_MIN_NCCL_VERSION = 23007


def _zero_sm_supported():
    if not paddle.is_compiled_with_cuda() or paddle.is_compiled_with_rocm():
        return False
    if not hasattr(core, "nccl_mem_alloc"):
        return False
    return core.nccl_version() >= _MIN_NCCL_VERSION


class TestCommunicationZeroSMAPI(test_base.CommunicationTestDistBase):
    def setUp(self):
        # Skipped from setUp rather than through a decorator so that the check
        # runs the same way as the base class' own device-count skip.
        if not _zero_sm_supported():
            self.skipTest(
                "zero-SM collectives need a CUDA build running against NCCL "
                "2.30.7 or newer"
            )
        super().setUp(num_of_devices=2)

    def test_zero_sm_api(self):
        self.run_test_case("communication_zero_sm_api_dygraph.py")

    def tearDown(self):
        super().tearDown()


if __name__ == "__main__":
    unittest.main()
