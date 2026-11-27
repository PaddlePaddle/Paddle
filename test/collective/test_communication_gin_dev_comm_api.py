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

import os
import unittest

import communication_gin_dev_comm_api_dygraph as gin_dev_comm_case
import test_communication_api_base as test_base

import paddle
from paddle.base import core

# The NCCL device (LSA/GIN) API that the GIN primitives are built on.
_MIN_NCCL_VERSION = 23007


def _build_skip_reason():
    """Why this build or machine cannot have the GIN primitives, or None.

    These are the conditions of the WITH_GIN_OVERLAP gate in
    cmake/configure.cmake that Python can see, plus the GPU. The gate also
    needs nccl_device.h at build time, which only the run itself can tell:
    without it the primitives raise Unavailable and the test skips then.
    """
    if not paddle.is_compiled_with_cuda() or paddle.is_compiled_with_rocm():
        return "the GIN primitives need a CUDA build"
    cuda_major = int(str(paddle.version.cuda()).split(".")[0])
    if cuda_major < 13:
        return "the GIN primitives need Paddle built with CUDA 13.0 or newer"
    if not any(arch >= 90 for arch in paddle.version.cuda_archs()):
        return "the GIN primitives need Paddle built for sm_90 or newer"
    if core.nccl_version() < _MIN_NCCL_VERSION:
        return "the GIN primitives need NCCL 2.30.7 or newer"
    if paddle.device.cuda.device_count() < 2:
        return "number of GPUs is not enough"
    for device_id in range(2):
        major, _ = paddle.device.cuda.get_device_capability(device_id)
        if major < 9:
            return "the GIN primitives need GPUs of compute capability 9.0+"
    return None


class TestCommunicationGinDevCommAPI(test_base.CommunicationTestDistBase):
    def setUp(self):
        # Skipped from setUp, like test_communication_zero_sm_api, so that the
        # check runs the same way as the base class' own device-count skip.
        skip_reason = _build_skip_reason()
        if skip_reason is not None:
            self.skipTest(skip_reason)
        super().setUp(num_of_devices=2)

    def test_gin_dev_comm_api(self):
        skip_file = os.path.join(self._log_dir.name, "gin_unavailable.txt")
        self.run_test_case(
            "communication_gin_dev_comm_api_dygraph.py",
            user_defined_envs={gin_dev_comm_case.SKIP_FILE_ENV: skip_file},
        )
        if os.path.exists(skip_file):
            with open(skip_file) as reason:
                self.skipTest(reason.read())

    def tearDown(self):
        super().tearDown()


if __name__ == "__main__":
    unittest.main()
