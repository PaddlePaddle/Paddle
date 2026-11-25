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

"""Sharding stage 1 on two GPUs against the run without the option under
test; the cases are in sharding_gin_overlap_dygraph.py.

TestShardingStage1FusedAllGather (any 2-GPU CUDA build) checks
param_allgather_fuse_mb and muon_owner_partition_mb without gin_overlap.
TestShardingGinOverlap needs a build with gin_overlap (WITH_GIN_OVERLAP) and
two GPUs of compute capability 9.0 or newer, and is skipped otherwise.
"""

import unittest

import sharding_gin_overlap_dygraph as sharding_case
import test_communication_api_base as test_base

import paddle
from paddle.base import core

_SCRIPT = "sharding_gin_overlap_dygraph.py"
# The NCCL device (LSA/GIN) API that gin_overlap is built on.
_MIN_NCCL_VERSION = 23007


def _gin_overlap_skip_reason():
    """Why this build or machine cannot run gin_overlap, or None: the
    conditions of the WITH_GIN_OVERLAP gate in cmake/configure.cmake that
    Python can see, the gin_overlap bindings, and the GPUs."""
    if not paddle.is_compiled_with_cuda() or paddle.is_compiled_with_rocm():
        return "gin_overlap needs a CUDA build"
    cuda_major = int(str(paddle.version.cuda()).split(".")[0])
    if cuda_major < 13:
        return "gin_overlap needs Paddle built with CUDA 13.0 or newer"
    if not any(arch >= 90 for arch in paddle.version.cuda_archs()):
        return "gin_overlap needs Paddle built for sm_90 or newer"
    if core.nccl_version() < _MIN_NCCL_VERSION:
        return "gin_overlap needs NCCL 2.30.7 or newer"
    if not hasattr(core, "gin_overlap"):
        return "this Paddle was built without gin_overlap (WITH_GIN_OVERLAP)"
    if paddle.device.cuda.device_count() < 2:
        return "number of GPUs is not enough"
    for device_id in range(2):
        major, _ = paddle.device.cuda.get_device_capability(device_id)
        if major < 9:
            return "gin_overlap needs GPUs of compute capability 9.0+"
    return None


class TestShardingStage1FusedAllGather(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=2)

    def test_fused_all_gather_and_owner_partition(self):
        self.run_test_case(
            _SCRIPT,
            user_defined_envs={
                sharding_case.CASE_ENV: sharding_case.PART_A_CASE
            },
        )

    def tearDown(self):
        super().tearDown()


class TestShardingGinOverlap(test_base.CommunicationTestDistBase):
    def setUp(self):
        # Skipped from setUp, like test_communication_gin_dev_comm_api, so
        # that the check runs the same way as the base class' own
        # device-count skip.
        skip_reason = _gin_overlap_skip_reason()
        if skip_reason is not None:
            self.skipTest(skip_reason)
        super().setUp(num_of_devices=2)

    def _run_case(self, case):
        self.run_test_case(
            _SCRIPT, user_defined_envs={sharding_case.CASE_ENV: case}
        )

    def test_adamw_default_config(self):
        self._run_case(sharding_case.GIN_ADAMW_CASE)

    def test_adamw_recv_pool_slot_reuse(self):
        self._run_case(sharding_case.GIN_ADAMW_RECV_POOL_WRAP_CASE)

    def test_muon_default_config(self):
        self._run_case(sharding_case.GIN_MUON_CASE)

    def tearDown(self):
        super().tearDown()


if __name__ == "__main__":
    unittest.main()
