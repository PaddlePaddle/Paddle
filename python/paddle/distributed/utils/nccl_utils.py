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


from paddle.base import core


def get_nccl_version_str(ver):
    if ver >= 10000:
        NCCL_MAJOR_VERSION = int(ver // 10000)
        ver = ver % 10000
    else:
        NCCL_MAJOR_VERSION = int(ver // 1000)
        ver = ver % 1000

    NCCL_MINOR_VERSION = int(ver // 100)
    NCCL_PATCH_VERSION = int(ver % 100)

    return f"{NCCL_MAJOR_VERSION}.{NCCL_MINOR_VERSION}.{NCCL_PATCH_VERSION}"


def check_nccl_version_for_p2p():
    nccl_version = core.nccl_version()
    nccl_version_str = get_nccl_version_str(nccl_version)
    nccl_version_baseline = 2804
    assert nccl_version >= nccl_version_baseline, (
        "The version of NCCL is required to be at least v2.8.4 while training with "
        f"pipeline/MoE parallelism, but we found v{nccl_version_str}. The previous version of NCCL has "
        "some bugs in p2p communication, and you can see more detailed description "
        "about this issue from ReleaseNotes of NCCL v2.8.4 "
        "(https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-8-4.html#rel_2-8-4)."
    )


def check_nccl_version_for_bf16():
    nccl_version = core.nccl_version()
    nccl_version_baseline = 21000
    return nccl_version >= nccl_version_baseline
