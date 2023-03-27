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

import logging
import subprocess


def get_nccl_version_str():
    nccl_version_str = subprocess.check_output(
        r"ldconfig -v | grep 'libnccl.so' | tail -n1 | sed -r 's/^.*\.so\.//'",
        stderr=subprocess.DEVNULL,
        shell=True,
    ).decode('utf-8')

    # NOTE: This is a hacking method to get nccl version, but it will return None
    # if current platform is not Linux. So we only check nccl version for Linux
    # platform while training with pipeline parallelism.
    if nccl_version_str:
        nccl_version_str = nccl_version_str.replace("\n", "")

    return nccl_version_str


def check_nccl_version_for_p2p():
    nccl_version_str = get_nccl_version_str()
    if nccl_version_str:
        nccl_version_str = nccl_version_str.replace("\n", "")
        nccl_version_int = [int(s) for s in nccl_version_str.split(".")]
        nccl_version_baseline = [2, 8, 4]
        assert nccl_version_int >= nccl_version_baseline, (
            "The version of NCCL is required to be at least v2.8.4 while training with "
            "pipeline/MoE parallelism, but we found v{}. The previous version of NCCL has "
            "some bugs in p2p communication, and you can see more detailed description "
            "about this issue from ReleaseNotes of NCCL v2.8.4 "
            "(https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-8-4.html#rel_2-8-4).".format(
                nccl_version_str
            )
        )
    else:
        logging.warning("No version for NCCL library found!")


def check_nccl_version_for_bf16():
    nccl_version_str = get_nccl_version_str()
    if nccl_version_str:
        nccl_version_str = nccl_version_str.replace("\n", "")
        nccl_version_int = [int(s) for s in nccl_version_str.split(".")]
        nccl_version_baseline = [2, 10, 0]
        return nccl_version_int >= nccl_version_baseline

    return False
