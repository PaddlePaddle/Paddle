# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import shutil
import subprocess

import paddle
from paddle.distributed.utils.log_utils import get_logger

logger = get_logger("INFO", "root")

SUCCESS_CODE = 0
FAIL_CODE = 1


def _get_cpu_info(numa_id):
    """
    get cpu info from lscpu
    """

    def _process_raw_cpu_info(i):
        processed_cpu_info = []
        cpu_ranges = i.split(',')
        for cpu_range in cpu_ranges:
            start, end = int(cpu_range.split("-")[0]), int(
                cpu_range.split("-")[1]
            )
            processed_cpu_info.extend(list(range(start, end + 1)))
        return processed_cpu_info

    try:
        cpus = None
        cmd = ["lscpu"]
        output = subprocess.check_output(cmd).decode("utf-8").split(os.linesep)
        numa_key = f"node{numa_id}"
        for line in output:
            if line.find(numa_key) >= 0:
                raw_cpu_info = line.strip().split()[3]
                cpus = _process_raw_cpu_info(raw_cpu_info)
                break
        return cpus
    except Exception as e:
        logger.warn(f"_get_cpu_info failed, reason:{e}")
        return None


def _has_nvidia_smi():
    """
    check if nvidia-smi is available
    """
    return shutil.which("nvidia-smi")


def _get_gpu_device(local_rank):
    """
    get currently used gpu physical device id
    """
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is None or cuda_visible_devices == "":
        return str(local_rank)
    cuda_visible_devices = cuda_visible_devices.split(',')
    if len(cuda_visible_devices) <= local_rank:
        return None
    return cuda_visible_devices[local_rank]


def _get_gpu_numa_info(gpu_id):
    """
    get gpu numa info from nvidia-smi
    """
    try:
        cmd = ["nvidia-smi", "topo", "-C", "-i", gpu_id]
        output = subprocess.check_output(cmd, timeout=3).decode("utf-8")
        numa_id = output.strip().split()[-1]
        return numa_id
    except Exception as e:
        logger.warn(f"_get_cpu_info failed, reason:{e}")
        return None


def set_affinity_gpu():
    """
    set affinity for gpu
    """
    if not _has_nvidia_smi():
        logger.warn(
            "nvidia-smi is not available, set_affinity is aborted, plz check your environment."
        )
        return FAIL_CODE
    local_rank = max(int(os.getenv("PADDLE_LOCAL_RANK", "0")), 0)
    device_id = _get_gpu_device(local_rank)
    if device_id is None:
        logger.warn(
            "Failed to get device id from cuda_visible_devices, set_affinity is aborted, plz check your environment."
        )
        return FAIL_CODE
    numa_id = _get_gpu_numa_info(device_id)
    if numa_id is None:
        logger.warn(
            "Failed to get numa info, set_affinity is aborted, plz check your environment."
        )
        return FAIL_CODE
    if numa_id == "N/A":
        logger.warn(
            "nvidia-smi topo return numa id as N/A, set_affinity is aborted, plz check your environment. (Notice: This is expected behavior when executed on single numa node environment)"
        )
        return FAIL_CODE
    affinity_mask = _get_cpu_info(numa_id)
    if affinity_mask is None:
        logger.warn(
            "Failed to get cpu info, set_affinity is aborted, plz check your environment."
        )
        return FAIL_CODE
    affinity = os.sched_getaffinity(0)
    logger.info(f"Check affinity before setting: {affinity}")
    os.sched_setaffinity(0, affinity_mask)
    affinity = os.sched_getaffinity(0)
    logger.info(f"check affinity after setting: {affinity}")
    return SUCCESS_CODE


def set_affinity():
    if paddle.device.is_compiled_with_cuda():
        return set_affinity_gpu()
    else:
        # TODO(@gexiao): supports other devices if needed
        logger.warn("Currently set_affinity only supports gpu env.")
        return FAIL_CODE
