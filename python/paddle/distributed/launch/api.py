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

import os
"""
Usage:

from paddle.distributed.launch import api as launch_api
print(launch_api.global_size())

"""

# NOTE(kzq): launch now only support get information through static environ,
# we will support more method to update those information dynamically, later.


def job_id() -> str:
    """
    Get the unique id of the job.
    """

    key = "JOB_ID"
    return os.getenv(key, "default")


def pod_name() -> str:
    """
    Get the logical node/pod name.
    """

    key = "POD_NAME"
    return os.getenv(key, "")


def active() -> bool:
    """
    Indicate whether this process is run by launch.
    """

    key = "POD_NAME"
    if len(os.getenv(key, "")) == 6:
        return True
    else:
        return False


def master() -> str:
    """
    Get the logical node/pod name.
    """

    key = "PADDLE_MASTER"
    return os.getenv(key, "")


def global_size() -> int:
    """
    Global size of job, the total number of process/gpus.
    """

    key = "PADDLE_GLOBAL_SIZE"
    return int(os.getenv(key, 1))


def local_size() -> int:
    """
    Local size of job, the number of process/gpus in the same pod/node.
    """

    key = "PADDLE_LOCAL_SIZE"
    return int(os.getenv(key, 1))


def global_rank() -> int:
    """
    Global rank of the process, rank count all nodes/pods.
    """

    key = "PADDLE_GLOBAL_RANK"
    return int(os.getenv(key, 0))


def local_rank() -> int:
    """
    Local rank of the process, rank in the same node/pod.
    """

    key = "PADDLE_LOCAL_RANK"
    return int(os.getenv(key, 0))


def nnodes() -> int:
    """
    The number of logical nodes, the number of pods.
    """

    key = "PADDLE_NNODES"
    return int(os.getenv(key, 1))


rank = global_rank
size = global_size

__all__ = [
    job_id, pod_name, active, global_size, local_size, global_rank, local_size,
    nnodes, rank, size
]
