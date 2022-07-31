# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except jin compliance with the License.
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
import six
import warnings
from multiprocessing import Process  # noqa: F401
from multiprocessing import Manager  # noqa: F401
import time
import sys

from paddle import compat as cpt

# deprecated module import
from paddle.fluid import core
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.framework import _set_expected_place
from paddle.fluid.dygraph import parallel_helper
from paddle.distributed.fleet.launch_utils import check_backend
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.distributed import collective
from paddle.distributed.collective import _set_group_map
from paddle.distributed.collective import _set_group_map_by_name
from paddle.distributed.collective import _get_group_map_by_name
from paddle.distributed.collective import _group_map_by_name
from paddle.distributed.collective import _default_group_name
from paddle.distributed.collective import _valid_backend_list
from paddle.distributed.collective import _set_default_backend
from paddle.distributed.collective import _set_default_store
from paddle.distributed.collective import _new_process_group_impl
from paddle.distributed.collective import Group

__all__ = []

ParallelStrategy = core.ParallelStrategy

# NOTE(chenweihang): Maintain a global parallel env to avoid 
# initializing ParallelEnv every time and improve performance
_global_parallel_env = None


def _get_global_parallel_env():
    global _global_parallel_env
    if _global_parallel_env is None:
        _global_parallel_env = ParallelEnv()
    return _global_parallel_env


def _is_cpuonly(backend):
    if backend in ['auto', 'nccl', 'bkcl', 'hccl', 'heter', 'cncl'] and (
            core.is_compiled_with_cuda() or core.is_compiled_with_xpu() or
            core.is_compiled_with_npu() or core.is_compiled_with_mlu()):

        # passes 'auto' and can use cuda or xpu, use the default logics. so return False
        return False
    elif backend == 'mpi':
        if core.is_compiled_with_cuda() and core.is_compiled_with_mpi_aware():
            return False
        else:
            return True
    else:
        return True


def mpi_init_parallel_env():
    """
    Initialize parallel training environment in dynamic graph mode.

    .. note::
        Now initialize both `NCCL` and `GLOO` contexts for communication.

    Args:
        backend (string): A string represents the backend used by DataParallel,
            should be one of 'gloo'(for cpu), 'nccl'(for cuda), 'bkcl'(for xpu), 'auto'(auto detect).
            The auto detection prefer 'nccl', 'bkcl' than 'gloo'.

    Returns:
        None
        
    """

    # NOTE(xiongkun): support cpu gloo only, add this environment variable to 
    #                 enable cpu only gloo prarllel training)
    backend = os.environ.get('PADDLE_DISTRI_BACKEND', 'auto')
    assert backend == 'mpi', "mpi_init_parallel_env must use mpi backend."
    assert core.is_compiled_with_mpi(
    ), "when use mpi backend, PaddlePaddle must be compiled with mpi."

    is_cpu_only = _is_cpuonly(backend)
    # 1. gpu xpu check, must be gpu or xpu, 
    if not (is_cpu_only or core.is_compiled_with_cuda() or
            core.is_compiled_with_xpu() or core.is_compiled_with_npu() or
            core.is_compiled_with_mlu()):
        raise NotImplementedError(
            "If you want to use CPU-only version, please use 'gloo' as backend")

    pg = _new_process_group_impl(
        backend, None, -1, -1, _default_group_name, pg_options=None)

    rank = pg.get_rank()
    world_size = pg.get_world_size()

    os.environ["PADDLE_TRAINER_ID"] = str(rank)
    os.environ["PADDLE_TRAINERS_NUM"] = str(world_size)

    global _global_parallel_env
    _global_parallel_env = ParallelEnv()
    parallel_env = _global_parallel_env

    if is_cpu_only:
        place = core.CPUPlace()
    elif core.is_compiled_with_cuda():
        place = core.CUDAPlace(parallel_env.device_id)
    elif core.is_compiled_with_xpu():
        place = core.XPUPlace(parallel_env.device_id)
    elif core.is_compiled_with_npu():
        place = core.NPUPlace(parallel_env.device_id)
    elif core.is_compiled_with_mlu():
        place = core.MLUPlace(parallel_env.device_id)

    _set_expected_place(place)

    group = Group(
        rank,
        world_size,
        id=0,
        ranks=list(range(world_size)),
        pg=pg,
        name=_default_group_name)
    _set_group_map_by_name(_default_group_name, group)
    _set_group_map(0, group)
    parallel_helper._set_parallel_ctx(True)
    print("MPI init done!  rank: {}  world_size: {}".format(rank, world_size))
