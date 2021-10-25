# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.framework import _set_expected_place
from paddle.fluid.dygraph import parallel_helper
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.distributed.fleet.base.private_helper_function import wait_server_ready  # noqa: F401

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


def _start_kv_server(port, http_server_d, size):
    from paddle.distributed.fleet.utils.http_server import KVServer
    http_server = KVServer(int(port), size=size)
    http_server.start()
    wait_seconds = 3
    while http_server_d.get("running", False) or not http_server.should_stop():
        time.sleep(wait_seconds)
    http_server.stop()


def _is_cpuonly(backend):
    check_backend(backend)
    if backend in ['auto', 'nccl', 'bkcl', 'hccl', 'heter'] and (
            core.is_compiled_with_cuda() or core.is_compiled_with_xpu() or
            core.is_compiled_with_npu()):
        # passes 'auto' and can use cuda or xpu or npu, use the default logics.
        # So return False.
        return False
    else:
        return True


def _check_var_exists(var_name):
    var = os.environ.get(var_name, None)
    if var is None:
        raise ValueError("paddle.distributed initialize error, "
                         "environment variable %s is needed, but not set." %
                         var_name)


def init_parallel_env():
    """
    Initialize parallel training environment in dynamic graph mode.

    Returns:
        None
        
    Examples:
        .. code-block:: python
            # required: gpu
            import paddle
            import paddle.nn as nn
            import paddle.optimizer as opt
            import paddle.distributed as dist

            class LinearNet(nn.Layer):
                def __init__(self):
                    super(LinearNet, self).__init__()
                    self._linear1 = nn.Linear(10, 10)
                    self._linear2 = nn.Linear(10, 1)
                    
                def forward(self, x):
                    return self._linear2(self._linear1(x))

            def train():
                # 1. initialize parallel environment
                dist.init_parallel_env()

                # 2. create data parallel layer & optimizer
                layer = LinearNet()
                dp_layer = paddle.DataParallel(layer)

                loss_fn = nn.MSELoss()
                adam = opt.Adam(
                    learning_rate=0.001, parameters=dp_layer.parameters())

                # 3. run layer
                inputs = paddle.randn([10, 10], 'float32')
                outputs = dp_layer(inputs)
                labels = paddle.randn([10, 1], 'float32')
                loss = loss_fn(outputs, labels)
                
                loss.backward()

                adam.step()
                adam.clear_grad()

            if __name__ == '__main__':
                dist.spawn(train)
    """

    # 0. Get env & check world size
    global _global_parallel_env
    # when call init_parallel_env, need update `_global_parallel_env`
    _global_parallel_env = ParallelEnv()
    parallel_env = _global_parallel_env
    # if not parallel, `init_parallel_env` do nothing
    if parallel_env.world_size < 2:
        warnings.warn(
            "Currently not a parallel execution environment, "
            "`paddle.distributed.init_parallel_env` will not do anything.")
        return

    # 1. Check backend
    backend = os.getenv("PADDLE_DISTRI_BACKEND", "auto")
    print("### DEBUG ### backend is: ", backend)
    # NOTE(liubo48): this env should be set/detected in distributed.launch.
    nranks_per_node = int(os.getenv("PADDLE_LOCAL_TRAINERS_NUM", "2"))
    assert nranks_per_node != 0, "The nranks of node must not be zero."
    is_cpu_only = _is_cpuonly(backend)
    enable_gloo = is_cpu_only or (backend == "heter")

    # 2. Check device
    if not (is_cpu_only or core.is_compiled_with_cuda() or
            core.is_compiled_with_xpu() or core.is_compiled_with_npu()):
        raise NotImplementedError(
            "If you want to use CPU-only version, please use 'gloo' as backend")

    # 3. Check env
    if not is_cpu_only and core.is_compiled_with_cuda():
        _check_var_exists("FLAGS_selected_gpus")
    elif not is_cpu_only and core.is_compiled_with_xpu():
        _check_var_exists('FLAGS_selected_xpus')
    elif not is_cpu_only and core.is_compiled_with_npu():
        _check_var_exists('FLAGS_selected_npus')

    _check_var_exists("PADDLE_TRAINER_ID")
    _check_var_exists("PADDLE_CURRENT_ENDPOINT")
    _check_var_exists("PADDLE_TRAINERS_NUM")
    _check_var_exists("PADDLE_TRAINER_ENDPOINTS")

    # 4: Start httpsever for gloo context initialzation if required.
    if enable_gloo:
        # NOTE(liubo48): this is the requirement for heter backend.
        # The number of cards on each heter node need to be same,
        # and only use local rank 0 to do gloo communication.
        if (backend == 'heter'):
            gloo_worker_size = int(parallel_env.world_size / nranks_per_node)
        else:
            gloo_worker_size = parallel_env.world_size
        assert gloo_worker_size >= 2, 'size of gloo workers should be >= 2.'
        ep_rank_0 = parallel_env.trainer_endpoints[0].split(":")
        manager = Manager()
        # glboal dict to store status
        http_server_d = manager.dict()
        http_server_d["running"] = False
        if parallel_env.rank == 0:
            # The scope for worker used by http server is '_worker'
            worker_size = {'_worker': gloo_worker_size}
            http_server = Process(
                target=_start_kv_server,
                args=(int(ep_rank_0[1]), http_server_d, worker_size))
            http_server.daemon = True
            http_server_d["running"] = True
            http_server.start()

    # 5. Construct ParallelStrategy
    strategy = ParallelStrategy()
    if parallel_helper._is_parallel_ctx_initialized():
        warnings.warn("The parallel environment has been initialized.")
    strategy.nranks = parallel_env.world_size
    strategy.local_rank = parallel_env.rank
    strategy.local_nranks = nranks_per_node
    strategy.trainer_endpoints = parallel_env.trainer_endpoints
    strategy.current_endpoint = parallel_env.current_endpoint
    strategy.nrings = parallel_env.nrings

    # NOTE(chenweihang): [ why config global place here? ]
    # the dygraph mode will be set to default mode,
    # users will not call `dygraph.guard` or `enable_dygraph`
    # directly, if they want to switch default place,
    # they need to call a function to change default place,
    # here just set correctly place to users
    if is_cpu_only:
        place = core.CPUPlace()
    elif core.is_compiled_with_cuda():
        place = core.CUDAPlace(parallel_env.device_id)
    elif core.is_compiled_with_xpu():
        place = core.XPUPlace(parallel_env.device_id)
    elif core.is_compiled_with_npu():
        place = core.NPUPlace(parallel_env.device_id)
    _set_expected_place(place)

    # 6. Initialize parallel context
    if is_cpu_only:
        parallel_helper._set_parallel_ctx(
            core.GLOOParallelContext(strategy, place))
    elif (backend == "heter"):
        # NOTE(liubo48): The construction of heter parallel context will
        # detect the target place by checking the compilation option.
        parallel_helper._set_parallel_ctx(
            core.HeterParallelContext(strategy, parallel_env.device_id))
    elif core.is_compiled_with_cuda():
        parallel_helper._set_parallel_ctx(
            core.NCCLParallelContext(strategy, place))
    elif core.is_compiled_with_xpu():
        parallel_helper._set_parallel_ctx(
            core.BKCLParallelContext(strategy, place))
    elif core.is_compiled_with_npu():
        parallel_helper._set_parallel_ctx(
            core.HCCLParallelContext(strategy, place))

    # NOTE(liubo48): During the gloo rendezvous stage,
    # other participants do not need to wait for rank 0;
    # and rank 0 here do not need to wait for other endpoints either.
    # other_endpoints = strategy.trainer_endpoints[:]
    # other_endpoints.remove(strategy.current_endpoint)
    # if not is_cpu_only and strategy.local_rank == 0:
    #     wait_server_ready(other_endpoints)

    parallel_helper._init_parallel_ctx()

    # 7: Stop httpsever for gloo context initialization if required.
    if enable_gloo and parallel_env.rank == 0:
        http_server_d["running"] = False
        http_server.join()


def get_rank():
    """
    Returns the rank of current trainer.

    Its value is equal to the value of the environment variable ``PADDLE_TRAINER_ID`` . 
    The default value is 0.

    Returns:
        (int) The rank of current trainer.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.distributed as dist

            # execute this command in terminal: export PADDLE_TRAINER_ID=0
            print("The rank is %d" % dist.get_rank())
            # The rank is 0
    """
    return _get_global_parallel_env().rank


def get_world_size():
    """
    Returns the number of trainers (number of processes participating in current job).

    Its value is equal to the value of the environment variable ``PADDLE_TRAINERS_NUM`` . 
    The default value is 1.

    Returns:
        (int) The number of trainers.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.distributed as dist

            # execute this command in terminal: export PADDLE_TRAINERS_NUM=4
            print("The world_size is %d" % dist.get_world_size())
            # The world_size is 4
    """
    return _get_global_parallel_env().world_size
