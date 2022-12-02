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
from collections import OrderedDict

import numpy as np

import paddle
from paddle import _legacy_C_ops
from paddle.fluid import core, framework
from paddle.fluid.dygraph import base as imperative_base
from paddle.fluid.framework import ParamBase, _in_legacy_dygraph


class ParallelEnv:
    """
    .. note::
        This API is not recommended, if you need to get rank and world_size,
        it is recommended to use ``paddle.distributed.get_rank()`` and
        ``paddle.distributed.get_world_size()`` .

    This class is used to obtain the environment variables required for
    the parallel execution of ``paddle.nn.Layer`` in dynamic mode.

    The parallel execution in dynamic mode needs to be started using ``paddle.distributed.launch``
    or ``paddle.distributed.spawn`` .

    Examples:
      .. code-block:: python

        import paddle
        import paddle.distributed as dist

        def train():
            # 1. initialize parallel environment
            dist.init_parallel_env()

            # 2. get current ParallelEnv
            parallel_env = dist.ParallelEnv()
            print("rank: ", parallel_env.rank)
            print("world_size: ", parallel_env.world_size)

            # print result in process 1:
            # rank: 1
            # world_size: 2
            # print result in process 2:
            # rank: 2
            # world_size: 2

        if __name__ == '__main__':
            # 1. start by ``paddle.distributed.spawn`` (default)
            dist.spawn(train, nprocs=2)
            # 2. start by ``paddle.distributed.launch``
            # train()
    """

    def __init__(self):
        self._rank = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        self._world_size = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
        self._device_type = str(os.getenv("PADDLE_XCCL_BACKEND", ""))

        # imperative only support one gpu or xpu
        if self._device_type != "":
            FLAGS_selected_custom_devices = 'FLAGS_selected_{}s'.format(
                self._device_type
            )
            selected_custom_devices = os.getenv(
                FLAGS_selected_custom_devices, "0"
            ).split(",")
            self._device_id = int(selected_custom_devices[0])
        else:
            if core.is_compiled_with_cuda():
                selected_gpus = os.getenv("FLAGS_selected_gpus", "0").split(",")
                self._device_id = int(selected_gpus[0])
            elif core.is_compiled_with_xpu():
                selected_xpus = os.getenv("FLAGS_selected_xpus", "0").split(",")
                self._device_id = int(selected_xpus[0])
            elif core.is_compiled_with_npu():
                selected_npus = os.getenv("FLAGS_selected_npus", "0").split(",")
                self._device_id = int(selected_npus[0])
            elif core.is_compiled_with_mlu():
                selected_mlus = os.getenv("FLAGS_selected_mlus", "0").split(",")
                self._device_id = int(selected_mlus[0])

        self._trainer_endpoints = os.getenv(
            "PADDLE_TRAINER_ENDPOINTS", ""
        ).split(",")
        self._current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT", "")
        self._nrings = int(os.getenv("FLAGS_nccl_nrings", "1"))
        assert (
            self._nrings > 0
        ), "nccl_nrings must be an integer greater than 0."
        assert (
            self._nrings < 9
        ), "nccl_nrings should be less than 9, which is enough in most scenarios."

    @property
    def rank(self):
        """
        Rank of current trainer.

        Its value is equal to the value of the environment variable ``PADDLE_TRAINER_ID`` . The default value is 0.

        Examples:
          .. code-block:: python

            # execute this command in terminal: export PADDLE_TRAINER_ID=0
            import paddle.distributed as dist

            env = dist.ParallelEnv()
            print("The rank is %d" % env.rank)
            # The rank is 0
        """
        return self._rank

    @property
    def world_size(self):
        """
        The number of trainers (number of processes participating in current job).

        Its value is equal to the value of the environment variable ``PADDLE_TRAINERS_NUM`` . The default value is 1.

        Examples:
          .. code-block:: python

            # execute this command in terminal: export PADDLE_TRAINERS_NUM=4
            import paddle.distributed as dist

            env = dist.ParallelEnv()
            print("The world_size is %d" % env.world_size)
            # The world_size is 4
        """
        return self._world_size

    @property
    def device_id(self):
        """
        The ID of selected GPU card for parallel training.

        Its value is equal to the value of the environment variable ``FLAGS_selected_gpus`` . The default value is 0.

        Examples:
          .. code-block:: python

            # execute this command in terminal: export FLAGS_selected_gpus=1
            import paddle.distributed as dist

            env = dist.ParallelEnv()
            print("The device id are %d" % env.device_id)
            # The device id are 1
        """
        return self._device_id

    @property
    def device_type(self):
        """
        The type of custom device for parallel training.

        Its value is equal to the value of the environment variable ``PADDLE_XCCL_BACKEND`` . The default value is None.

        """
        return self._device_type

    @property
    def current_endpoint(self):
        """
        The endpoint of current trainer, it is in the form of (node IP + port).

        Its value is equal to the value of the environment variable ``PADDLE_CURRENT_ENDPOINT`` . The default value is "".

        Examples:
          .. code-block:: python

            # execute this command in terminal: export PADDLE_CURRENT_ENDPOINT=127.0.0.1:6170
            import paddle.distributed as dist

            env = dist.ParallelEnv()
            print("The current endpoint are %s" % env.current_endpoint)
            # The current endpoint are 127.0.0.1:6170
        """
        return self._current_endpoint

    @property
    def trainer_endpoints(self):
        """
        The endpoints of all trainer nodes in the task,
        which are used to broadcast the NCCL ID when NCCL2 is initialized.

        Its value is equal to the value of the environment variable ``PADDLE_TRAINER_ENDPOINTS`` . The default value is "".

        Examples:
          .. code-block:: python

            # execute this command in terminal: export PADDLE_TRAINER_ENDPOINTS=127.0.0.1:6170,127.0.0.1:6171
            import paddle.distributed as dist

            env = dist.ParallelEnv()
            print("The trainer endpoints are %s" % env.trainer_endpoints)
            # The trainer endpoints are ['127.0.0.1:6170', '127.0.0.1:6171']
        """
        return self._trainer_endpoints

    @property
    def nrings(self):
        """
        Nrings of current trainer.

        Its value is equal to the value of the environment variable ``FLAGS_nccl_nrings`` . The default value is 1.

        Examples:
          .. code-block:: python

            # execute this command in terminal: export FLAGS_nccl_nrings=1
            import paddle.distributed as dist

            env = dist.ParallelEnv()
            print("The nrings is %d" % env.nrings)
            # the number of ring is 1
        """
        return self._nrings

    # [aliases] Compatible with old method names
    local_rank = rank
    nranks = world_size
    dev_id = device_id


def _coalesce_tensors(var_groups):
    coalesced_grads_and_grad_vars = []
    for group_id, grad_vars in var_groups.items():
        flattened_vars = []
        g_var_shapes = []
        for g_var in grad_vars:
            g_var_shapes.append(g_var.shape)
            flattened_vars.append(
                paddle.reshape(x=g_var, shape=[np.prod(g_var.shape)])
            )
        coalesced_grad = paddle.concat(flattened_vars)
        coalesced_grads_and_grad_vars.append(
            [coalesced_grad, grad_vars, g_var_shapes]
        )
    return coalesced_grads_and_grad_vars


@framework.dygraph_only
def _reshape_inplace(x, shape):
    x_shape = framework._varbase_creator(dtype=x.dtype)
    framework._dygraph_tracer().trace_op(
        type="reshape2",
        inputs={'X': x},
        outputs={'Out': x, 'XShape': x_shape},
        attrs={'shape': shape},
    )


@framework.dygraph_only
def _split_tensors(coalesced_grads_and_grad_vars):
    if _in_legacy_dygraph():
        for (
            coalesced_grad,
            origin_grad_vars,
            grad_shapes,
        ) in coalesced_grads_and_grad_vars:
            grad_var_len = [np.prod(g_shape) for g_shape in grad_shapes]
            framework._dygraph_tracer().trace_op(
                type='split',
                inputs={'X': coalesced_grad},
                outputs={'Out': origin_grad_vars},
                attrs={'sections': grad_var_len, 'axis': 0},
            )
            for g_var, g_shape in zip(origin_grad_vars, grad_shapes):
                _reshape_inplace(x=g_var, shape=g_shape)
                assert g_var.shape == g_shape
    elif framework.in_dygraph_mode():
        for (
            coalesced_grad,
            origin_grad_vars,
            grad_shapes,
        ) in coalesced_grads_and_grad_vars:
            grad_var_len = [np.prod(g_shape) for g_shape in grad_shapes]
            attrs = ()
            attrs += ('sections', grad_var_len)
            attrs += ('axis', 0)
            _legacy_C_ops.split(coalesced_grad, origin_grad_vars, *attrs)
            for g_var, g_shape in zip(origin_grad_vars, grad_shapes):
                g_var.reshape_(shape=g_shape)
                assert g_var.shape == g_shape


@imperative_base.no_grad
@framework.dygraph_only
def build_groups(vars, group_size):
    group_idx = 0
    memory_counter = 0
    var_groups = OrderedDict()
    dtype = vars[0].dtype

    for var in vars:
        bytes = np.prod(var.shape) * core.size_of_dtype(var.dtype)
        if memory_counter < group_size and dtype == var.dtype:
            memory_counter += bytes
        else:
            memory_counter = bytes
            dtype = var.dtype
            group_idx += 1
        var_groups.setdefault(group_idx, []).append(var)
    return _coalesce_tensors(var_groups)


@imperative_base.no_grad
@framework.dygraph_only
def sync_params_buffers(
    model, comm_group=None, src_rank=0, is_model_parallel=False
):
    model_vars = []
    for _, param in model._obtain_parameters_buffers().items():
        if not isinstance(param, (core.VarBase, core.eager.Tensor)):
            raise TypeError(
                "The data type of '%s' must be Varbase or eager.Tensor"
                % param.name
            )

        # is_distributed param not need to sync when in mp mode
        if isinstance(param, (ParamBase, core.eager.Tensor)):
            if is_model_parallel and param.is_distributed:
                continue

            # NOTE(shenliang03): Support situations that do not require synchronization parameters,
            # such as moe's expert parameters
            if getattr(param, "no_sync", False):
                continue
        if param.type == core.VarDesc.VarType.VOCAB:
            continue

        model_vars.append(param.detach())
    if len(model_vars) == 0:
        return

    # group size is 128M
    coalesced_vars = build_groups(model_vars, 128 * 1024 * 1024)

    for coalesced_var, _, _ in coalesced_vars:
        paddle.distributed.broadcast(
            coalesced_var, src=src_rank, group=comm_group, sync_op=True
        )

    for coalesced_var, origin_vars, var_shapes in coalesced_vars:
        var_len = [np.prod(v_shape) for v_shape in var_shapes]
        framework._dygraph_tracer().trace_op(
            type='split',
            inputs={'X': coalesced_var},
            outputs={'Out': origin_vars},
            attrs={'sections': var_len, 'axis': 0},
        )
