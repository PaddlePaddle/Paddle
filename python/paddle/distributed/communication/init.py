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

import paddle.distributed as dist
import paddle.fluid.framework as framework
import paddle.fluid.core as core
import paddle.distributed.communication.utils as utils
import paddle.distributed.collective as collective


def init_process_group(backend, store, rank, world_size):
    """
    Initialize the default distributed process group. This will also initialize the distributed package.

    Args:
        backend (string): A string represents the backend to use, should be one of 'gloo', 'nccl'.
        store (Store): A key/value store accessible to all workers, only 'TCPStore' is supported for now.
        rank (int): Represent the rank of the current process.
        world_size (int): The total number of processes participating in this distributed job.

    Returns:
        Group: The global group instance.

    Warning:
        This API is only supported for dygraph mode for now.

    Examples:
        .. code-block:: python
        # Execute this script using distributed launch with two devices.
        import paddle
        import paddle.distributed as dist

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        is_master = (rank == 0)
        store = dist.TCPStore("localhost", 12345, is_master, world_size)

        group = dist.init_process_group("nccl", store, rank, world_size)
    """

    valid_backend_list = utils._ProcessGroupManager.valid_backend_list
    default_group_name = utils._ProcessGroupManager.default_group_name
    global_group_id = utils._ProcessGroupManager.global_group_id

    if backend not in valid_backend_list:
        raise NotImplementedError(
            "Valid backends are {}. But input {} as parameter.".format(
                valid_backend_list, backend))

    if dist.is_initialized():
        raise RuntimeError(
            "The default process group is already initialized. You are trying to init it twice."
        )

    if not framework.in_dygraph_mode():
        raise NotImplementedError(
            "This API is only supported in dygraph for now.")

    utils._set_place(backend)
    collective._set_default_store(store)

    pg = collective._new_process_group_impl(backend,
                                            store,
                                            rank,
                                            world_size,
                                            default_group_name,
                                            pg_options=None)

    ranks = list(range(world_size))
    group = collective.Group(rank,
                             world_size,
                             id=global_group_id,
                             ranks=ranks,
                             pg=pg,
                             name=default_group_name)
    collective._set_group_map_by_name(default_group_name, group)
    collective._set_group_map(global_group_id, group)

    dist.barrier(group=group)
    return group
