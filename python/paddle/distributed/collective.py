# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


def init_distributed_context(backend,
                             rank,
                             rank_num,
                             endpoints,
                             current_endpoint,
                             timeout=999999,
                             group_name='',
                             group_num=1,
                             prefix="",
                             iface="",
                             fs_path="",
                             fs_name="",
                             fs_ugi=""):
    """

    Initialize the default distributed context.

    Args:
        backend (str): The backend to use, one of 'nccl' or 'gloo'.
        rank (int): Rank of the current process starting from 0.
        rank_num (int): Number of processes in the group.
        timeout (int): Timeout in seconds for gloo only. 
        group_name (str): Name of the group.
        group_num (int): Number of groups generated.
        prefix (str): Path prefix to initialized gloo.
        iface (str): Network card used to initialized gloo.
        fs_path (str): A file system path used to initialized gloo.
        fs_name (str): A file system name used to initialized gloo.
        fs_ugi (str): A file system ugi (name and password) used to 
                      initialized gloo.

    Returns:
        None

    Examples:
        .. code-block:: python

        import paddle
        place = paddle.fluid.CUDAPlace(0)
        paddle.distributed.init_distributed_context('nccl', 2, 1)
    """
    global _default_backend
    if not backend in ['nccl', 'gloo']:
        raise ValueError("backend must be on of 'nccl' or 'gloo' in lowercase, "
                         "but the given %s." % backend)
    if _default_backend:
        raise RuntimeError("The default process group has been initialized.")

    _default_backend = backend
    _default_group.rank = rank
    _default_group.nranks = rank_num

    if rank_num < 2:
        raise ValueError(
            "At least 2 ranks are required to use distributed training.")

    if rank >= rank_num or rank < 0:
        raise ValueError("The value of rank must be in [0, rank_num)")

    if backend == 'nccl':
        prepare_context()
    elif backend == 'gloo':
        strategy = fluid.core.GlooParallelStrategy()
        strategy.rank = rank
        strategy.rank_num = rank_num
        strategy.prefix = ""
        strategy.iface = "lo"
        strategy.init_seconds = timeout
        strategy.run_seconds = timeout
        strategy.path = '/tmp/tmp0'
        strategy.fs_name = ""
        strategy.fs_ugi = ""
        gloo = fluid.core.GlooParallelContext(strategy)
        gloo.init()
    else:
        raise ValueError("Unknow backend: %s" % backend)
