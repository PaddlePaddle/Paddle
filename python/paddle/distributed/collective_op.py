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


def init_distributed_context(rank, rank_num, **kwargs):
    """

    Initialize the default distributed context.

    Args:
        rank (int): Rank of the current process starting from 0.
        rank_num (int): Number of processes in the group.
        
        The following optional args are supported:
            timeout (int): Timeout in seconds for gloo only. 
            prefix (str): Path prefix to initialized gloo.
            iface (str): Network card used to initialized gloo.
            path (str): A file system path used to initialized gloo, for single machine training, you need not to set it; for distributed training, its a path on hdfs.
            fs_name (str): A file system name used to initialized gloo.
            fs_ugi (str): A file system ugi (name and password) used to 
                          initialized gloo.

    Returns:
        None

    Examples:
        .. code-block:: python

        import paddle
        place = paddle.fluid.CUDAPlace(0)
        paddle.distributed.init_distributed_context(2, 1)
    """
    if rank_num < 2:
        raise ValueError(
            "At least 2 ranks are required to use distributed training.")

    if rank >= rank_num or rank < 0:
        raise ValueError("The value of rank must be in [0, rank_num)")

    # initialize nccl context
    prepare_context()

    # initialize gloo context
    path = kwargs.get('path', '/tmp/gloo_init_tmp')
    prefix = kwargs.get('prefix', '')
    iface = kwargs.get('iface', 'lo')
    timeout = int(kwargs.get('timeout', '9999999'))
    fs_name = kwargs.get('fs_name', '')
    fs_ugi = kwargs.get('fs_ugi', '')
    strategy = fluid.core.GlooParallelStrategy()
    strategy.rank = rank
    strategy.rank_num = rank_num
    strategy.prefix = prefix
    strategy.iface = iface
    strategy.init_seconds = timeout
    strategy.run_seconds = timeout
    strategy.path = path
    strategy.fs_name = fs_name
    strategy.fs_ugi = fs_ugi
    gloo = fluid.core.GlooParallelContext(strategy)
    gloo.init()
