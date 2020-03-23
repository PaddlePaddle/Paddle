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
import os
import sys
import time
import math
import socket
import contextlib
from contextlib import closing
from six import string_types
import numpy as np
from collections import OrderedDict
from paddle import fluid
import paddle.fluid.unique_name as nameGen
from paddle.fluid import core

from paddle.fluid import framework
from paddle.fluid.layers import collective
from paddle.fluid.dygraph import to_variable, no_grad, layers
from paddle.fluid.framework import Variable
from paddle.fluid.executor import global_scope

from paddle.fluid.dygraph.parallel import Env, DataParallel, ParallelStrategy
from paddle.fluid.layers.collective import _c_allreduce, _c_allgather, _c_broadcast, \
                                            _c_sync_comm_stream, _c_sync_calc_stream
from paddle.fluid.io import BatchSampler, DataLoader

__parallel_context_init = False

class DistributedBatchSampler(BatchSampler):
    """Sampler that restricts data loading to a subset of the dataset.

    In such case, each process can pass a DistributedBatchSampler instance 
    as a DataLoader sampler, and load a subset of the original dataset that 
    is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.
        
    Args:
        data_source: this could be a `fluid.io.Dataset` implement
                     or other python object which implemented
                     `__len__` for BatchSampler to get sample
                     number of data source.
        batch_size(int): sample indice number in a mini-batch indices.
        shuffle(bool): whther to shuffle indices order before genrating
            batch indices. Default False.
        drop_last(bool): whether drop the last incomplete batch dataset size
            is not divisible by the batch size. Default False
    """

    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False):
        self.dataset = dataset

        assert isinstance(batch_size, int) and batch_size > 0, \
                "batch_size should be a positive integer"
        self.batch_size = batch_size
        assert isinstance(shuffle, bool), \
                "shuffle should be a boolean value"
        self.shuffle = shuffle
        assert isinstance(drop_last, bool), \
                "drop_last should be a boolean number"
        self.drop_last = drop_last
        self.nranks = get_nranks()
        self.local_rank = get_local_rank()
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.nranks))
        self.total_size = self.num_samples * self.nranks

    def __iter__(self):
        num_samples = len(self.dataset)
        indices = np.arange(num_samples).tolist()
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        if self.shuffle:
            np.random.RandomState(self.epoch).shuffle(indices)
            self.epoch += 1
        # subsample
        indices = indices[self.local_rank * self.num_samples: 
                            (self.local_rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples
        _sample_iter = iter(indices)

        batch_indices = []
        for idx in _sample_iter:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices

    def __len__(self):
        num_samples = self.num_samples
        num_samples += int(not self.drop_last) * (self.batch_size - 1)
        return num_samples // self.batch_size


def _all_gather(x, nranks, ring_id=0, use_calc_stream=True):
    return _c_allgather(x, nranks, ring_id=ring_id, use_calc_stream=use_calc_stream)


def get_local_rank():
    return Env().local_rank


def get_nranks():
    return Env().nranks


def wait_server_ready(endpoints):
    assert not isinstance(endpoints, string_types)
    while True:
        all_ok = True
        not_ready_endpoints = []
        for ep in endpoints:
            ip_port = ep.split(":")
            with closing(
                    socket.socket(socket.AF_INET,
                                  socket.SOCK_STREAM)) as sock:
                sock.settimeout(2)
                result = sock.connect_ex((ip_port[0], int(ip_port[1])))
                if result != 0:
                    all_ok = False
                    not_ready_endpoints.append(ep)
        if not all_ok:
            sys.stderr.write("server not ready, wait 3 sec to retry...\n")
            sys.stderr.write("not ready endpoints:" + str(
                not_ready_endpoints) + "\n")
            sys.stderr.flush()
            time.sleep(3)
        else:
            break


def init_communicator(program, rank, nranks, wait_port,
                     current_endpoint, endpoints):
    if nranks < 2:
        return
    other_endpoints = endpoints[:]
    other_endpoints.remove(current_endpoint)
    if rank == 0 and wait_port:
        wait_server_ready(other_endpoints)
    block = program.global_block()
    nccl_id_var = block.create_var(
        name=nameGen.generate('nccl_id'),
        persistable=True,
        type=core.VarDesc.VarType.RAW)

    block.append_op(
        type='c_gen_nccl_id',
        inputs={},
        outputs={'Out': nccl_id_var},
        attrs={
            'rank': rank,
            'endpoint': current_endpoint,
            'other_endpoints': other_endpoints
        })

    block.append_op(
        type='c_comm_init',
        inputs={'X': nccl_id_var},
        outputs={},
        attrs={
            'nranks': nranks,
            'rank': rank,
            'ring_id': 0,
        })


def prepare_distributed_context(place=None):
    if place is None:
        place = fluid.CUDAPlace(Env().dev_id) if Env().nranks > 1 \
            else fluid.CUDAPlace(0)
    
    strategy = ParallelStrategy()
    strategy.nranks = Env().nranks
    strategy.local_rank = Env().local_rank
    strategy.trainer_endpoints = Env().trainer_endpoints
    strategy.current_endpoint = Env().current_endpoint

    if strategy.nranks < 2:
        return

    global __parallel_context_init

    if not __parallel_context_init and isinstance(place, core.CUDAPlace):
        def _init_context():
            communicator_prog = framework.Program()
            init_communicator(communicator_prog, strategy.local_rank, strategy.nranks, 
                            True, strategy.current_endpoint, strategy.trainer_endpoints)
            exe = fluid.Executor(place)
            exe.run(communicator_prog)

        if fluid.in_dygraph_mode():
            fluid.disable_dygraph()
            _init_context()
            fluid.enable_dygraph(place)
        else:
            _init_context()

    else:
        assert ("Only support CUDAPlace for now.")

    __parallel_context_init = True
    return strategy


class DistributedDataParallel(DataParallel):
    def __init__(self, layers, strategy=None):
        if strategy is None:
            strategy = ParallelStrategy()
            strategy.nranks = Env().nranks
            strategy.local_rank = Env().local_rank
            strategy.trainer_endpoints = Env().trainer_endpoints
            strategy.current_endpoint = Env().current_endpoint

        super(DistributedDataParallel, self).__init__(layers, strategy)

    @no_grad
    def apply_collective_grads(self):
        """
        AllReduce the Parameters' gradient.
        """
        if not self._is_data_parallel_mode():
            return

        grad_var_set = set()
        grad_vars = []
        for param in self._layers.parameters():
            # NOTE(zcd): The grad_ivar maybe no generated.
            if param.trainable and param._grad_ivar():
                g_var = param._grad_ivar()
                grad_vars.append(g_var)
                assert g_var not in grad_var_set
                grad_var_set.add(g_var)

        mega_bytes = 128 * 1024 * 1024
        group_idx = 0
        memory_counter = 0
        grad_var_groups = OrderedDict()
        dtype = grad_vars[0].dtype
        for g_var in grad_vars:
            # Note: the dtype of the same group should be the same.
            bytes = np.prod(g_var.shape) * core.size_of_dtype(g_var.dtype)
            if memory_counter < mega_bytes and dtype == g_var.dtype:
                memory_counter += bytes
            else:
                memory_counter = bytes
                group_idx += 1
            grad_var_groups.setdefault(group_idx, []).append(g_var)

        coalesced_grads_and_vars = self._coalesce_tensors(grad_var_groups)

        for coalesced_grad, _, _ in coalesced_grads_and_vars:
            collective._c_allreduce(coalesced_grad, coalesced_grad, use_calc_stream=True)

        self._split_tensors(coalesced_grads_and_vars)
