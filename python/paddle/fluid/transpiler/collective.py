#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import sys
import math
from functools import reduce

import collections
import six
import logging

import numpy as np

from ..core import VarDesc
from ..framework import Program, \
        default_main_program, default_startup_program

from .details import wait_server_ready

__all__ = ['SGD', 'LocalSGD']


class Collective(object):
    '''
    '''

    def __init__(self):
        self.global_ring_id = 0
        self.endpoints = None
        self.current_endpoint = None
        self.nranks = None
        self.rank = None
        self.startup_program = None
        self.main_program = None

    def transpile(self, startup_program, main_program, rank, endpoints,
                  current_endpoint, wait_port):
        # in case of '127.0.0.1:6700,127.0.0.1:6701,...'
        if isinstance(endpoints, str):
            endpoints = endpoints.split(',')

        self.startup_program = startup_program
        if startup_program is None:
            self.startup_program = default_startup_program()

        self.main_program = main_program
        if main_program is None:
            self.main_program = default_main_program()

        self.nranks = len(endpoints)
        if self.nranks == 1:
            raise ValueError('the number of endpoints must > 1')

        if rank < 0:
            raise ValueError('rank must >= 0')
        self.rank = rank

        if current_endpoint not in endpoints:
            raise ValueError('current endpoint %s is not in %s',
                             current_endpoint, str(endpoints))

        self.endpoints = endpoints
        self.current_endpoint = current_endpoint

        self.wait_port = wait_port

        self.startup_program._origin_program = self.startup_program.clone()
        self._transpile_startup_program()

        self.main_program._origin_program = self.main_program.clone()
        self._transpile_main_program()

    def _transpile_main_program(self):
        raise NotImplementedError('call the inherited method of subclasses')

    def _transpile_startup_program(self):
        self._init_communicator(self.startup_program, self.current_endpoint,
                                self.endpoints, self.rank, self.global_ring_id,
                                self.wait_port)
        self._broadcast_params()

    def _init_communicator(self, program, current_endpoint, endpoints, rank,
                           ring_id, wait_port):
        nranks = len(endpoints)
        other_endpoints = endpoints[:]
        other_endpoints.remove(current_endpoint)
        if rank == 0 and wait_port:
            wait_server_ready(other_endpoints)

        block = program.global_block()
        nccl_id_var = block.create_var(
            persistable=True, type=VarDesc.VarType.RAW)
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
            attrs={'nranks': nranks,
                   'rank': rank,
                   'ring_id': ring_id})

    def _broadcast_params(self):
        block = self.startup_program.global_block()
        for var in block.iter_parameters():
            block.append_op(
                type='c_broadcast',
                inputs={'X': var},
                outputs={'Out': var},
                attrs={'ring_id': self.global_ring_id,
                       'root': 0})
        block.append_op(
            type='c_sync_comm_stream', attrs={'ring_id': self.global_ring_id})

    def _is_optimizer_op(self, op):
        return 'Param' in op.input_names and 'Grad' in op.input_names


class SGD(Collective):
    '''
    '''

    def __init__(self):
        Collective.__init__(self)

    def _transpile_main_program(self):
        self._insert_allreduce_ops()

    def _insert_allreduce_ops(self):
        block = self.main_program.global_block()

        grad2param = {}
        for idx, op in reversed(list(enumerate(block.ops))):
            if self._is_optimizer_op(op):
                param = block.vars[op.input('Param')[0]]
                grad2param[op.input('Grad')[0]] = param
                continue

            # TODO(liuyi05): use kBackward op tpe to check?
            for name in op.output_arg_names:
                if name in grad2param:
                    grad = block.vars[name]
                    block._insert_op(idx + 1, type='c_sync_compute_stream')
                    block._insert_op(
                        idx + 2,
                        type='c_allreduce',
                        inputs={'X': [grad]},
                        outputs={'Out': [grad]},
                        attrs={'reduce_type': 0})
                    del grad2param[name]

        for idx, op in enumerate(block.ops):
            if self._is_optimizer_op(op):
                block._insert_op(
                    idx,
                    type='c_sync_comm_stream',
                    attrs={'ring_id': self.global_ring_id})
                break


class LocalSGD(Collective):
    '''
    '''

    def __init__(self):
        Collective.__init__(self)
        self.snapshot_key = '@SNAPSHOT'

    def _transpile_startup_program(self):
        Collective._transpile_startup_program(self)

        block = self.startup_program.global_block()
        for param in block.iter_parameters():
            snapshot_name = param.name + self.snapshot_key
            snapshot = block.create_var(
                name=snapshot_name,
                shape=param.shape,
                persistable=True,
                stop_gradient=True)
            block.append_op(
                type='assign',
                inputs={'X': [param]},
                outputs={'Out': [snapshot]})

    def snapshot_name(self, param_name):
        return param_name + self.snapshot_key

    def _transpile_main_program(self):
        block = self.main_program.global_block()
        ordered_param_snapshot = []
        for idx, op in reversed(list(enumerate(block.ops))):
            if self._is_optimizer_op(op):
                param = block.vars[op.input('Param')[0]]
                snapshot = block.create_var(
                    name=self.snapshot_name(param.name),
                    shape=param.shape,
                    persistable=True,
                    stop_gradient=True)

                block._insert_op(
                    idx + 1,
                    type='elementwise_sub',
                    inputs={'X': [snapshot],
                            'Y': [param]},
                    outputs={'Out': [param]})
                block._insert_op(idx + 2, type='c_sync_compute_stream')
                block._insert_op(
                    idx + 3,
                    type='c_allreduce',
                    inputs={'X': [param]},
                    outputs={'Out': [param]},
                    attrs={'reduce_type': 0})

                ordered_param_snapshot.append((param, snapshot))

        block.append_op(
            type='c_sync_comm_stream', attrs={'ring_id': self.global_ring_id})

        for param_snapshot in reversed(ordered_param_snapshot):
            param = param_snapshot[0]
            snapshot = param_snapshot[1]
            block.append_op(
                type='scale',
                inputs={'X': [param]},
                outputs={'Out': [param]},
                attrs={'scale': 1.0 / self.nranks})
            block.append_op(
                type='elementwise_sub',
                inputs={'X': [snapshot],
                        'Y': [param]},
                outputs={'Out': [param]})
            block.append_op(
                type='assign',
                inputs={'X': [param]},
                outputs={'Out': [snapshot]})
