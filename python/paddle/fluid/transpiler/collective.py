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

from .. import core, unique_name
from ..framework import Program, default_main_program, default_startup_program
from ..backward import _append_grad_suffix_
from .details import wait_server_ready
from .. import layers

__all__ = ['GradAllReduce', 'LocalSGD', 'DistributedClassificationOptimizer']

OpRole = core.op_proto_and_checker_maker.OpRole


class Collective(object):
    '''
    '''

    def __init__(self, nrings):
        self.nrings = nrings
        self.endpoints = None
        self.current_endpoint = None
        self.nranks = None
        self.rank = None
        self.startup_program = None
        self.main_program = None
        op_maker = core.op_proto_and_checker_maker
        self.op_role_key = op_maker.kOpRoleAttrName()
        self.op_role_var_key = op_maker.kOpRoleVarAttrName()

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

        self._transpile_startup_program()

        self._transpile_main_program()

    def _transpile_main_program(self):
        raise NotImplementedError('call the inherited method of subclasses')

    def _transpile_startup_program(self):
        for ring_id in range(self.nrings):
            self._init_communicator(self.startup_program, self.current_endpoint,
                                    self.endpoints, self.rank, ring_id,
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
            name=unique_name.generate('nccl_id'),
            persistable=True,
            type=core.VarDesc.VarType.RAW)
        block.append_op(
            type='c_gen_nccl_id',
            inputs={},
            outputs={'Out': nccl_id_var},
            attrs={
                'rank': rank,
                'endpoint': current_endpoint,
                'other_endpoints': other_endpoints,
                self.op_role_key: OpRole.Forward
            })
        block.append_op(
            type='c_comm_init',
            inputs={'X': nccl_id_var},
            outputs={},
            attrs={
                'nranks': nranks,
                'rank': rank,
                'ring_id': ring_id,
                self.op_role_key: OpRole.Forward
            })

    def _broadcast_params(self):
        block = self.startup_program.global_block()
        ring_id = -1
        for param in block.iter_parameters():
            if param.is_distributed:
                continue

            ring_id = (ring_id + 1) % self.nrings
            block.append_op(
                type='c_broadcast',
                inputs={'X': param},
                outputs={'Out': param},
                attrs={
                    'ring_id': ring_id,
                    'root': 0,
                    self.op_role_key: OpRole.Forward
                })

        for ring_id in range(self.nrings):
            block.append_op(
                type='c_sync_comm_stream',
                inputs={'X': param},
                outputs={'Out': param},
                attrs={'ring_id': ring_id,
                       self.op_role_key: OpRole.Forward})

    def _is_loss_grad_op(self, op):
        if self.op_role_key not in op.attr_names:
            return False
        op_role = int(op.all_attrs()[self.op_role_key])
        return op_role & int(OpRole.Backward) and op_role & int(OpRole.Loss)

    def _is_backward_op(self, op):
        return self.op_role_key in op.attr_names and \
                int(op.all_attrs()[self.op_role_key]) & int(OpRole.Backward)

    def _is_update_op(self, op):
        return 'Param' in op.input_names and 'Grad' in op.input_names and \
                'LearningRate' in op.input_names

    def _is_optimizer_op(self, op):
        return self.op_role_key in op.attr_names and \
                int(op.all_attrs()[self.op_role_key]) & int(OpRole.Optimize)


class GradAllReduce(Collective):
    '''
    '''

    def __init__(self, nrings=2):
        Collective.__init__(self, nrings)

    def _transpile_main_program(self):
        self._insert_scale_loss_grad_ops()
        self._insert_allreduce_ops()

    def _insert_scale_loss_grad_ops(self):
        '''
        In order to keep the learning rate consistent in different numbers of
        training workers, we scale the loss grad by the number of workers
        '''
        block = self.main_program.global_block()
        for idx, op in reversed(list(enumerate(block.ops))):
            if self._is_loss_grad_op(op):
                loss_grad_var = block.vars[op.output_arg_names[0]]
                block._insert_op(
                    idx + 1,
                    type='scale',
                    inputs={'X': loss_grad_var},
                    outputs={'Out': loss_grad_var},
                    attrs={
                        'scale': 1.0 / self.nranks,
                        self.op_role_key: OpRole.Backward
                    })

    def _insert_allreduce_ops(self):
        block = self.main_program.global_block()
        ring_id = -1
        grad = None
        for idx, op in reversed(list(enumerate(block.ops))):
            if self._is_backward_op(op) and \
                    self.op_role_var_key in op.attr_names:
                op_role_var = op.all_attrs()[self.op_role_var_key]

                if len(op_role_var) == 0:
                    continue
                assert len(op_role_var) % 2 == 0

                offset = idx
                for i in range(0, len(op_role_var), 2):
                    param = block.vars[op_role_var[i]]
                    grad = block.vars[op_role_var[i + 1]]
                    if param.is_distributed:
                        continue

                    if offset == idx:
                        offset += 1
                        block._insert_op(
                            offset,
                            type='c_sync_calc_stream',
                            inputs={'X': grad},
                            outputs={'Out': grad},
                            attrs={self.op_role_key: OpRole.Backward})
                        offset += 1

                    # As we search ops reversedly, we should insert c_allreduce_sum
                    # op in the same way to keep the ring_id alternate
                    ring_id = (ring_id + 1) % self.nrings
                    block._insert_op(
                        offset,
                        type='c_allreduce_sum',
                        inputs={'X': grad},
                        outputs={'Out': grad},
                        attrs={
                            'ring_id': ring_id,
                            self.op_role_key: OpRole.Backward
                        })

        if grad is None:
            return

        for idx, op in enumerate(block.ops):
            if self._is_optimizer_op(op):
                for ring_id in range(self.nrings):
                    block._insert_op(
                        idx + ring_id,
                        type='c_sync_comm_stream',
                        inputs={'X': grad},
                        outputs={'Out': grad},
                        attrs={
                            'ring_id': ring_id,
                            self.op_role_key: OpRole.Backward
                        })
                break


class LocalSGD(Collective):
    '''
    '''

    def __init__(self, nrings=2):
        Collective.__init__(self, nrings)
        self.snapshot_key = '@SNAPSHOT'

    def _transpile_startup_program(self):
        Collective._transpile_startup_program(self)

        block = self.startup_program.global_block()
        non_dist_params = []
        for param in block.iter_parameters():
            if not param.is_distributed:
                non_dist_params.append(param)

        for param in non_dist_params:
            snapshot = block.create_var(
                name=self.snapshot_name(param.name),
                shape=param.shape,
                persistable=True,
                stop_gradient=True)
            block.append_op(
                type='assign',
                inputs={'X': [param]},
                outputs={'Out': [snapshot]},
                attrs={self.op_role_key: OpRole.Forward})

    def snapshot_name(self, param_name):
        return param_name + self.snapshot_key

    def _transpile_main_program(self):
        block = self.main_program.global_block()
        ordered_param_snapshot = []
        ring_id = -1
        for idx, op in reversed(list(enumerate(block.ops))):
            if self._is_update_op(op):
                param = block.vars[op.input('Param')[0]]
                if param.is_distributed:
                    continue

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
                    outputs={'Out': [param]},
                    attrs={self.op_role_key: OpRole.Optimize})
                block._insert_op(
                    idx + 2,
                    type='c_sync_calc_stream',
                    inputs={'X': param},
                    outputs={'Out': param},
                    attrs={self.op_role_key: OpRole.Optimize})
                ring_id = (ring_id + 1) % self.nrings
                block._insert_op(
                    idx + 3,
                    type='c_allreduce_sum',
                    inputs={'X': [param]},
                    outputs={'Out': [param]},
                    attrs={
                        'ring_id': ring_id,
                        self.op_role_key: OpRole.Optimize
                    })

                ordered_param_snapshot.append((param, snapshot))

        for ring_id in range(self.nrings):
            block.append_op(
                type='c_sync_comm_stream',
                inputs={'X': param},
                outputs={'Out': param},
                attrs={'ring_id': ring_id,
                       self.op_role_key: OpRole.Optimize})

        for param_snapshot in reversed(ordered_param_snapshot):
            param = param_snapshot[0]
            snapshot = param_snapshot[1]
            block.append_op(
                type='scale',
                inputs={'X': [param]},
                outputs={'Out': [param]},
                attrs={
                    'scale': 1.0 / self.nranks,
                    self.op_role_key: OpRole.Optimize
                })
            block.append_op(
                type='elementwise_sub',
                inputs={'X': [snapshot],
                        'Y': [param]},
                outputs={'Out': [param]},
                attrs={self.op_role_key: OpRole.Optimize})
            block.append_op(
                type='assign',
                inputs={'X': [param]},
                outputs={'Out': [snapshot]},
                attrs={self.op_role_key: OpRole.Optimize})


class DistributedClassificationOptimizer(object):
    '''
    A optimizer wrapper to generate backward network for distributed
    classification training of model parallelism.
    '''

    def __init__(self, optimizer, batch_size):
        self._optimizer = optimizer
        self._batch_size = batch_size

    def minimize(self, loss):
        # TODO: use paddle enforce
        assert loss._get_info('shard_logit')

        shard_logit = loss._get_info('shard_logit')
        shard_prob = loss._get_info('shard_prob')
        shard_label = loss._get_info('shard_label')
        shard_dim = loss._get_info('shard_dim')

        op_maker = core.op_proto_and_checker_maker
        op_role_key = op_maker.kOpRoleAttrName()
        op_role_var_key = op_maker.kOpRoleVarAttrName()
        backward_role = int(op_maker.OpRole.Backward)
        loss_backward_role = int(op_maker.OpRole.Loss) | int(
            op_maker.OpRole.Backward)

        # minimize a scalar of reduce_sum to generate the backward network
        scalar = layers.reduce_sum(shard_logit)
        ret = self._optimizer.minimize(scalar)

        block = loss.block
        # remove the unnecessary ops
        index = 0
        for i, op in enumerate(block.ops):
            if op.all_attrs()[op_role_key] == loss_backward_role:
                index = i
                break

        # TODO: use paddle enforce
        assert block.ops[index - 1].type == 'reduce_sum'
        assert block.ops[index].type == 'fill_constant'
        assert block.ops[index + 1].type == 'reduce_sum_grad'
        block._remove_op(index + 1)
        block._remove_op(index)
        block._remove_op(index - 1)

        # insert the calculated gradient 
        dtype = shard_logit.dtype
        shard_one_hot = layers.create_tensor(dtype, name='shard_one_hot')
        block._insert_op(
            index - 1,
            type='one_hot',
            inputs={'X': shard_label},
            outputs={'Out': shard_one_hot},
            attrs={
                'depth': shard_dim,
                'allow_out_of_range': True,
                op_role_key: backward_role
            })
        shard_logit_grad = layers.create_tensor(
            dtype, name=_append_grad_suffix_(shard_logit.name))
        block._insert_op(
            index,
            type='elementwise_sub',
            inputs={'X': shard_prob,
                    'Y': shard_one_hot},
            outputs={'Out': shard_logit_grad},
            attrs={op_role_key: backward_role})
        block._insert_op(
            index + 1,
            type='scale',
            inputs={'X': shard_logit_grad},
            outputs={'Out': shard_logit_grad},
            attrs={
                'scale': 1.0 / self._batch_size,
                op_role_key: loss_backward_role
            })
        return ret
