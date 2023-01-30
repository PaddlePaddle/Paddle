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

<<<<<<< HEAD
=======
from __future__ import print_function

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import sys
import math
from functools import reduce
import os

import collections
<<<<<<< HEAD
=======
import six
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import logging

import numpy as np

from .. import core, unique_name
from ..framework import Program, default_main_program, default_startup_program
from .details import wait_server_ready

__all__ = ['GradAllReduce', 'LocalSGD', 'MultiThread']

OpRole = core.op_proto_and_checker_maker.OpRole


<<<<<<< HEAD
class Collective:
    ''' '''
=======
class Collective(object):
    '''
    '''
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def __init__(self, nrings):
        self.nrings = nrings
        self.endpoints = None
        self.current_endpoint = None
        self.other_endpoints = None
        self.nranks = None
        self.rank = None
        self.startup_program = None
        self.main_program = None
        op_maker = core.op_proto_and_checker_maker
        self.op_role_key = op_maker.kOpRoleAttrName()
        self.op_role_var_key = op_maker.kOpRoleVarAttrName()

<<<<<<< HEAD
    def transpile(
        self,
        startup_program,
        main_program,
        rank,
        endpoints,
        current_endpoint,
        wait_port,
    ):
=======
    def transpile(self, startup_program, main_program, rank, endpoints,
                  current_endpoint, wait_port):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
        if (
            self.nranks == 1
            and self.mode != "single_process_multi_thread"
            and self.mode != "box"
        ):
=======
        if self.nranks == 1 and self.mode != "single_process_multi_thread" and self.mode != "box":
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            raise ValueError('the number of endpoints must > 1')

        if rank < 0:
            raise ValueError('rank must >= 0')
        self.rank = rank

        if current_endpoint not in endpoints:
<<<<<<< HEAD
            raise ValueError(
                'current endpoint %s is not in %s',
                current_endpoint,
                str(endpoints),
            )
=======
            raise ValueError('current endpoint %s is not in %s',
                             current_endpoint, str(endpoints))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.endpoints = endpoints
        self.current_endpoint = current_endpoint

        if current_endpoint:
            nranks = len(endpoints)
            other_endpoints = endpoints[:]
            other_endpoints.remove(current_endpoint)
            self.other_endpoints = other_endpoints

        self.wait_port = wait_port

        self.startup_program._origin_program = self.startup_program.clone()
        self._transpile_startup_program()

        self.main_program._origin_program = self.main_program.clone()
        self._transpile_main_program()

    def _transpile_main_program(self):
        raise NotImplementedError('call the inherited method of subclasses')

    def _transpile_startup_program(self):
        for ring_id in range(self.nrings):
<<<<<<< HEAD
            self._init_communicator(
                self.startup_program,
                self.current_endpoint,
                self.endpoints,
                self.rank,
                ring_id,
                self.wait_port,
            )
        self._broadcast_params()

    def _init_communicator(
        self,
        program,
        current_endpoint,
        endpoints,
        rank,
        ring_id,
        wait_port,
        has_multitrainer=False,
    ):
=======
            self._init_communicator(self.startup_program, self.current_endpoint,
                                    self.endpoints, self.rank, ring_id,
                                    self.wait_port)
        self._broadcast_params()

    def _init_communicator(self,
                           program,
                           current_endpoint,
                           endpoints,
                           rank,
                           ring_id,
                           wait_port,
                           has_multitrainer=False):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        nranks = len(endpoints)
        other_endpoints = endpoints[:]
        other_endpoints.remove(current_endpoint)
        block = program.global_block()

        if rank == 0 and wait_port:
            wait_server_ready(other_endpoints)

        block = program.global_block()
        if core.is_compiled_with_npu():
<<<<<<< HEAD
            hccl_id_var = block.create_var(
                name=unique_name.generate('hccl_id'),
                persistable=True,
                type=core.VarDesc.VarType.RAW,
            )
            endpoint_to_index_map = {e: idx for idx, e in enumerate(endpoints)}
            block.append_op(
                type='c_gen_hccl_id',
                inputs={},
                outputs={'Out': hccl_id_var},
                attrs={
                    'rank': rank,
                    'endpoint': current_endpoint,
                    'other_endpoints': other_endpoints,
                    self.op_role_key: OpRole.Forward,
                },
            )
            block.append_op(
                type='c_comm_init_hccl',
                inputs={'X': hccl_id_var},
                outputs={},
                attrs={
                    'rank': rank,
                    'ring_id': ring_id,
                    'device_id': int(os.getenv("FLAGS_selected_npus")),
                    'rank_ids': nranks,
                    self.op_role_key: OpRole.Forward,
                },
            )
        else:
            nccl_id_var = block.create_var(
                name=unique_name.generate('nccl_id'),
                persistable=True,
                type=core.VarDesc.VarType.RAW,
            )
            block.append_op(
                type='c_gen_nccl_id',
                inputs={},
                outputs={'Out': nccl_id_var},
                attrs={
                    'rank': rank,
                    'endpoint': current_endpoint,
                    'other_endpoints': other_endpoints,
                    self.op_role_key: OpRole.Forward,
                },
            )
            if not has_multitrainer:
                block.append_op(
                    type='c_comm_init',
                    inputs={'X': nccl_id_var},
                    outputs={},
                    attrs={
                        'nranks': nranks,
                        'rank': rank,
                        'ring_id': ring_id,
                        self.op_role_key: OpRole.Forward,
                    },
                )
            else:
                block.append_op(
                    type='c_comm_init_multitrainer',
                    inputs={'X': nccl_id_var},
                    outputs={},
                    attrs={
                        'ntrainers': nranks,
                        'trainer_id': rank,
                        'ring_id': ring_id,
                        self.op_role_key: OpRole.Forward,
                    },
                )
=======
            hccl_id_var = block.create_var(name=unique_name.generate('hccl_id'),
                                           persistable=True,
                                           type=core.VarDesc.VarType.RAW)
            endpoint_to_index_map = {e: idx for idx, e in enumerate(endpoints)}
            block.append_op(type='c_gen_hccl_id',
                            inputs={},
                            outputs={'Out': hccl_id_var},
                            attrs={
                                'rank': rank,
                                'endpoint': current_endpoint,
                                'other_endpoints': other_endpoints,
                                self.op_role_key: OpRole.Forward
                            })
            block.append_op(type='c_comm_init_hccl',
                            inputs={'X': hccl_id_var},
                            outputs={},
                            attrs={
                                'rank': rank,
                                'ring_id': ring_id,
                                'device_id':
                                int(os.getenv("FLAGS_selected_npus")),
                                'rank_ids': nranks,
                                self.op_role_key: OpRole.Forward
                            })
        else:
            nccl_id_var = block.create_var(name=unique_name.generate('nccl_id'),
                                           persistable=True,
                                           type=core.VarDesc.VarType.RAW)
            block.append_op(type='c_gen_nccl_id',
                            inputs={},
                            outputs={'Out': nccl_id_var},
                            attrs={
                                'rank': rank,
                                'endpoint': current_endpoint,
                                'other_endpoints': other_endpoints,
                                self.op_role_key: OpRole.Forward
                            })
            if not has_multitrainer:
                block.append_op(type='c_comm_init',
                                inputs={'X': nccl_id_var},
                                outputs={},
                                attrs={
                                    'nranks': nranks,
                                    'rank': rank,
                                    'ring_id': ring_id,
                                    self.op_role_key: OpRole.Forward
                                })
            else:
                block.append_op(type='c_comm_init_multitrainer',
                                inputs={'X': nccl_id_var},
                                outputs={},
                                attrs={
                                    'ntrainers': nranks,
                                    'trainer_id': rank,
                                    'ring_id': ring_id,
                                    self.op_role_key: OpRole.Forward
                                })
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def _broadcast_params(self):
        block = self.startup_program.global_block()
        ring_id = -1
        for param in block.iter_parameters():
            if param.is_distributed:
                continue

            ring_id = (ring_id + 1) % self.nrings
<<<<<<< HEAD
            block.append_op(
                type='c_broadcast',
                inputs={'X': param},
                outputs={'Out': param},
                attrs={
                    'ring_id': ring_id,
                    'root': 0,
                    self.op_role_key: OpRole.Forward,
                },
            )

        for ring_id in range(self.nrings):
            block.append_op(
                type='c_sync_comm_stream',
                inputs={'X': param},
                outputs={'Out': param},
                attrs={'ring_id': ring_id, self.op_role_key: OpRole.Forward},
            )
=======
            block.append_op(type='c_broadcast',
                            inputs={'X': param},
                            outputs={'Out': param},
                            attrs={
                                'ring_id': ring_id,
                                'root': 0,
                                self.op_role_key: OpRole.Forward
                            })

        for ring_id in range(self.nrings):
            block.append_op(type='c_sync_comm_stream',
                            inputs={'X': param},
                            outputs={'Out': param},
                            attrs={
                                'ring_id': ring_id,
                                self.op_role_key: OpRole.Forward
                            })
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def _is_loss_grad_op(self, op):
        if self.op_role_key not in op.attr_names:
            return False
        op_role = int(op.all_attrs()[self.op_role_key])
        return op_role & int(OpRole.Backward) and op_role & int(OpRole.Loss)

    def _is_backward_op(self, op):
<<<<<<< HEAD
        return self.op_role_key in op.attr_names and int(
            op.all_attrs()[self.op_role_key]
        ) & int(OpRole.Backward)

    def _is_update_op(self, op):
        return (
            'Param' in op.input_names
            and 'Grad' in op.input_names
            and "LearningRate" in op.input_names
        )

    def _is_optimizer_op(self, op):
        return self.op_role_key in op.attr_names and int(
            op.all_attrs()[self.op_role_key]
        ) & int(OpRole.Optimize)


class GradAllReduce(Collective):
    ''' '''
=======
        return self.op_role_key in op.attr_names and \
                int(op.all_attrs()[self.op_role_key]) & int(OpRole.Backward)

    def _is_update_op(self, op):
        return 'Param' in op.input_names and 'Grad' in op.input_names and \
                "LearningRate" in op.input_names

    def _is_optimizer_op(self, op):
        return self.op_role_key in op.attr_names and \
                int(op.all_attrs()[self.op_role_key]) & int(OpRole.Optimize)


class GradAllReduce(Collective):
    '''
    '''
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def __init__(self, nrings=2):
        Collective.__init__(self, nrings)
        self.mode = "grad_allreduce"

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
<<<<<<< HEAD
                block._insert_op(
                    idx + 1,
                    type='scale',
                    inputs={'X': loss_grad_var},
                    outputs={'Out': loss_grad_var},
                    attrs={
                        'scale': 1.0 / self.nranks,
                        self.op_role_key: OpRole.Backward,
                    },
                )
=======
                block._insert_op(idx + 1,
                                 type='scale',
                                 inputs={'X': loss_grad_var},
                                 outputs={'Out': loss_grad_var},
                                 attrs={
                                     'scale': 1.0 / self.nranks,
                                     self.op_role_key: OpRole.Backward
                                 })
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def _insert_allreduce_ops(self):
        block = self.main_program.global_block()
        ring_id = -1
        grad = None
        for idx, op in reversed(list(enumerate(block.ops))):
<<<<<<< HEAD
            if (
                self._is_backward_op(op)
                and self.op_role_var_key in op.attr_names
            ):
=======
            if self._is_backward_op(op) and \
                    self.op_role_var_key in op.attr_names:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
                            attrs={self.op_role_key: OpRole.Backward},
                        )
=======
                            attrs={self.op_role_key: OpRole.Backward})
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                        offset += 1

                    # As we search ops reversedly, we should insert c_allreduce_sum
                    # op in the same way to keep the ring_id alternate
                    ring_id = (ring_id + 1) % self.nrings
<<<<<<< HEAD
                    block._insert_op(
                        offset,
                        type='c_allreduce_sum',
                        inputs={'X': grad},
                        outputs={'Out': grad},
                        attrs={
                            'ring_id': ring_id,
                            self.op_role_key: OpRole.Backward,
                        },
                    )
=======
                    block._insert_op(offset,
                                     type='c_allreduce_sum',
                                     inputs={'X': grad},
                                     outputs={'Out': grad},
                                     attrs={
                                         'ring_id': ring_id,
                                         self.op_role_key: OpRole.Backward
                                     })
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        if grad is None:
            return

        for idx, op in enumerate(block.ops):
            if self._is_optimizer_op(op):
                for ring_id in range(self.nrings):
<<<<<<< HEAD
                    block._insert_op(
                        idx + ring_id,
                        type='c_sync_comm_stream',
                        inputs={'X': grad},
                        outputs={'Out': grad},
                        attrs={
                            'ring_id': ring_id,
                            self.op_role_key: OpRole.Backward,
                        },
                    )
=======
                    block._insert_op(idx + ring_id,
                                     type='c_sync_comm_stream',
                                     inputs={'X': grad},
                                     outputs={'Out': grad},
                                     attrs={
                                         'ring_id': ring_id,
                                         self.op_role_key: OpRole.Backward
                                     })
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                break


class LocalSGD(Collective):
<<<<<<< HEAD
    ''' '''
=======
    '''
    '''
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def __init__(self, nrings=2):
        Collective.__init__(self, nrings)
        self.snapshot_key = '@SNAPSHOT'
        self.mode = "local_sgd"

    def _transpile_startup_program(self):
        Collective._transpile_startup_program(self)

        block = self.startup_program.global_block()
        non_dist_params = []
        for param in block.iter_parameters():
            if not param.is_distributed:
                non_dist_params.append(param)

        for param in non_dist_params:
<<<<<<< HEAD
            snapshot = block.create_var(
                name=self.snapshot_name(param.name),
                shape=param.shape,
                persistable=True,
                stop_gradient=True,
            )
            block.append_op(
                type='assign',
                inputs={'X': [param]},
                outputs={'Out': [snapshot]},
                attrs={self.op_role_key: OpRole.Forward},
            )
=======
            snapshot = block.create_var(name=self.snapshot_name(param.name),
                                        shape=param.shape,
                                        persistable=True,
                                        stop_gradient=True)
            block.append_op(type='assign',
                            inputs={'X': [param]},
                            outputs={'Out': [snapshot]},
                            attrs={self.op_role_key: OpRole.Forward})
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

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

<<<<<<< HEAD
                snapshot = block.create_var(
                    name=self.snapshot_name(param.name),
                    shape=param.shape,
                    persistable=True,
                    stop_gradient=True,
                    dtype=param.dtype,
                )

                block._insert_op(
                    idx + 1,
                    type='elementwise_sub',
                    inputs={'X': [snapshot], 'Y': [param]},
                    outputs={'Out': [param]},
                    attrs={self.op_role_key: OpRole.Optimize},
                )
                block._insert_op(
                    idx + 2,
                    type='c_sync_calc_stream',
                    inputs={'X': param},
                    outputs={'Out': param},
                    attrs={self.op_role_key: OpRole.Optimize},
                )
                ring_id = (ring_id + 1) % self.nrings
                block._insert_op(
                    idx + 3,
                    type='c_allreduce_sum',
                    inputs={'X': [param]},
                    outputs={'Out': [param]},
                    attrs={
                        'ring_id': ring_id,
                        self.op_role_key: OpRole.Optimize,
                    },
                )
=======
                snapshot = block.create_var(name=self.snapshot_name(param.name),
                                            shape=param.shape,
                                            persistable=True,
                                            stop_gradient=True,
                                            dtype=param.dtype)

                block._insert_op(idx + 1,
                                 type='elementwise_sub',
                                 inputs={
                                     'X': [snapshot],
                                     'Y': [param]
                                 },
                                 outputs={'Out': [param]},
                                 attrs={self.op_role_key: OpRole.Optimize})
                block._insert_op(idx + 2,
                                 type='c_sync_calc_stream',
                                 inputs={'X': param},
                                 outputs={'Out': param},
                                 attrs={self.op_role_key: OpRole.Optimize})
                ring_id = (ring_id + 1) % self.nrings
                block._insert_op(idx + 3,
                                 type='c_allreduce_sum',
                                 inputs={'X': [param]},
                                 outputs={'Out': [param]},
                                 attrs={
                                     'ring_id': ring_id,
                                     self.op_role_key: OpRole.Optimize
                                 })
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

                ordered_param_snapshot.append((param, snapshot))

        for ring_id in range(self.nrings):
<<<<<<< HEAD
            block.append_op(
                type='c_sync_comm_stream',
                inputs={'X': param},
                outputs={'Out': param},
                attrs={'ring_id': ring_id, self.op_role_key: OpRole.Optimize},
            )
=======
            block.append_op(type='c_sync_comm_stream',
                            inputs={'X': param},
                            outputs={'Out': param},
                            attrs={
                                'ring_id': ring_id,
                                self.op_role_key: OpRole.Optimize
                            })
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        for param_snapshot in reversed(ordered_param_snapshot):
            param = param_snapshot[0]
            snapshot = param_snapshot[1]
<<<<<<< HEAD
            block.append_op(
                type='scale',
                inputs={'X': [param]},
                outputs={'Out': [param]},
                attrs={
                    'scale': 1.0 / self.nranks,
                    self.op_role_key: OpRole.Optimize,
                },
            )
            block.append_op(
                type='elementwise_sub',
                inputs={'X': [snapshot], 'Y': [param]},
                outputs={'Out': [param]},
                attrs={self.op_role_key: OpRole.Optimize},
            )
            block.append_op(
                type='assign',
                inputs={'X': [param]},
                outputs={'Out': [snapshot]},
                attrs={self.op_role_key: OpRole.Optimize},
            )


class SingleProcessMultiThread(GradAllReduce):
    """
    single process multi thread mode
    """
=======
            block.append_op(type='scale',
                            inputs={'X': [param]},
                            outputs={'Out': [param]},
                            attrs={
                                'scale': 1.0 / self.nranks,
                                self.op_role_key: OpRole.Optimize
                            })
            block.append_op(type='elementwise_sub',
                            inputs={
                                'X': [snapshot],
                                'Y': [param]
                            },
                            outputs={'Out': [param]},
                            attrs={self.op_role_key: OpRole.Optimize})
            block.append_op(type='assign',
                            inputs={'X': [param]},
                            outputs={'Out': [snapshot]},
                            attrs={self.op_role_key: OpRole.Optimize})


class SingleProcessMultiThread(GradAllReduce):
    '''
    '''
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def __init__(self):
        GradAllReduce.__init__(self, 1)
        self.mode = "single_process_multi_thread"
<<<<<<< HEAD
        self.fuse_allreduce = int(os.getenv("PADDLE_FUSE_ALLREDUCE", "1"))
        self.loss_scale = int(os.getenv("PADDLE_LOSS_SCALE", "1"))
        self.gpu_nums = len(
            os.getenv("FLAGS_selected_gpus", "0,1,2,3,4,5,6,7").split(",")
        )

    def _transpile_startup_program(self):
        nodes_num = 0
        if len(self.endpoints) > 1:
            nodes_num = len(set([x.split(':')[0] for x in self.endpoints]))
        # diffent ip num is multi node
        if nodes_num > 1:
            self.nranks = nodes_num
            print("begin to _transpile_startup_program for multi-node")
            print("current_endpoint: ", self.current_endpoint)
            print("total endpoints: ", self.endpoints)
            print("rank: %d, ring_id: %d" % (self.rank, self.nrings))
            for ring_id in range(self.nrings):
                self._init_communicator(
                    self.startup_program,
                    self.current_endpoint,
                    self.endpoints,
                    self.rank,
                    ring_id,
                    self.wait_port,
                    True,
                )
        else:
            self.nranks = 1
            print("begin to _transpile_startup_program for single-node")
            block = self.startup_program.global_block()
            block.append_op(type='c_comm_init_all', attrs={'ring_id': 0})

    def _transpile_main_program(self):
        # not need loss scale and no dense param
        param_cnt = self._get_update_param_count()
        if self.loss_scale is 0 and param_cnt is 0:
            return
        # scale loss
        self._insert_scale_loss_grad_ops()
        # no param
        if param_cnt is 0:
            return
        # fuse allreduce
        if self.fuse_allreduce > 0:
            print("begin used fuse_allreduce param count = %s" % (param_cnt))
            # use fuse allreduce
            self._insert_fuse_allreduce_ops()
        else:
            self._insert_allreduce_ops()

    def _get_update_param_count(self):
        """
        get need update param count
        """
        param_count = 0
        block = self.main_program.global_block()
        for idx, op in reversed(list(enumerate(block.ops))):
            if not self._is_backward_op(op):
                continue
            if not self.op_role_var_key in op.attr_names:
                continue
            op_role_var = op.all_attrs()[self.op_role_var_key]
            if len(op_role_var) == 0:
                continue

            assert len(op_role_var) % 2 == 0
            for i in range(0, len(op_role_var), 2):
                param = block.vars[op_role_var[i]]
                if param.is_distributed:
                    continue
                param_count = param_count + 1

        return param_count

    def _insert_scale_loss_grad_ops(self):
        '''
        In order to keep the learning rate consistent in different numbers of
        training workers, we scale the loss grad by the number of workers
        '''
        scale = 1.0 / self.nranks / self.gpu_nums
        print("begin _insert_scale_loss_grad_ops scale = %s" % (scale))
        block = self.main_program.global_block()
        for idx, op in reversed(list(enumerate(block.ops))):
            if not self._is_loss_grad_op(op):
                continue
            loss_grad_var = block.vars[op.output_arg_names[0]]
            block._insert_op(
                idx + 1,
                type='scale',
                inputs={'X': loss_grad_var},
                outputs={'Out': loss_grad_var},
                attrs={'scale': scale, self.op_role_key: OpRole.Backward},
            )

    def _insert_fuse_allreduce_ops(self):
        """
        insert coalesce_tensor and all reduce ops
        """
        block = self.main_program.global_block()
        ring_id = -1
        grad = None
        input_grads = []
        global_offset = 0  # find insert offset of fuse tensor, after the max dense grad offset
        for idx, op in reversed(list(enumerate(block.ops))):
            if (
                self._is_backward_op(op)
                and self.op_role_var_key in op.attr_names
            ):
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
                        input_grads.append(grad)
                        global_offset = max(global_offset, offset + 1)
        if grad is None:
            return

        # init output_grads
        output_grads = input_grads
        # init fused_output with temp shape, it will calculate real shape depend on inputs
        fused_output = block.create_var(
            name="fused_output",
            shape=[1],
            persistable=False,
            dtype=core.VarDesc.VarType.FP32,
            stop_gradient=True,
        )
        # fuse all grad tensors
        coalesce_tensor_attrs = {
            "copy_data": True,
            "set_constant": False,
            "dtype": core.VarDesc.VarType.FP32,
        }
        block._insert_op(
            global_offset,
            type='coalesce_tensor',
            inputs={'Input': input_grads},
            outputs={'Output': output_grads, 'FusedOutput': fused_output},
            attrs=coalesce_tensor_attrs,
        )
        global_offset += 1
        # grads aggregation of multi-gpus
        block._insert_op(
            global_offset,
            type='c_sync_calc_stream',
            inputs={'X': fused_output},
            outputs={'Out': fused_output},
            attrs={self.op_role_key: OpRole.Backward},
        )
        global_offset += 1
        ring_id = (ring_id + 1) % self.nrings
        block._insert_op(
            global_offset,
            type='c_allreduce_sum',
            inputs={'X': fused_output},
            outputs={'Out': fused_output},
            attrs={'ring_id': ring_id, self.op_role_key: OpRole.Backward},
        )
        global_offset += 1

        # sync before adam
        block._insert_op(
            global_offset,
            type='c_sync_comm_stream',
            inputs={'X': fused_output},
            outputs={'Out': fused_output},
            attrs={'ring_id': ring_id, self.op_role_key: OpRole.Backward},
        )
        global_offset += 1


class MultiThread(GradAllReduce):
    ''' '''
=======

    def _transpile_startup_program(self):
        block = self.startup_program.global_block()
        block.append_op(type='c_comm_init_all', attrs={'ring_id': 0})


class MultiThread(GradAllReduce):
    '''
    '''
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def __init__(self, nrings=1, trans_mode="all_reduce"):
        GradAllReduce.__init__(self, nrings)
        self.mode = "box"
        self.trans_mode = trans_mode
        self.fuse_grad_size_in_num = 128
<<<<<<< HEAD
        gpu_nums = os.getenv("FLAGS_selected_gpus", "0,1,2,3,4,5,6,7,8").split(
            ","
        )
=======
        gpu_nums = os.getenv("FLAGS_selected_gpus",
                             "0,1,2,3,4,5,6,7,8").split(",")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.gpu_num = len(gpu_nums)

    def _transpile_startup_program(self):
        if len(self.endpoints) > 1:
            print("begin to _transpile_startup_program for multi-node")
            print("current_endpoint: ", self.current_endpoint)
            print("total endpoints: ", self.endpoints)
            print("rank: %d, ring_id: %d" % (self.rank, self.nrings))
            for ring_id in range(self.nrings):
<<<<<<< HEAD
                self._init_communicator(
                    self.startup_program,
                    self.current_endpoint,
                    self.endpoints,
                    self.rank,
                    ring_id,
                    self.wait_port,
                    True,
                )
=======
                self._init_communicator(self.startup_program,
                                        self.current_endpoint, self.endpoints,
                                        self.rank, ring_id, self.wait_port,
                                        True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        else:
            if "xpu" in self.trans_mode:
                print(
                    "begin to _transpile_startup_program for single-node in XPU"
                )
                block = self.startup_program.global_block()
                block.append_op(
                    type='c_comm_init_all',
                    attrs={
<<<<<<< HEAD
                        'devices': list(
                            map(
                                int, os.getenv("FLAGS_selected_gpus").split(",")
                            )
                        ),
                        'ring_id': 0,
                    },
                )
=======
                        'devices':
                        list(
                            map(int,
                                os.getenv("FLAGS_selected_gpus").split(","))),
                        'ring_id':
                        0
                    })
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            else:
                print("begin to _transpile_startup_program for single-node")
                block = self.startup_program.global_block()
                block.append_op(type='c_comm_init_all', attrs={'ring_id': 0})

    def _transpile_main_program(self):
        self._insert_scale_loss_grad_ops()
        if self.trans_mode == "all_gather":
            print("begin to transpile in all-gather mode")
            self.allgather_ranks = self.nranks * self.gpu_num
            self._insert_allgather_ops()
            self._update_adam_ops()
        elif self.trans_mode == "fuse_all_reduce":
            print("begin to transpile in fuse all-reduce mode")
            self._insert_fuse_allreduce_ops()
<<<<<<< HEAD
        elif (
            self.trans_mode == "all_reduce_xpu"
            and len(os.getenv("FLAGS_selected_gpus").split(",")) == 1
        ):
=======
        elif self.trans_mode == "all_reduce_xpu" and len(
                os.getenv("FLAGS_selected_gpus").split(",")) == 1:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            print(
                "skip transpile in all-reduce-xpu mode when number of devices is only one"
            )
        else:
            print("begin to transpile in all-reduce mode")
            self._insert_allreduce_ops()

    def _insert_allgather_ops(self):
        """
        insert allgather op to the main_program
        """
        block = self.main_program.global_block()
        ring_id = -1
        grad = None
        for idx, op in reversed(list(enumerate(block.ops))):
<<<<<<< HEAD
            if (
                self._is_backward_op(op)
                and self.op_role_var_key in op.attr_names
            ):
=======
            if self._is_backward_op(op) and \
                    self.op_role_var_key in op.attr_names:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                op_role_var = op.all_attrs()[self.op_role_var_key]
                if len(op_role_var) == 0:
                    continue
                assert len(op_role_var) % 2 == 0

                offset = idx
                for i in range(0, len(op_role_var), 2):
                    param = block.vars[op_role_var[i]]
                    new_grad_var = block.create_var(
                        name=op_role_var[i] + "_allgather",
                        shape=[self.allgather_ranks] + list(param.shape),
                        persistable=False,
                        dtype=core.VarDesc.VarType.FP32,
<<<<<<< HEAD
                        stop_gradient=True,
                    )
=======
                        stop_gradient=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    grad = block.vars[op_role_var[i + 1]]
                    if param.is_distributed:  # no need to care: used in PLSC
                        continue

                    if offset == idx:
                        offset += 1
                        block._insert_op(
                            offset,
                            type='c_sync_calc_stream',
                            inputs={'X': grad},
                            outputs={'Out': grad},
<<<<<<< HEAD
                            attrs={self.op_role_key: OpRole.Backward},
                        )
=======
                            attrs={self.op_role_key: OpRole.Backward})
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                        offset += 1

                    # As we search ops reversedly, we should insert c_allgather
                    # op in the same way to keep the ring_id alternate
                    ring_id = (ring_id + 1) % self.nrings
<<<<<<< HEAD
                    block._insert_op(
                        offset,
                        type='c_allgather',
                        inputs={'X': grad},
                        outputs={'Out': new_grad_var},
                        attrs={
                            'nranks': self.allgather_ranks,
                            'ring_id': ring_id,
                            self.op_role_key: OpRole.Backward,
                        },
                    )
=======
                    block._insert_op(offset,
                                     type='c_allgather',
                                     inputs={'X': grad},
                                     outputs={'Out': new_grad_var},
                                     attrs={
                                         'nranks': self.allgather_ranks,
                                         'ring_id': ring_id,
                                         self.op_role_key: OpRole.Backward
                                     })
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        if grad is None:
            return

        for idx, op in enumerate(block.ops):
            if self._is_optimizer_op(op):
                for ring_id in range(self.nrings):
<<<<<<< HEAD
                    block._insert_op(
                        idx + ring_id,
                        type='c_sync_comm_stream',
                        inputs={'X': grad},
                        outputs={'Out': grad},
                        attrs={
                            'ring_id': ring_id,
                            self.op_role_key: OpRole.Backward,
                        },
                    )
=======
                    block._insert_op(idx + ring_id,
                                     type='c_sync_comm_stream',
                                     inputs={'X': grad},
                                     outputs={'Out': grad},
                                     attrs={
                                         'ring_id': ring_id,
                                         self.op_role_key: OpRole.Backward
                                     })
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                break

    def _update_adam_ops(self):
        """
        remove the original adam op, and add new adam ops
        """
        block = self.main_program.global_block()

        for idx, op in reversed(list(enumerate(block.ops))):
            if self._is_optimizer_op(op):
                offset = idx
<<<<<<< HEAD
                if (
                    op.type != 'adam' and op.type != 'lamb'
                ):  # filter out scale op
=======
                if op.type != 'adam' and op.type != 'lamb':  # filter out scale op
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    continue
                param_name = op.input("Param")[0]
                inputs = {
                    "Param": block.vars[op.input("Param")[0]],
                    "LearningRate": block.vars[op.input("LearningRate")[0]],
                    "Moment1": block.vars[op.input("Moment1")[0]],
                    "Moment2": block.vars[op.input("Moment2")[0]],
                    "Beta1Pow": block.vars[op.input("Beta1Pow")[0]],
<<<<<<< HEAD
                    "Beta2Pow": block.vars[op.input("Beta2Pow")[0]],
=======
                    "Beta2Pow": block.vars[op.input("Beta2Pow")[0]]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                }
                outputs = {
                    "ParamOut": block.vars[op.output("ParamOut")[0]],
                    "Moment1Out": block.vars[op.output("Moment1Out")[0]],
                    "Moment2Out": block.vars[op.output("Moment2Out")[0]],
                    "Beta1PowOut": block.vars[op.output("Beta1PowOut")[0]],
<<<<<<< HEAD
                    "Beta2PowOut": block.vars[op.output("Beta2PowOut")[0]],
                }
                attrs = {
                    "epsilon": op.attr('epsilon'),
                    "beta1": op.attr('beta1'),
                    "beta2": op.attr('beta2'),
                    "lazy_mode": op.attr('lazy_mode'),
                    "min_row_size_to_use_multithread": op.attr(
                        'min_row_size_to_use_multithread'
                    ),
=======
                    "Beta2PowOut": block.vars[op.output("Beta2PowOut")[0]]
                }
                attrs = {
                    "epsilon":
                    op.attr('epsilon'),
                    "beta1":
                    op.attr('beta1'),
                    "beta2":
                    op.attr('beta2'),
                    "lazy_mode":
                    op.attr('lazy_mode'),
                    "min_row_size_to_use_multithread":
                    op.attr('min_row_size_to_use_multithread')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                }
                split_vars = [
                    block.create_var(
                        name=param_name + "_" + str(i),
                        shape=block.vars[op.input("Param")[0]].shape,
                        persistable=False,
                        dtype=core.VarDesc.VarType.FP32,
<<<<<<< HEAD
                        stop_gradient=True,
                    )
                    for i in range(self.allgather_ranks)
                ]
                block._insert_op(
                    offset,
                    type="split",
                    inputs={
                        'X': block.vars[op.input("Param")[0] + "_allgather"]
                    },
                    outputs={'Out': split_vars},
                    attrs={'num': self.allgather_ranks, 'axis': 0},
                )
=======
                        stop_gradient=True) for i in range(self.allgather_ranks)
                ]
                block._insert_op(offset,
                                 type="split",
                                 inputs={
                                     'X':
                                     block.vars[op.input("Param")[0] +
                                                "_allgather"]
                                 },
                                 outputs={'Out': split_vars},
                                 attrs={
                                     'num': self.allgather_ranks,
                                     'axis': 0
                                 })
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                offset += 1

                for i in range(self.allgather_ranks):
                    inputs["Grad"] = split_vars[i]
<<<<<<< HEAD
                    block._insert_op(
                        offset,
                        type=op.type,
                        inputs=inputs,
                        outputs=outputs,
                        attrs=attrs,
                    )
=======
                    block._insert_op(offset,
                                     type=op.type,
                                     inputs=inputs,
                                     outputs=outputs,
                                     attrs=attrs)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    offset += 1
                # remove the original adam op
                block._remove_op(offset)

    def _insert_fuse_allreduce_ops(self):
        """
        insert coalesce_tensor and all reduce ops
        """
        block = self.main_program.global_block()
        ring_id = 0 % self.nrings
        grad = None
        param_grads = []
        # find all grad params
        for op in reversed(block.ops):
<<<<<<< HEAD
            if (
                self._is_backward_op(op)
                and self.op_role_var_key in op.attr_names
            ):
                op_role_var = op.all_attrs()[self.op_role_var_key]
                if len(op_role_var) == 0:
                    continue
                assert len(op_role_var) % 2 == 0, (
                    "vars need to be one param var followed by one grad var, "
                    "but got odd number of vars"
                )
=======
            if self._is_backward_op(op) and \
                    self.op_role_var_key in op.attr_names:
                op_role_var = op.all_attrs()[self.op_role_var_key]
                if len(op_role_var) == 0:
                    continue
                assert len(op_role_var) % 2 == 0, "vars need to be one param var followed by one grad var, " \
                                                  "but got odd number of vars"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                for i in range(0, len(op_role_var), 2):
                    param_name = op_role_var[i]
                    param = block.var(param_name)
                    grad_name = op_role_var[i + 1]
                    grad = block.var(grad_name)
                    if param.is_distributed:
                        continue
                    param_grads.append(grad)
        if grad is None:
            return

        segments = []
        last_dtype = None
        # split the grad based on dtype and fused size
        for var in param_grads:
<<<<<<< HEAD
            if (
                len(segments) == 0
                or len(segments[-1]) == self.fuse_grad_size_in_num
                or var.dtype != last_dtype
            ):
=======
            if len(segments) == 0 \
                    or len(segments[-1]) == self.fuse_grad_size_in_num \
                    or var.dtype != last_dtype:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                segments.append([var])
                last_dtype = var.dtype
            else:
                segments[-1].append(var)

        fused_vars = []
        for idx, op in enumerate(block.ops):
            if self._is_optimizer_op(op):
                for segment in segments:
                    # insert coalesce tensor
<<<<<<< HEAD
                    tmp_var = block.create_var(
                        name=unique_name.generate(
                            'FusedOutput_{}'.format(segment[0].name)
                        ),
                        dtype=segment[0].dtype,
                        persistable=False,
                        stop_gradient=True,
                    )
                    fused_vars.append(tmp_var)
                    block._insert_op(
                        idx,
                        type="coalesce_tensor",
                        inputs={"Input": segment},
                        outputs={"Output": segment, "FusedOutput": tmp_var},
                        attrs={
                            "copy_data": True,
                            "use_align": True,
                            "dtype": segment[0].dtype,
                            self.op_role_key: OpRole.Backward,
                        },
                    )
=======
                    tmp_var = block.create_var(name=unique_name.generate(
                        'FusedOutput_{}'.format(segment[0].name)),
                                               dtype=segment[0].dtype,
                                               persistable=False,
                                               stop_gradient=True)
                    fused_vars.append(tmp_var)
                    block._insert_op(idx,
                                     type="coalesce_tensor",
                                     inputs={"Input": segment},
                                     outputs={
                                         "Output": segment,
                                         "FusedOutput": tmp_var
                                     },
                                     attrs={
                                         "copy_data": True,
                                         "use_align": True,
                                         "dtype": segment[0].dtype,
                                         self.op_role_key: OpRole.Backward
                                     })
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                break

        # insert the allreduce_sum op
        for idx, op in enumerate(block.ops):
            if self._is_optimizer_op(op):
                for fused_var in fused_vars:
<<<<<<< HEAD
                    block._insert_op(
                        idx,
                        type='c_allreduce_sum',
                        inputs={'X': fused_var},
                        outputs={'Out': fused_var},
                        attrs={
                            'ring_id': ring_id,
                            'use_calc_stream': False,
                            self.op_role_key: OpRole.Backward,
                        },
                    )
                    block._insert_op(
                        idx,
                        type='c_sync_calc_stream',
                        inputs={'X': fused_var},
                        outputs={'Out': fused_var},
                        attrs={self.op_role_key: OpRole.Backward},
                    )
=======
                    block._insert_op(idx,
                                     type='c_allreduce_sum',
                                     inputs={'X': fused_var},
                                     outputs={'Out': fused_var},
                                     attrs={
                                         'ring_id': ring_id,
                                         'use_calc_stream': False,
                                         self.op_role_key: OpRole.Backward
                                     })
                    block._insert_op(idx,
                                     type='c_sync_calc_stream',
                                     inputs={'X': fused_var},
                                     outputs={'Out': fused_var},
                                     attrs={self.op_role_key: OpRole.Backward})
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                break

        if len(fused_vars) == 0:
            block._sync_with_cpp()
            return

        # insert the sync comm op
        for idx, op in enumerate(block.ops):
            if self._is_optimizer_op(op):
<<<<<<< HEAD
                block._insert_op(
                    idx,
                    type='c_sync_comm_stream',
                    inputs={'X': fused_vars[0]},
                    outputs={'Out': fused_vars[0]},
                    attrs={
                        'ring_id': ring_id,
                        self.op_role_key: OpRole.Backward,
                    },
                )
=======
                block._insert_op(idx,
                                 type='c_sync_comm_stream',
                                 inputs={'X': fused_vars[0]},
                                 outputs={'Out': fused_vars[0]},
                                 attrs={
                                     'ring_id': ring_id,
                                     self.op_role_key: OpRole.Backward
                                 })
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                break
        block._sync_with_cpp()
