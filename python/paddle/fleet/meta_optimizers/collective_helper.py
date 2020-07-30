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

import paddle.fluid as fluid
from paddle.fluid import core, unique_name
from paddle.fluid.transpiler.details import wait_server_ready

OpRole = core.op_proto_and_checker_maker.OpRole


class CollectiveHelper(object):
    def __init__(self, nrings=1, wait_port='6174'):
        self.nrings = nrings
        self.wait_port = wait_port

        op_maker = core.op_proto_and_checker_maker
        self.op_role_key = op_maker.kOpRoleAttrName()
        self.op_role_var_key = op_maker.kOpRoleVarAttrName()

    def update_startup_program(self, startup_program=None):
        if startup_program is None:
            startup_program = fluid.default_startup_program()

        import paddle.fleet as fleet
        endpoints = fleet.get_trainer_endpoints()
        current_endpoint = fleet.current_trainer_endpoint()
        for ring_id in range(self.nrings):
            self._init_communicator(
                startup_program, current_endpoint, endpoints,
                fleet.worker_index(), ring_id, self.wait_port)
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

    def is_loss_grad_op(self, op):
        if self.op_role_key not in op.attr_names:
            return False
        op_role = int(op.all_attrs()[self.op_role_key])
        return op_role & int(OpRole.Backward) and op_role & int(OpRole.Loss)

    def is_backward_op(self, op):
        return self.op_role_key in op.attr_names and \
                int(op.all_attrs()[self.op_role_key]) & int(OpRole.Backward)

    def is_update_op(self, op):
        return 'Param' in op.input_names and 'Grad' in op.input_names and \
                "LearningRate" in op.input_names

    def is_optimizer_op(self, op):
        return self.op_role_key in op.attr_names and \
                int(op.all_attrs()[self.op_role_key]) & int(OpRole.Optimize)
