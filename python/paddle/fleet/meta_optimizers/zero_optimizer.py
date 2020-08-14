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

from .common import OpRole, OP_ROLE_KEY, OP_ROLE_VAR_KEY, CollectiveHelper
from .common import is_update_op, is_loss_grad_op, is_backward_op, is_optimizer_op
from .meta_optimizer_base import MetaOptimizerBase
from paddle.fluid import unique_name, core

import math
import re

__all__ = ["ZeroOptimizer"]


class ZeroOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(ZeroOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = []
        # device_id(int) -> param_name(set)
        self._device2params = {}
        # param_name(str) -> device_id(int)
        self._param2device = {}
        # varname(str) -> param(Variable)
        self._varname2param = {}
        # varname(str) -> grad(Variable)
        self._varname2grad = {}
        # self._nrings(int) is for nccl communicate
        self._nrings = 1

    def _can_apply(self):
        return self.user_defined_strategy.zero

    def _disable_strategy(self, dist_strategy):
        dist_strategy.zero = False

    def _split_params(self, params_grads):
        device_num = self.role_maker.worker_num()
        param_num = len(params_grads)
        param_sub_num = int(math.ceil(1.0 * param_num / device_num))
        for i in range(device_num):
            param_names = [
                x[0].name
                for x in params_grads[(i * param_sub_num):(i + 1) *
                                      param_sub_num]
            ]
            self._device2params[i] = param_names
            for param_name in param_names:
                self._param2device[param_name] = i
        return

    def _is_opti_var(self, var_name):
        if var_name in self._varname2param:
            return True
        for suffix in [
                "_moment1_0", "_moment2_0", "_beta1_pow_acc_0",
                "_beta2_pow_acc_0"
        ]:
            base_name = re.sub(suffix, '', var_name)
            if base_name in self._varname2param:
                return True
        return False

    def _var_device_id(self, var_name):
        if not self._is_opti_var(var_name):
            return -1
        if var_name in self._param2device:
            return self._param2device[var_name]
        for suffix in [
                "_moment1_0", "_moment2_0", "_beta1_pow_acc_0",
                "_beta2_pow_acc_0"
        ]:
            base_name = re.sub(suffix, '', var_name)
            if base_name in self._param2device:
                return self._param2device[base_name]
        return -1

    def _insert_scale_loss_grad_ops(self, block, scale=1.0):
        '''
        In order to keep the learning rate consistent in different numbers of
        training workers, we scale the loss grad by the number of workers
        '''
        for idx, op in reversed(list(enumerate(block.ops))):
            if is_loss_grad_op(op):
                loss_grad_var = block.vars[op.output_arg_names[0]]
                block._insert_op(
                    idx + 1,
                    type='scale',
                    inputs={'X': loss_grad_var},
                    outputs={'Out': loss_grad_var},
                    attrs={'scale': scale,
                           OP_ROLE_KEY: OpRole.Backward})

    def _insert_broadcast_ops(self, block):
        for op_idx, op in reversed(list(enumerate(block.ops))):
            if int(op.attr('op_role')) == int(OpRole.Optimize):
                continue
            for input_name in op.desc.input_arg_names():
                if input_name in self._param2device:
                    root_device = self._param2device[input_name]
                    if self._param2device[
                            input_name] == self.role_maker.worker_index():
                        param = self._varname2param[input_name]
                    else:
                        broadcast_name = unique_name.generate(input_name +
                                                              "@BroadCast")
                        op._rename_input(input_name, broadcast_name)
                        param = block.create_var(
                            name=broadcast_name,
                            shape=self._varname2param[input_name].shape,
                            dtype=self._varname2param[input_name].dtype,
                            persistable=False)
                    print("main_block insert broadcast op for %s" % param.name)
                    # TODO(mapingshuo) OP_ROLE_KEY should be forward if the param
                    # is used in forward network
                    block._insert_op(
                        op_idx,
                        type='c_sync_comm_stream',
                        inputs={'X': param},
                        outputs={'Out': param},
                        attrs={'ring_id': 0,
                               OP_ROLE_KEY: OpRole.Backward})
                    block._insert_op(
                        op_idx,
                        type='c_broadcast',
                        inputs={'X': param},
                        outputs={'Out': param},
                        attrs={
                            'ring_id': 0,
                            'root': root_device,
                            OP_ROLE_KEY: OpRole.Forward
                        })
                    if root_device != self.role_maker.worker_index():
                        block._insert_op(
                            op_idx,
                            type='c_sync_calc_stream',
                            inputs={'X': param},
                            outputs={'Out': param},
                            attrs={'ring_id': 0,
                                   OP_ROLE_KEY: OpRole.Backward})
                        block._insert_op(
                            op_idx,
                            type="fill_constant",
                            outputs={"Out": param},
                            attrs={
                                "shape": param.shape,
                                "dtype": param.dtype,
                                "value": 0.0,
                            })

    def _insert_allreduce_ops(self, block):
        ring_id = 0
        grad = None
        for idx, op in reversed(list(enumerate(block.ops))):
            if is_backward_op(op) and \
                    OP_ROLE_VAR_KEY in op.attr_names:
                op_role_var = op.all_attrs()[OP_ROLE_VAR_KEY]

                if len(op_role_var) == 0:
                    continue
                assert len(op_role_var) % 2 == 0

                offset = idx
                for i in range(0, len(op_role_var), 2):
                    # param = block.vars[op_role_var[i]]
                    grad = block.vars[op_role_var[i + 1]]
                    # TODO(mapingshuo): what is is_distributed
                    # if param.is_distributed:
                    #     continue

                    if offset == idx:
                        offset += 1
                        block._insert_op(
                            offset,
                            type='c_sync_calc_stream',
                            inputs={'X': grad},
                            outputs={'Out': grad},
                            attrs={OP_ROLE_KEY: OpRole.Backward})
                        offset += 1

                    # As we search ops reversedly, we should insert c_allreduce_sum
                    # op in the same way to keep the ring_id alternate
                    print("add allreduce op for {}".format(grad.name))
                    block._insert_op(
                        offset,
                        type='c_allreduce_sum',
                        inputs={'X': grad},
                        outputs={'Out': grad},
                        attrs={
                            'ring_id': ring_id,
                            OP_ROLE_KEY: OpRole.Backward
                        })

        if grad is None:
            return

        for idx, op in enumerate(block.ops):
            if is_optimizer_op(op):
                block._insert_op(
                    idx + ring_id,
                    type='c_sync_comm_stream',
                    inputs={'X': grad},
                    outputs={'Out': grad},
                    attrs={'ring_id': ring_id,
                           OP_ROLE_KEY: OpRole.Backward})
                break

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        optimize_ops, params_grads = self.inner_opt.minimize(
            loss, startup_program, parameter_list, no_grad_set)

        if startup_program is None:
            startup_program = default_startup_program()

        # step1: initialize nccl
        # TODO(mapingshuo) fix get_trainer_endpoints
        print("work idx: ", self.role_maker.worker_index())
        endpoints = self.role_maker.get_trainer_endpoints()
        current_endpoint = endpoints[self.role_maker.worker_index()]
        collective_helper = CollectiveHelper(self.role_maker, self._nrings)
        for ring_id in range(self._nrings):
            collective_helper._init_communicator(
                startup_program, current_endpoint, endpoints,
                self.role_maker.worker_index(), ring_id, '6174')

        # step2: split params
        self._varname2param = {x[0].name: x[0] for x in params_grads}
        self._varname2grad = {x[1].name: x[1] for x in params_grads}
        self._split_params(params_grads)
        print(self._device2params)

        # step3: remove ops that generate params
        main_block = loss.block
        startup_block = startup_program.global_block()
        for idx, op in reversed(list(enumerate(startup_block.ops))):
            for output_name in op.desc.output_arg_names():
                var_device_id = self._var_device_id(output_name)
                if var_device_id == -1 or var_device_id == self.role_maker.worker_index(
                ):
                    continue
                print("%d: startup_block remove op %s" %
                      (self.role_maker.worker_index(), op.type))
                startup_block._remove_op(idx)
                break

        for idx, op in reversed(list(enumerate(main_block.ops))):
            for output_name in op.desc.output_arg_names():
                if (output_name in self._varname2param) and \
                    (output_name not in self._device2params[self.role_maker.worker_index()]):
                    print("%d: main_block remove op %s" %
                          (self.role_maker.worker_index(), op.type))
                    main_block._remove_op(idx)
                    break

        # step4 add broadcast ops
        self._insert_broadcast_ops(main_block)

        # step5: remove Parameter from main program and startup program
        for var_name, var in main_block.vars.items():
            var_device_id = self._var_device_id(var_name)
            if var_device_id == -1 or var_device_id == self.role_maker.worker_index(
            ):
                continue
            print("%d: main_block remove %s" %
                  (self.role_maker.worker_index(), var_name))
            main_block._remove_var(var_name)

        for var_name, var in startup_block.vars.items():
            var_device_id = self._var_device_id(var_name)
            if var_device_id == -1 or var_device_id == self.role_maker.worker_index(
            ):
                continue
            print("%d: startup_block remove %s" %
                  (self.role_maker.worker_index(), var_name))
            startup_block._remove_var(var_name)

        # step6: insert reduce_sum for grad
        self._insert_scale_loss_grad_ops(
            main_block, scale=1.0 / self.role_maker.worker_num())
        self._insert_allreduce_ops(main_block)
        print("insert allreduce done")

        return optimize_ops, params_grads

    # def _broadcast_params(self, block):
    #     ring_id = 0
    #     for param in block.iter_parameters():
    #         if param.is_distributed:
    #             continue

    #         block.append_op(
    #             type='c_broadcast',
    #             inputs={'X': param},
    #             outputs={'Out': param},
    #             attrs={
    #                 'ring_id': ring_id,
    #                 'root': 0,
    #                 OP_ROLE_KEY: OpRole.Forward
    #             })

    #     block.append_op(
    #         type='c_sync_comm_stream',
    #         inputs={'X': param},
    #         outputs={'Out': param},
    #         attrs={'ring_id': ring_id,
    #                OP_ROLE_KEY: OpRole.Forward})

    # def minimize_impl_allreduce(self,
    #                             loss,
    #                             startup_program=None,
    #                             parameter_list=None,
    #                             no_grad_set=None):

    #     optimize_ops, params_grads = self.inner_opt.minimize(
    #         loss, startup_program, parameter_list, no_grad_set)

    #     if startup_program is None:
    #         startup_program = default_startup_program()

    #     print("work idx: ", self.role_maker.worker_index())
    #     endpoints = self.role_maker.get_trainer_endpoints()
    #     current_endpoint = endpoints[self.role_maker.worker_index()]

    #     collective_helper = CollectiveHelper(self.role_maker, self._nrings)
    #     for ring_id in range(self._nrings):
    #         collective_helper._init_communicator(
    #             startup_program, current_endpoint, endpoints,
    #             self.role_maker.worker_index(), ring_id, '6174')
    #     main_block = loss.block
    #     startup_block = startup_program.global_block()
    #     self._broadcast_params(startup_block)

    #     self._insert_scale_loss_grad_ops(
    #         main_block, scale=1.0 / self.role_maker.worker_num())
    #     self._insert_allreduce_ops(main_block)
    #     print("insert allreduce done")
    #     return optimize_ops, params_grads
