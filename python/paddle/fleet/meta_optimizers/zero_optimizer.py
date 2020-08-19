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
from paddle.fluid.contrib.mixed_precision.decorator import OptimizerWithMixedPrecision
import paddle.fluid as fluid

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
        self._fuse_broadcast_num = 20
        self._fuse_broadcast_MB_bytes = 64
        self._dtype_to_size = {
            core.VarDesc.VarType.FP16: 2,
            core.VarDesc.VarType.FP32: 4,
            core.VarDesc.VarType.FP64: 8,
            core.VarDesc.VarType.INT16: 2,
            core.VarDesc.VarType.INT32: 4,
            core.VarDesc.VarType.INT64: 8,
            core.VarDesc.VarType.BOOL: 1,
            core.VarDesc.VarType.UINT8: 1,
        }

    def _get_var_size(self, param):
        """
        input:
            - param: var
        return:
            var size in Bytes
        """
        assert -1 not in param.shape
        return reduce(
            lambda x, y: x * y,
            param.shape) * self._dtype_to_size[param.dtype] / 1024.0 / 1024.0

    def _can_apply(self):
        return self.user_defined_strategy.zero

    def _disable_strategy(self, dist_strategy):
        dist_strategy.zero = False

    def _split_params(self, params_grads):
        total_param_mem = 0.0
        param2mem = []
        for param, _ in params_grads:
            mem = self._get_var_size(param)
            total_param_mem += mem
            param2mem.append((param.name, mem))
            # print(param.name, mem)
        # print("total_param_mem: ", total_param_mem)
        device_num = self.role_maker.worker_num()
        # print("device_num: ", device_num)
        self._device2params = {x: [] for x in range(device_num)}
        device_idx = 0
        mem_accu = 0.0
        for param_name, mem in param2mem:
            # print("accu", param_name, mem)
            # print("mem_accu: ", mem_accu)
            # print("device_idx + 1: ", device_idx + 1)
            # print("tgt: ", total_param_mem * 1.0 * (device_idx + 1) / device_num)
            if mem_accu > total_param_mem * 1.0 * (device_idx + 1) / device_num:
                device_idx += 1
            self._device2params[device_idx].append(param_name)
            self._param2device[param_name] = device_idx
            mem_accu += mem
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

    def _insert_broadcast_ops(self, block, fuse_broadcast=False):
        def _insert_comm_sync(insert_idx, dummy_var_name):
            for r in range(self._nrings):
                block._insert_op(
                    insert_idx,
                    type='c_sync_comm_stream',
                    inputs={'X': dummy_var_name},
                    outputs={'Out': dummy_var_name},
                    attrs={'ring_id': r,
                           OP_ROLE_KEY: OpRole.Backward})
                insert_idx += 1
            return self._nrings

        def _insert_broadcast_inner(insert_idx, attrs):
            for attr in attrs:
                block._insert_op(insert_idx, **attr)
                insert_idx += 1
            return len(attrs)

        def _insert_fill_constant(insert_idx, attrs, dummy_var_name):
            for attr in attrs:
                block._insert_op(insert_idx, **attr)
                insert_idx += 1
            block._insert_op(
                insert_idx,
                type='c_sync_calc_stream',
                inputs={'X': dummy_var_name},
                outputs={'Out': dummy_var_name},
                attrs={OP_ROLE_KEY: OpRole.Backward})
            return len(attrs) + 1

        def _insert_cache(cache,
                          prepend_comm_sync=False,
                          append_comm_sync=False):
            insert_idx = cache["insert_idx"]
            dummy_var_name = cache["dummy_var_name"]
            assert (len(cache["broadcast_ops"]) > 0)

            if prepend_comm_sync:
                insert_idx += _insert_comm_sync(insert_idx, dummy_var_name)

            if len(cache["fill_constant_ops"]) > 0:
                insert_idx += _insert_fill_constant(
                    insert_idx, cache["fill_constant_ops"], dummy_var_name)

            insert_idx += _insert_broadcast_inner(insert_idx,
                                                  cache["broadcast_ops"])

            if append_comm_sync:
                insert_idx += _insert_comm_sync(insert_idx, dummy_var_name)

            return insert_idx - cache["insert_idx"]

        ring_id = -1
        broadcast_caches = []
        cache = {
            "fill_constant_ops": [],
            "broadcast_ops": [],
            "insert_idx": 0,
            "dummy_var_name": "",
            "param2broadcast": {},
            "mem_accu": 0.0,
        }
        for op_idx, op in reversed(list(enumerate(block.ops))):
            if cache["mem_accu"] >= self._fuse_broadcast_MB_bytes:
                # if len(cache["broadcast_ops"]) > self._fuse_broadcast_num:
                cache["insert_idx"] = op_idx
                broadcast_caches.insert(0, cache)
                cache = {
                    "fill_constant_ops": [],
                    "broadcast_ops": [],
                    "insert_idx": 0,
                    "dummy_var_name": "",
                    "param2broadcast": {},
                    "mem_accu": 0.0,
                }
            if int(op.attr('op_role')) == int(OpRole.Optimize):
                continue
            for input_name in op.desc.input_arg_names():
                if input_name not in self._param2device:
                    continue
                root_device = self._param2device[input_name]
                if input_name in cache["param2broadcast"]:
                    # skip broadcast because it reuse the old broadcast var
                    broadcast_name = cache["param2broadcast"][input_name]
                    if input_name != broadcast_name:
                        op._rename_input(input_name, broadcast_name)
                    continue
                if root_device == self.role_maker.worker_index():
                    broadcast_var = self._varname2param[input_name]
                else:
                    broadcast_var = block.create_var(
                        name=unique_name.generate(input_name + "@BroadCast"),
                        shape=self._varname2param[input_name].shape,
                        dtype=self._varname2param[input_name].dtype,
                        persistable=False)
                if input_name != broadcast_var.name:
                    op._rename_input(input_name, broadcast_var.name)
                cache["param2broadcast"][input_name] = broadcast_var.name
                cache["mem_accu"] += self._get_var_size(broadcast_var)
                cache["dummy_var_name"] = broadcast_var.name

                print("main_block insert broadcast op for %s" %
                      broadcast_var.name)
                # TODO(mapingshuo) OP_ROLE_KEY should be forward if the param
                # is used in forward network
                ring_id = (ring_id + 1) % self._nrings
                cache["broadcast_ops"].insert(0, {
                    "type": 'c_broadcast',
                    "inputs": {
                        'X': broadcast_var.name
                    },
                    "outputs": {
                        'Out': broadcast_var.name
                    },
                    "attrs": {
                        'ring_id': ring_id,
                        'root': root_device,
                        OP_ROLE_KEY: OpRole.Forward
                    }
                })
                if root_device != self.role_maker.worker_index():
                    cache["fill_constant_ops"].insert(0, {
                        "type": "fill_constant",
                        "outputs": {
                            "Out": broadcast_var.name
                        },
                        "attrs": {
                            "shape": broadcast_var.shape,
                            "dtype": broadcast_var.dtype,
                            "value": 0.0,
                        }
                    })
        if len(cache["broadcast_ops"]) > 0:
            cache["insert_idx"] = 0
            broadcast_caches.insert(0, cache)

        print("insert_idx: ", [x["insert_idx"] for x in broadcast_caches])
        move_ahead = 1
        for idx, cache in reversed(list(enumerate(broadcast_caches))):
            if idx < move_ahead:
                cache["insert_idx"] = 0
            else:
                cache["insert_idx"] = broadcast_caches[idx - move_ahead][
                    "insert_idx"]
        print("insert_idx: ", [x["insert_idx"] for x in broadcast_caches])

        inserted_op_num = 0
        for idx, cache in enumerate(broadcast_caches):
            prepend_comm_sync = True
            append_comm_sync = True
            cache["insert_idx"] += inserted_op_num
            inserted_op_num += _insert_cache(
                cache,
                prepend_comm_sync=prepend_comm_sync,
                append_comm_sync=append_comm_sync)
        return

    def _insert_allreduce_ops(self, block):
        ring_id = -1
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
                    ring_id = (ring_id + 1) % self._nrings
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
                for ring_id in range(self._nrings):
                    block._insert_op(
                        idx + ring_id,
                        type='c_sync_comm_stream',
                        inputs={'X': grad},
                        outputs={'Out': grad},
                        attrs={
                            'ring_id': ring_id,
                            OP_ROLE_KEY: OpRole.Backward
                        })
                break

    def _check_broadcast(self, block):
        """
        if a var is broadcasted, it should have a sync_comm before
        this var is used, if not, raise error.
        if the broadcasted var has a fill_constant op, the fill_constant
        op should stay forward before the broadcast op, and before a
        sync_calc op. Otherwise, raise error.
        """
        broadcast_vars = {}
        for idx, op in enumerate(block.ops):
            if op.type == "c_broadcast":
                var_name = op.desc.input_arg_names()[0]
                if "@BroadCast" in var_name:
                    if var_name in broadcast_vars:
                        print("error: var_name areadly exist: ", var_name)
                        print("the old pos is ",
                              broadcast_vars[var_name]["broadcast_pos"])
                        print("the new pos is ", idx)
                    assert (var_name not in broadcast_vars)
                    broadcast_vars[var_name] = {
                        "fill_constant_pos": -1,
                        "broadcast_pos": idx,
                    }

        for idx, op in enumerate(block.ops):
            if op.type == "fill_constant":
                var_name = op.desc.output_arg_names()[0]
                if var_name in broadcast_vars:
                    broadcast_vars[var_name]["fill_constant_pos"] = idx
                continue

        last_sync_comm_op_idx = -1
        last_sync_calc_op_idx = -1
        for idx, op in enumerate(block.ops):
            if op.type == "c_sync_comm_stream":
                last_sync_comm_op_idx = idx
                continue
            if op.type == "c_sync_calc_stream":
                last_sync_calc_op_idx = idx
                continue
            if op.type == "c_broadcast":
                var_name = op.desc.input_arg_names()[0]
                if "@BroadCast" in var_name:
                    if broadcast_vars[var_name]["fill_constant_pos"] != -1:
                        # print("before_broadcast: ", var_name, 
                        #         broadcast_vars[var_name]["fill_constant_pos"],
                        #         last_sync_calc_op_idx,
                        #         broadcast_vars[var_name]["broadcast_pos"],
                        #         )
                        assert (last_sync_calc_op_idx != -1)
                        assert (broadcast_vars[var_name]["fill_constant_pos"] <
                                last_sync_calc_op_idx)
                        assert (last_sync_calc_op_idx < idx)
                    continue
            for input_name in op.desc.input_arg_names():
                if input_name in broadcast_vars:
                    # print("after_broadcast: ",
                    #         broadcast_vars[input_name]["broadcast_pos"],
                    #         last_sync_comm_op_idx,
                    #         idx)
                    assert (broadcast_vars[input_name]["broadcast_pos"] != -1)
                    assert (broadcast_vars[input_name]["broadcast_pos"] <
                            last_sync_comm_op_idx)
                    assert (last_sync_comm_op_idx < idx)
        print("check done: ")
        return

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        self._nrings = 10

        optimizer = fluid.contrib.mixed_precision.decorate(self.inner_opt)
        optimize_ops, params_grads = optimizer.minimize(
            loss, startup_program, parameter_list, no_grad_set)

        if startup_program is None:
            startup_program = default_startup_program()
        main_block = loss.block
        startup_block = startup_program.global_block()

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
        startup_block._sync_with_cpp()

        # step2: split params
        self._varname2param = {x[0].name: x[0] for x in params_grads}
        self._varname2grad = {x[1].name: x[1] for x in params_grads}
        self._split_params(params_grads)
        print(self._device2params)

        # step3: remove ops that generate params
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
        startup_block._sync_with_cpp()

        for idx, op in reversed(list(enumerate(main_block.ops))):
            for output_name in op.desc.output_arg_names():
                if (output_name in self._varname2param) and \
                    (output_name not in self._device2params[self.role_maker.worker_index()]):
                    print("%d: main_block remove op %s" %
                          (self.role_maker.worker_index(), op.type))
                    main_block._remove_op(idx)
                    break
        main_block._sync_with_cpp()

        # step4 add broadcast ops
        self._insert_broadcast_ops(main_block)
        main_block._sync_with_cpp()

        # step5: remove Parameter from main program and startup program
        for var_name, var in main_block.vars.items():
            var_device_id = self._var_device_id(var_name)
            if var_device_id == -1 or var_device_id == self.role_maker.worker_index(
            ):
                continue
            print("%d: main_block remove %s" %
                  (self.role_maker.worker_index(), var_name))
            main_block._remove_var(var_name)
        main_block._sync_with_cpp()

        for var_name, var in startup_block.vars.items():
            var_device_id = self._var_device_id(var_name)
            if var_device_id == -1 or var_device_id == self.role_maker.worker_index(
            ):
                continue
            print("%d: startup_block remove %s" %
                  (self.role_maker.worker_index(), var_name))
            startup_block._remove_var(var_name)
        startup_block._sync_with_cpp()

        # step6: insert reduce_sum for grad
        self._insert_scale_loss_grad_ops(
            main_block, scale=1.0 / self.role_maker.worker_num())
        main_block._sync_with_cpp()
        self._insert_allreduce_ops(main_block)
        main_block._sync_with_cpp()
        print("insert allreduce done")
        self._check_broadcast(main_block)
        return optimize_ops, params_grads

    def _broadcast_params(self, block):
        ring_id = -1
        for param in block.iter_parameters():
            if param.is_distributed:
                continue
            ring_id = (ring_id + 1) % self._nrings
            block.append_op(
                type='c_broadcast',
                inputs={'X': param},
                outputs={'Out': param},
                attrs={
                    'ring_id': ring_id,
                    'root': 0,
                    OP_ROLE_KEY: OpRole.Forward
                })
        for ring_id in range(self._nrings):
            block.append_op(
                type='c_sync_comm_stream',
                inputs={'X': param},
                outputs={'Out': param},
                attrs={'ring_id': ring_id,
                       OP_ROLE_KEY: OpRole.Forward})

    def minimize_impl_allreduce(self,
                                loss,
                                startup_program=None,
                                parameter_list=None,
                                no_grad_set=None):

        self._nrings = 3
        optimize_ops, params_grads = self.inner_opt.minimize(
            loss, startup_program, parameter_list, no_grad_set)

        if startup_program is None:
            startup_program = default_startup_program()

        print("work idx: ", self.role_maker.worker_index())
        endpoints = self.role_maker.get_trainer_endpoints()
        current_endpoint = endpoints[self.role_maker.worker_index()]

        collective_helper = CollectiveHelper(self.role_maker, self._nrings)
        for ring_id in range(self._nrings):
            collective_helper._init_communicator(
                startup_program, current_endpoint, endpoints,
                self.role_maker.worker_index(), ring_id, '6174')
        main_block = loss.block
        startup_block = startup_program.global_block()
        self._broadcast_params(startup_block)

        self._insert_scale_loss_grad_ops(
            main_block, scale=1.0 / self.role_maker.worker_num())
        self._insert_allreduce_ops(main_block)
        print("insert allreduce done")
        return optimize_ops, params_grads
