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


def _pretty_op_desc_(op_desc, prefix):
    out_s = "%s\tname:[%s]\n%s    \tinputs:[%s]\n%s    \toutputs:[%s]" % \
            (prefix + "_op", str(op_desc.type()), prefix + "_input", " ".join(op_desc.input_arg_names()),
             prefix + "_output", " ".join(op_desc.output_arg_names()))
    return out_s


class SubProgram(object):
    def __init__(self, block):
        self._block = block
        self._allreduce_vars = []
        self._allreduce_ops = []
        self._fill_constant_ops = []
        self._broadcast_ops = []
        # sub program start idx
        self._start_idx = -1
        # sub program end idx
        self._end_idx = -1
        # reduce var names
        self._sync_reduce_var_names = []
        # param name to broadcast name
        self._param2broadcast = {}
        # parameter mems
        self._param_mem = 0.0


class ProgramDeps(object):
    def __init__(self, block, start_vars, end_vars):
        self._block = block
        # vars where to start to build the deps
        self._start_vars = start_vars
        # vars where to stop to build the deps
        self._end_vars = end_vars
        # var name -> op idxs which depends on this var
        self._var_deps = {}
        # sub block deps which is a subset of this topo
        self._sub_block_deps = {}
        # var name -> op idxs which generate var
        self._var_to_generate_op = {}
        self._should_removed_var = set()
        self._father_block_deps = None
        self._build_deps()

    def get_sub_block_deps(self, idx):
        if idx in self._sub_block_deps:
            return self._sub_block_deps[idx]
        else:
            return None

    def get_var_deps(self, var_name):
        if var_name in self._var_deps:
            return self._var_deps[var_name]
        else:
            return None

    def _build_deps(self, ):
        for var_name in self._start_vars:
            self._var_deps[var_name] = [-1]
            self._var_to_generate_op[var_name] = [-1]

        for idx, op in enumerate(self._block.ops):
            if op.type in [
                    "c_allreduce_sum", "c_sync_comm_stream",
                    "c_calc_comm_stream"
            ]:
                continue
            input_vars = op.desc.input_arg_names()
            output_vars = op.desc.output_arg_names()
            deps_reduce = False
            for input_name in input_vars:
                if input_name in self._var_deps:
                    deps_reduce = True
            if deps_reduce:
                for input_name in input_vars:
                    if input_name in self._var_deps:
                        self._var_deps[input_name].append(idx)
                for output_name in output_vars:
                    self._var_deps[output_name] = []
                    if output_name not in self._var_to_generate_op:
                        self._var_to_generate_op[output_name] = [idx]
                    else:
                        self._var_to_generate_op[output_name].append(idx)
                if op.type == "conditional_block":
                    # subblock
                    assert (op.desc.has_attr("sub_block"))
                    subblock_idx = op.desc.attr("sub_block").id
                    subblock_deps = ProgramDeps(
                        self._block.program.block(subblock_idx),
                        op.desc.input_arg_names(), op.desc.output_arg_names())
                    self._sub_block_deps[subblock_idx] = subblock_deps
                    subblock_deps._father_block_deps = self

    def crop_input_var_from_op(self, op_idx, var_name):
        if var_name in self._var_deps:
            # update var -> dep_var_op
            if self._var_deps[var_name] != []:
                assert (op_idx in self._var_deps[var_name])
                self._var_deps[var_name].remove(op_idx)
            # update _should_removed_var
            if var_name in self._start_vars:
                self._should_removed_var.discard(var_name)
            elif self._var_deps[var_name] == []:  # no more deps of this var
                self._should_removed_var.add(var_name)
            elif self._var_to_generate_op[var_name][-1] >= self._var_deps[
                    var_name][-1]:
                # there are circle in the graph
                self._should_removed_var.add(var_name)
            else:  # input_name should not be deleted
                self._should_removed_var.discard(var_name)

    def crop_output_var_from_op(self, op_idx, var_name):
        if var_name in self._var_to_generate_op:
            assert (op_idx in self._var_to_generate_op[var_name])
            self._var_to_generate_op[var_name].remove(op_idx)
        if self._block.has_var(var_name) and self._var_to_generate_op[
                var_name] == []:
            print("main_block remove var {}".format(var_name))
            self._block._remove_var(var_name)

    def remove_op(self, op_idx):
        # update deps
        op = self._block.ops[op_idx]
        print("main_block remove op {}".format(op.type))
        for input_name in op.desc.input_arg_names():
            self.crop_input_var_from_op(op_idx, input_name)
        for output_name in op.desc.output_arg_names():
            self.crop_output_var_from_op(op_idx, output_name)
        self._block._remove_op(op_idx)

    def should_remove_op(self, op_idx):
        op = self._block.ops[op_idx]
        for output_name in op.desc.output_arg_names():
            if output_name not in self._should_removed_var:
                return False
        return True


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
        # reduced grads to param name
        self._reduced_grads_to_param = {}
        # self._nrings(int) is for nccl communicate
        self._nrings = 1
        # self._sub_progs
        self._sub_progs = []
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

    def _split_program(self, block):
        for op_idx, op in reversed(list(enumerate(block.ops))):
            if int(op.attr('op_role')) != int(OpRole.Optimize):
                last_backward_op_idx = op_idx + 1
                break
        sub_prog = SubProgram(block)
        sub_prog._end_idx = last_backward_op_idx
        for op_idx in reversed(range(last_backward_op_idx)):
            op = block.ops[op_idx]
            assert (int(op.attr('op_role')) != int(OpRole.Optimize))
            if sub_prog._param_mem >= self._fuse_broadcast_MB_bytes:
                sub_prog._start_idx = op_idx + 1
                self._sub_progs.insert(0, sub_prog)
                sub_prog = SubProgram(block)
                sub_prog._end_idx = op_idx + 1

            # find broadcast vars
            for input_name in op.desc.input_arg_names():
                if input_name not in self._param2device:
                    continue
                root_device = self._param2device[input_name]
                if input_name in sub_prog._param2broadcast:
                    # skip broadcast because it reuse the old broadcast var
                    broadcast_name = sub_prog._param2broadcast[input_name]
                    if input_name != broadcast_name:
                        op._rename_input(input_name, broadcast_name)
                    continue
                if root_device == self.role_maker.worker_index():
                    broadcast_var_name = input_name
                else:
                    broadcast_var_name = unique_name.generate(input_name +
                                                              "@BroadCast")
                sub_prog._param2broadcast[input_name] = broadcast_var_name
                sub_prog._param_mem += self._get_var_size(self._varname2param[
                    input_name])

            # find reduce vars
            if is_backward_op(op) and \
                    OP_ROLE_VAR_KEY in op.attr_names:
                op_role_var = op.all_attrs()[OP_ROLE_VAR_KEY]
                if len(op_role_var) != 0:
                    assert len(op_role_var) % 2 == 0
                    for i in range(0, len(op_role_var), 2):
                        param, reduced_grad = op_role_var[i], op_role_var[i + 1]
                        sub_prog._allreduce_vars.append(reduced_grad)
                        assert (
                            reduced_grad not in self._reduced_grads_to_param)
                        self._reduced_grads_to_param[reduced_grad] = param

        if sub_prog._param_mem > 0:
            sub_prog._start_idx = 0
            self._sub_progs.insert(0, sub_prog)
        return

    def _is_gradient_clip_sum_op(self, op):
        return op.type == "sum" and op.desc.has_attr("op_namescope") \
            and op.desc.attr("op_namescope").startswith("/gradient_clip_@CLIP")

    def _is_amp_sum_op(self, op):
        return op.type == "sum" and op.desc.has_attr("op_namescope") \
            and op.desc.attr("op_namescope").startswith("/mixed_precision")

    def _is_amp_subblock(self, op):
        return op.type == "conditional_block" and op.desc.has_attr("op_namescope") \
            and op.desc.attr("op_namescope").startswith("/mixed_precision")

    def _prune_main_program(self, block):
        """
        calculate deps from allredce op to optimize op,
        remove ops and vars not needed in this worker
        """
        with open("debug_program", 'w') as f:
            f.write(str(block.program))
        # build prog deps
        reduced_grads = []
        var_to_reduce_var = {}
        for idx, op in enumerate(block.ops):
            input_names = op.desc.input_arg_names()
            output_names = op.desc.output_arg_names()
            if op.type == "c_allreduce_sum":
                assert (len(output_names) == 1)
                output_name = output_names[0]
                reduced_grads.append(output_name)
                var_to_reduce_var[output_name] = output_name
            else:
                non_persistable_input = [
                    x for x in input_names if not block.var(x).persistable
                ]
                if len(non_persistable_input) == 1 and len(
                        output_names) == 1 and non_persistable_input[
                            0] in var_to_reduce_var:
                    var_to_reduce_var[output_names[0]] = var_to_reduce_var[
                        non_persistable_input[0]]

        params = []
        for var_name, _ in block.vars.items():
            if self._is_opti_var(var_name) and \
                self._var_device_id(var_name) != self.role_maker.worker_index():
                params.append(var_name)
        program_deps = ProgramDeps(block, reduced_grads, params)

        # Init
        for var_name in program_deps._end_vars:
            program_deps._should_removed_var.add(var_name)

        # Prune
        for idx, op in reversed(list(enumerate(block.ops))):
            if op.type in [
                    "c_allreduce_sum", "c_sync_comm_stream",
                    "c_calc_comm_stream", "c_gen_nccl_id", "c_comm_init"
            ]:
                pass
            elif self._is_gradient_clip_sum_op(op) or self._is_amp_sum_op(op):
                print("sum op: ")
                print(op)
                reversed_input_vars = []
                for input_name in op.desc.input_arg_names():
                    assert (input_name in var_to_reduce_var)
                    reduce_var = var_to_reduce_var[input_name]
                    param_name = self._reduced_grads_to_param[reduce_var]
                    if self._param2device[
                            param_name] != self.role_maker.worker_index():
                        program_deps.crop_input_var_from_op(idx, input_name)
                    else:
                        reversed_input_vars.append(input_name)
                op.desc.set_input("X", reversed_input_vars)
                assert (len(op.desc.output_arg_names()) == 1)
                sum_res = op.desc.output_arg_names()[0]
                block._insert_op(
                    idx + 1,
                    type='c_sync_comm_stream',
                    inputs={'X': sum_res},
                    outputs={'Out': sum_res},
                    attrs={'ring_id': 0,
                           OP_ROLE_KEY: OpRole.Optimize})
                block._insert_op(
                    idx + 1,
                    type='c_allreduce_sum',
                    inputs={'X': sum_res},
                    outputs={'Out': sum_res},
                    attrs={'ring_id': 0,
                           OP_ROLE_KEY: OpRole.Optimize})
                block._insert_op(
                    idx + 1,
                    type='c_sync_calc_stream',
                    inputs={'X': sum_res},
                    outputs={'Out': sum_res},
                    attrs={OP_ROLE_KEY: OpRole.Optimize})
            elif op.type == "conditional_block":
                assert (op.desc.has_attr("sub_block"))
                subblock_idx = op.desc.attr("sub_block").id
                subblock_deps = program_deps.get_sub_block_deps(subblock_idx)
                # only prune amp subblock
                if subblock_deps is None or not self._is_amp_subblock(op):
                    continue
                # init
                reversed_output_vars = []
                for output_name in op.desc.output("Out"):
                    if output_name in program_deps._should_removed_var:
                        subblock_deps._should_removed_var.add(output_name)
                        program_deps.crop_output_var_from_op(idx, output_name)
                    else:
                        reversed_output_vars.append(output_name)
                # prune
                for sub_op_idx, _ in reversed(
                        list(enumerate(subblock_deps._block.ops))):
                    if subblock_deps.should_remove_op(sub_op_idx):
                        subblock_deps.remove_op(sub_op_idx)
                reversed_input_vars = []
                for input_name in op.desc.input('Input'):
                    if input_name not in subblock_deps._should_removed_var:
                        reversed_input_vars.append(input_name)
                    else:
                        program_deps.crop_input_var_from_op(idx, input_name)
                op.desc.set_input('Input', reversed_input_vars)
                op.desc.set_output('Out', reversed_output_vars)
            else:
                if program_deps.should_remove_op(idx):
                    program_deps.remove_op(idx)

        block._sync_with_cpp()
        return

    def _add_broadcast_allreduce(self, block, sub_prog, offset):
        """
        add broadcast and allreduce
        """
        # insert reduce ops
        inserted_op_num = 0
        ring_id = -1

        if len(sub_prog._allreduce_vars) > 0:
            for i in range(self._nrings):
                block._insert_op(
                    offset + sub_prog._end_idx,
                    type='c_sync_comm_stream',
                    inputs={'X': sub_prog._allreduce_vars},
                    outputs={'Out': sub_prog._allreduce_vars},
                    attrs={'ring_id': i,
                           OP_ROLE_KEY: OpRole.Forward})
            inserted_op_num += self._nrings

            for var in sub_prog._allreduce_vars:
                ring_id = (ring_id + 1) % self._nrings
                block._insert_op(
                    offset + sub_prog._end_idx,
                    type='c_allreduce_sum',
                    inputs={'X': var},
                    outputs={'Out': var},
                    attrs={'ring_id': ring_id,
                           OP_ROLE_KEY: OpRole.Backward})
                inserted_op_num += 1

            block._insert_op(
                offset + sub_prog._end_idx,
                type='c_sync_calc_stream',
                inputs={'X': sub_prog._allreduce_vars[-1]},
                outputs={'Out': sub_prog._allreduce_vars[-1]},
                attrs={OP_ROLE_KEY: OpRole.Forward})
            inserted_op_num += 1

        block._sync_with_cpp()
        # insert broadcast ops
        for op_idx in reversed(
                range(offset + sub_prog._start_idx, offset +
                      sub_prog._end_idx)):
            op = block.ops[op_idx]
            for input_name in op.desc.input_arg_names():
                if input_name in sub_prog._param2broadcast and \
                    input_name != sub_prog._param2broadcast[input_name]:
                    op._rename_input(input_name,
                                     sub_prog._param2broadcast[input_name])

        for param_name, broadcast_name in sub_prog._param2broadcast.items():
            if param_name != broadcast_name:
                block.create_var(
                    name=broadcast_name,
                    shape=self._varname2param[param_name].shape,
                    dtype=self._varname2param[param_name].dtype,
                    persistable=False)

        comm_dep_vars = [v for k, v in sub_prog._param2broadcast.items()]
        for i in range(self._nrings):
            block._insert_op(
                offset + sub_prog._start_idx,
                type='c_sync_comm_stream',
                inputs={'X': comm_dep_vars},
                outputs={'Out': comm_dep_vars},
                attrs={'ring_id': i,
                       OP_ROLE_KEY: OpRole.Forward})
        inserted_op_num += self._nrings

        for param_name, broadcast_name in sub_prog._param2broadcast.items():
            broadcast_var = block.var(broadcast_name)
            root_device = self._param2device[param_name]
            ring_id = (ring_id + 1) % self._nrings
            block._insert_op(
                offset + sub_prog._start_idx,
                type='c_broadcast',
                inputs={'X': broadcast_var.name},
                outputs={'Out': broadcast_var.name},
                attrs={
                    'ring_id': ring_id,
                    'root': root_device,
                    OP_ROLE_KEY: OpRole.Forward
                })
            inserted_op_num += 1

        comm_dep_vars = [
            v for k, v in sub_prog._param2broadcast.items() if k != v
        ]
        if comm_dep_vars != []:
            block._insert_op(
                offset + sub_prog._start_idx,
                type='c_sync_calc_stream',
                inputs={'X': comm_dep_vars[-1]},
                outputs={'Out': comm_dep_vars[-1]},
                attrs={OP_ROLE_KEY: OpRole.Forward})
            inserted_op_num += 1

        for param_name, broadcast_name in sub_prog._param2broadcast.items():
            if param_name == broadcast_name:
                continue
            broadcast_var = block.var(broadcast_name)
            block._insert_op(
                offset + sub_prog._start_idx,
                type="fill_constant",
                outputs={"Out": broadcast_var.name},
                attrs={
                    "shape": broadcast_var.shape,
                    "dtype": broadcast_var.dtype,
                    "value": 0.0,
                })
            inserted_op_num += 1
        block._sync_with_cpp()
        return inserted_op_num

    def _prune_startup_program(self, block):
        for idx, op in reversed(list(enumerate(block.ops))):
            for output_name in op.desc.output_arg_names():
                var_device_id = self._var_device_id(output_name)
                if var_device_id == -1 or var_device_id == self.role_maker.worker_index(
                ):
                    continue
                print("%d: startup_block remove op %s" %
                      (self.role_maker.worker_index(), op.type))
                block._remove_op(idx)
                break
        for var_name, _ in block.vars.items():
            var_device_id = self._var_device_id(var_name)
            if var_device_id == -1 or var_device_id == self.role_maker.worker_index(
            ):
                continue
            print("%d: startup_block remove var %s" %
                  (self.role_maker.worker_index(), var_name))
            block._remove_var(var_name)
        block._sync_with_cpp()

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        self._nrings = 3

        ckpts = list(self.user_defined_strategy.recompute_configs[
            "checkpoints"])
        optimizer = self.inner_opt
        # if len(ckpts) > 0:
        #     print("add recompute")
        #     print(ckpts)
        #     optimizer = fluid.optimizer.RecomputeOptimizer(optimizer)
        #     optimizer._set_checkpoints(ckpts)

        optimizer = fluid.contrib.mixed_precision.decorate(
            optimizer, use_dynamic_loss_scaling=True)
        print("doing zero optimize...")
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

        # split params
        self._varname2param = {x[0].name: x[0] for x in params_grads}
        self._varname2grad = {x[1].name: x[1] for x in params_grads}
        self._split_params(params_grads)
        print(self._device2params)

        # step2: add broadcast and reduce ops
        print("insert broadcast and allreduce")
        self._split_program(main_block)
        inserted_op_num = 0
        for idx, subprog in enumerate(self._sub_progs):
            print("subprog_{}: ({}-{})".format(idx, subprog._start_idx,
                                               subprog._end_idx))
            inserted_op_num += self._add_broadcast_allreduce(
                main_block, subprog, inserted_op_num)

        main_block._sync_with_cpp()
        startup_block._sync_with_cpp()

        # step3: insert reduce_sum for grad
        self._insert_scale_loss_grad_ops(
            main_block, scale=1.0 / self.role_maker.worker_num())
        main_block._sync_with_cpp()

        # step4: remove unneeded ops and vars from block
        print("main_block remove ops and vars")
        self._prune_main_program(main_block)
        print("startup_block remove ops and vars")
        self._prune_startup_program(startup_block)

        # check op dependecy for broadcast
        self._check_broadcast(main_block)
        return optimize_ops, params_grads

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
                    # print("after_broadcast, op_type: ", op.type, ": ",
                    #         "var_name: ", input_name, 
                    #         broadcast_vars[input_name]["broadcast_pos"],
                    #         last_sync_comm_op_idx,
                    #         idx)
                    assert (broadcast_vars[input_name]["broadcast_pos"] != -1)
                    assert (broadcast_vars[input_name]["broadcast_pos"] <
                            last_sync_comm_op_idx)
                    assert (last_sync_comm_op_idx < idx)
        print("check done")
        return

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

    def _insert_broadcast_ops(self, block, fuse_broadcast=False):
        def _insert_cache(cache,
                          prepend_comm_sync=False,
                          append_comm_sync=False):
            insert_idx = cache["insert_idx"]
            dummy_var_name = cache["dummy_var_name"]
            assert (len(cache["broadcast_ops"]) > 0)

            if prepend_comm_sync:
                insert_idx += self._insert_comm_sync(block, insert_idx,
                                                     [dummy_var_name])

            if len(cache["fill_constant_ops"]) > 0:
                insert_idx += self._insert_fill_constant(
                    block, insert_idx, cache["fill_constant_ops"],
                    [dummy_var_name])

            insert_idx += self._insert_broadcast_inner(block, insert_idx,
                                                       cache["broadcast_ops"])

            if append_comm_sync:
                insert_idx += self._insert_comm_sync(block, insert_idx,
                                                     [dummy_var_name])

            return insert_idx - cache["insert_idx"]

        print("insert_idx: ", [x["insert_idx"] for x in self._sub_progs])
        move_ahead = 1
        for idx, cache in reversed(list(enumerate(self._sub_progs))):
            if idx < move_ahead:
                cache["insert_idx"] = 0
            else:
                cache["insert_idx"] = self._sub_progs[idx - move_ahead][
                    "insert_idx"]
        print("insert_idx: ", [x["insert_idx"] for x in self._sub_progs])

        inserted_op_num = 0
        for idx, cache in enumerate(self._sub_progs):
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
                    offset += 1
                    for ring_id in range(self._nrings):
                        block._insert_op(
                            offset,
                            type='c_sync_comm_stream',
                            inputs={'X': grad},
                            outputs={'Out': grad},
                            attrs={
                                'ring_id': ring_id,
                                OP_ROLE_KEY: OpRole.Backward
                            })
                        offset += 1

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

    def _insert_comm_sync(self, block, insert_idx, var_names):
        for r in range(self._nrings):
            block._insert_op(
                insert_idx,
                type='c_sync_comm_stream',
                inputs={'X': var_names},
                outputs={'Out': var_names},
                attrs={'ring_id': r,
                       OP_ROLE_KEY: OpRole.Backward})
            insert_idx += 1
        return self._nrings

    def _insert_broadcast_inner(self, block, insert_idx, broadcast_attrs):
        for attr in broadcast_attrs:
            block._insert_op(insert_idx, **attr)
            insert_idx += 1
        return len(broadcast_attrs)

    def _insert_fill_constant(self, block, insert_idx, fill_constant_attrs,
                              var_names):
        for attr in fill_constant_attrs:
            block._insert_op(insert_idx, **attr)
            insert_idx += 1
        block._insert_op(
            insert_idx,
            type='c_sync_calc_stream',
            inputs={'X': var_names},
            outputs={'Out': var_names},
            attrs={OP_ROLE_KEY: OpRole.Backward})
        return len(fill_constant_attrs) + 1
