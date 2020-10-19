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
from paddle.fluid.contrib.mixed_precision.decorator import decorate as amp_decorate

import paddle.fluid as fluid

from functools import reduce
import math
import re

__all__ = ["ShardingOptimizer"]


def _pretty_op_desc_(op_desc, prefix):
    out_s = "%s\tname:[%s]\n%s    \tinputs:[%s]\n%s    \toutputs:[%s]" % \
            (prefix + "_op", str(op_desc.type()), prefix + "_input", " ".join(op_desc.input_arg_names()),
             prefix + "_output", " ".join(op_desc.output_arg_names()))
    return out_s


class SubProgram(object):
    def __init__(self, block):
        self._block = block
        self._allreduce_vars = []
        # sub program start idx
        self._start_idx = -1
        # sub program end idx
        self._end_idx = -1
        # param name to broadcast name
        self._param2broadcast = {}
        self._broadcast_vars = []
        # cast op pairs, fp16 name (str) -> fp32 name (str)
        self._cast_ops = {}
        # fill constant vars
        self._fill_constant_vars = []
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
        self._var_to_use_op = {}
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
        if var_name in self._var_to_use_op:
            return self._var_to_use_op[var_name]
        else:
            return None

    def _build_deps(self, ):
        for var_name in self._start_vars:
            self._var_to_use_op[var_name] = []
            self._var_to_generate_op[var_name] = []

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
                if input_name in self._var_to_use_op:
                    deps_reduce = True
            if not deps_reduce:
                continue
            for input_name in input_vars:
                if input_name in self._var_to_use_op:
                    self._var_to_use_op[input_name].append(idx)
            for output_name in output_vars:
                if output_name not in self._var_to_use_op:
                    self._var_to_use_op[output_name] = []
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
        if var_name in self._var_to_use_op:
            # update var -> dep_var_op
            if self._var_to_use_op[var_name] != []:
                if op_idx not in self._var_to_use_op[var_name]:
                    raise ValueError(
                        "op_idx: {} is not in self._var_to_use_op[{}], "
                        "self._var_to_use_op[{}] is {}".format(
                            op_idx, var_name, var_name, self._var_to_use_op[
                                var_name]))
                self._var_to_use_op[var_name].remove(op_idx)
            # update _should_removed_var
            if var_name in self._start_vars:
                self._should_removed_var.discard(var_name)
            elif self._var_to_use_op[
                    var_name] == []:  # no more deps of this var
                self._should_removed_var.add(var_name)
            elif self._var_to_generate_op[var_name][-1] >= self._var_to_use_op[
                    var_name][-1]:
                # there are circle in the graph
                self._should_removed_var.add(var_name)
            else:  # input_name should not be deleted
                self._should_removed_var.discard(var_name)

    def crop_output_var_from_op(self, op_idx, var_name):
        if var_name in self._var_to_generate_op:
            assert (op_idx in self._var_to_generate_op[var_name])
            self._var_to_generate_op[var_name].remove(op_idx)
        if self._block.has_var(var_name):
            if var_name not in self._var_to_generate_op or self._var_to_generate_op[
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


class ShardingOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(ShardingOptimizer, self).__init__(optimizer)
        self.inner_opt = None
        self.meta_optimizers_white_list = [
            "RecomputeOptimizer",
            "AMPOptimizer",
        ]
        self._main_program = None
        self._startup_program = None
        # we do not allow meta optimizer to be inner optimizer currently
        # params and fp16 params is for broadcast
        self._params = set([])
        self._fp16_params = set([])
        # fp16 to fp32
        self._fp16_to_params = {}
        self._broadcast_vars = set([])
        # _param(str) -> device_id(int) 
        self._param2device = {}
        # varname(str) -> param(Variable)
        # reduced grads to param name
        self._reduced_grads_to_param = {}
        # self._nrings(int) is for nccl communicate
        self._nrings = 3
        # self._sub_progs
        self._sub_progs = []
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
        self._collective_helper = None

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
        return self.user_defined_strategy.sharding

    def _disable_strategy(self, dist_strategy):
        dist_strategy.sharding = False

    def _is_fp16_cast_op(self, block, op):
        if op.type != "cast":
            return False
        if is_optimizer_op(op):
            return False
        assert (len(op.desc.input_arg_names()) == 1)
        assert (len(op.desc.output_arg_names()) == 1)
        input_name, output_name = op.desc.input_arg_names()[
            0], op.desc.output_arg_names()[0]
        if input_name not in self._params:
            return False
        input_var = block.var(input_name)
        output_var = block.var(output_name)
        if input_var.dtype != core.VarDesc.VarType.FP32 or \
            output_var.dtype != core.VarDesc.VarType.FP16:
            return False
        return True

    def _is_fp32_cast_op(self, block, op):
        if op.type != "cast":
            return False
        if not is_optimizer_op(op):
            return False
        assert (len(op.desc.input_arg_names()) == 1)
        assert (len(op.desc.output_arg_names()) == 1)
        input_name, output_name = op.desc.input_arg_names()[
            0], op.desc.output_arg_names()[0]
        input_var = block.var(input_name)
        output_var = block.var(output_name)
        if input_var.dtype != core.VarDesc.VarType.FP16 or \
            output_var.dtype != core.VarDesc.VarType.FP32:
            return False
        return True

    def _split_params(self, params):
        param2device = {}
        total_param_mem = 0.0
        param2mem = []
        for param in params:
            mem = self._get_var_size(param)
            total_param_mem += mem
            param2mem.append((param.name, mem))
            # print(param.name, mem)
        # print("total_param_mem: ", total_param_mem)
        device_num = self.role_maker._worker_num()
        # print("device_num: ", device_num)
        device2params = {x: [] for x in range(device_num)}
        device_idx = 0
        mem_accu = 0.0
        for param_name, mem in param2mem:
            if mem_accu > total_param_mem * 1.0 * (device_idx + 1) / device_num:
                device_idx += 1
            device2params[device_idx].append(param_name)
            param2device[param_name] = device_idx
            mem_accu += mem
        # for debug
        print(device2params)
        return param2device

    def _is_opti_var(self, var_name):
        if var_name in self._params:
            return True
        for suffix in [
                "_moment1_0", "_moment2_0", "_beta1_pow_acc_0",
                "_beta2_pow_acc_0", "_velocity_0"
        ]:
            base_name = re.sub(suffix, '', var_name)
            if base_name in self._params:
                return True
        return False

    def _var_device_id(self, var_name):
        if not self._is_opti_var(var_name):
            return -1
        if var_name in self._param2device:
            return self._param2device[var_name]
        for suffix in [
                "_moment1_0", "_moment2_0", "_beta1_pow_acc_0",
                "_beta2_pow_acc_0", "_velocity_0"
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
                if input_name not in self._broadcast_vars:
                    continue
                root_device = self._param2device[input_name]
                if input_name in sub_prog._param2broadcast:
                    # skip broadcast because it reuse the old broadcast var
                    broadcast_name = sub_prog._param2broadcast[input_name]
                    if input_name != broadcast_name:
                        op._rename_input(input_name, broadcast_name)
                    continue
                if root_device == self.role_maker._worker_index():
                    broadcast_var_name = input_name
                else:
                    broadcast_var_name = unique_name.generate(input_name +
                                                              "@BroadCast")
                    sub_prog._fill_constant_vars.append(broadcast_var_name)
                sub_prog._param2broadcast[input_name] = broadcast_var_name
                sub_prog._broadcast_vars.append(
                    (broadcast_var_name, self._param2device[input_name]))
                sub_prog._param_mem += self._get_var_size(
                    self._main_program.global_block().var(input_name))

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

            # find cast op
            if self._is_fp16_cast_op(block, op):
                fp32_param = op.desc.input_arg_names()[0]
                fp16_param = op.desc.output_arg_names()[0]
                if self._param2device[
                        fp32_param] == self.role_maker._worker_index():
                    sub_prog._cast_ops[fp16_param] = fp32_param

        if sub_prog._param_mem > 0:
            sub_prog._start_idx = 0
            self._sub_progs.insert(0, sub_prog)
        return

    def is_gradient_clip_op(self, op):
        return op.desc.has_attr("op_namescope") \
            and op.desc.attr("op_namescope").startswith("/gradient_clip")

    # def _is_amp_sum_op(self, op):
    #     return op.type == "sum" and op.desc.has_attr("op_namescope") \
    #         and op.desc.attr("op_namescope").startswith("/mixed_precision")

    # def _is_amp_subblock(self, op):
    #     return op.type == "conditional_block" and op.desc.has_attr("op_namescope") \
    #         and op.desc.attr("op_namescope").startswith("/mixed_precision")

    def _is_weight_decay_op(self, op):
        return op.desc.has_attr("op_namescope") \
            and op.desc.attr("op_namescope").startswith("/regularization")

    def _prune_weight_decay(self, block):
        for idx, op in reversed(list(enumerate(block.ops))):
            if not self._is_weight_decay_op(op):
                continue
            if OP_ROLE_VAR_KEY not in op.attr_names:
                raise ValueError(
                    "The Weight Dacay op should hold op_role_var attribute"
                    "but the {} op does not hold op_role_var".format(op.type))
            op_role_var = op.all_attrs()[OP_ROLE_VAR_KEY]
            if self._param2device[op_role_var[
                    0]] != self.role_maker._worker_index():
                block._remove_op(idx)
        block._sync_with_cpp()

    def _prune_fp16(self, block):
        # remove cast
        for idx, op in reversed(list(enumerate(block.ops))):
            if not self._is_fp32_cast_op(block, op):
                continue
            output_name = op.desc.output_arg_names()[0]
            param_name = output_name.strip("@GRAD")
            if param_name not in self._param2device:
                raise ValueError("Input 'X' of check_finite_and_unscale must"
                                 "be grads, but {} is not a grad".format(
                                     input_name))
            if output_name in self._reduced_grads_to_param:
                continue
            if self._param2device[param_name] == self.role_maker._worker_index(
            ):
                continue
            block._remove_op(idx)
            block._remove_var(output_name)

        block._sync_with_cpp()
        update_loss_scaling_op_idx = -1
        inf_var_name = ''
        for idx, op in reversed(list(enumerate(block.ops))):
            if op.type == "update_loss_scaling":
                update_loss_scaling_op_idx = idx
                inf_var_name = op.desc.input('FoundInfinite')[0]
                op._rename_input(inf_var_name, inf_var_name + "@sharding")
            if op.type in ["check_finite_and_unscale", "update_loss_scaling"]:
                reversed_x = []
                for input_name in op.desc.input('X'):
                    param_name = input_name.strip("@GRAD")
                    if param_name not in self._param2device:
                        raise ValueError(
                            "Input 'X' of check_finite_and_unscale must"
                            "be grads, but {} is not a grad".format(input_name))
                    if self._param2device[
                            param_name] == self.role_maker._worker_index():
                        reversed_x.append(input_name)
                op.desc.set_input('X', reversed_x)
                op.desc.set_output('Out', reversed_x)
        if update_loss_scaling_op_idx == -1:
            return
        inf_var = block.var(inf_var_name)
        inf_var_fp32 = block.create_var(
            name=inf_var_name + "@cast_int32",
            shape=inf_var.shape,
            dtype=core.VarDesc.VarType.INT32)
        inf_var_sharding = block.create_var(
            name=inf_var_name + "@sharding",
            shape=inf_var.shape,
            dtype=inf_var.dtype)
        block._insert_op(
            update_loss_scaling_op_idx,
            type='cast',
            inputs={'X': inf_var},
            outputs={'Out': inf_var_fp32},
            attrs={
                "in_dtype": inf_var.dtype,
                "out_dtype": inf_var_fp32.dtype,
                OP_ROLE_KEY: OpRole.Optimize
            })
        self._insert_sync_calc_op(block, update_loss_scaling_op_idx + 1,
                                  [inf_var_fp32])
        block._insert_op(
            update_loss_scaling_op_idx + 2,
            type='c_allreduce_max',
            inputs={'X': inf_var_fp32},
            outputs={'Out': inf_var_fp32},
            attrs={'ring_id': 0,
                   OP_ROLE_KEY: OpRole.Optimize})
        comm_op_num = self._insert_sync_comm_ops(
            block, update_loss_scaling_op_idx + 3, [inf_var_fp32])
        block._insert_op(
            update_loss_scaling_op_idx + 3 + comm_op_num,
            type='cast',
            inputs={'X': inf_var_fp32},
            outputs={'Out': inf_var_sharding},
            attrs={
                "in_dtype": inf_var_fp32.dtype,
                "out_dtype": inf_var_sharding.dtype,
                OP_ROLE_KEY: OpRole.Optimize
            })
        block._sync_with_cpp()

    def _prune_gradient_clip(self, block):
        deperated_vars = set()
        deperate_op_idx = set()
        for idx, op in enumerate(block.ops):
            if not self.is_gradient_clip_op(op):
                continue
            if op.type == "sum":
                continue
            deperate_op = False
            for input_name in op.desc.input_arg_names():
                if input_name in deperated_vars:
                    deperate_op = True
                param_name = input_name.strip("@GRAD")
                if param_name in self._param2device:
                    if self._param2device[
                            param_name] != self.role_maker._worker_index():
                        deperate_op = True

            if deperate_op:
                deperate_op_idx.add(idx)
                for output_name in op.desc.output_arg_names():
                    deperated_vars.add(output_name)

        if not deperated_vars:
            # got no gradient_clip op
            return

        for idx, op in reversed(list(enumerate(block.ops))):
            if not self.is_gradient_clip_op(op):
                continue
            if idx in deperate_op_idx:
                block._remove_op(idx)
                continue
            reversed_inputs = []
            if op.type == "sum":
                for input_name in op.desc.input_arg_names():
                    if input_name not in deperated_vars:
                        reversed_inputs.append(input_name)
                op.desc.set_input("X", reversed_inputs)
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

        for var_name in deperated_vars:
            block._remove_var(var_name)
        block._sync_with_cpp()
        return

    def _prune_main_program(self, block):
        """
        calculate deps from allredce op to optimize op,
        remove ops and vars not needed in this worker
        """
        self._prune_weight_decay(block)
        self._prune_fp16(block)
        self._prune_gradient_clip(block)
        # if self.role_maker._worker_index() == 1:
        #     with open("debug_program", 'w') as f:
        #         f.write(str(block.program))

        # build prog deps
        reduced_grads = []
        for idx, op in enumerate(block.ops):
            input_names = op.desc.input_arg_names()
            output_names = op.desc.output_arg_names()
            if op.type == "c_allreduce_sum":
                assert (len(output_names) == 1)
                output_name = output_names[0]
                reduced_grads.append(output_name)

        params = []
        for var_name in list(block.vars.keys()):
            if self._is_opti_var(var_name) and \
                self._var_device_id(var_name) != self.role_maker._worker_index():
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

    def _remove_cast_op(self, block, sub_prog, offset):
        inserted_op_num = 0
        for op_idx in reversed(
                range(offset + sub_prog._start_idx, offset +
                      sub_prog._end_idx)):
            op = block.ops[op_idx]
            if self._is_fp16_cast_op(block, op):
                block._remove_op(op_idx)
                inserted_op_num -= 1
        block._sync_with_cpp()
        return inserted_op_num

    def _insert_broadcast_ops(self, block, insert_idx, broadcast2root):
        """
        _add_broadcast_ops
        """
        ring_id = -1
        # TODO(mapingshuo): correct OP_ROLE_KEY
        for broadcast_name, root_device in broadcast2root:
            ring_id = (ring_id + 1) % self._nrings
            block._insert_op(
                insert_idx,
                type='c_broadcast',
                inputs={'X': broadcast_name},
                outputs={'Out': broadcast_name},
                attrs={
                    'ring_id': ring_id,
                    'root': root_device,
                    OP_ROLE_KEY: OpRole.Forward
                })
        return

    def _insert_allreduce_ops(self, block, insert_idx, allreduce_vars):
        """
        _add_allreduce_ops
        """
        ring_id = -1
        for var in allreduce_vars:
            ring_id = (ring_id + 1) % self._nrings
            block._insert_op(
                insert_idx,
                type='c_allreduce_sum',
                inputs={'X': var},
                outputs={'Out': var},
                attrs={'ring_id': ring_id,
                       OP_ROLE_KEY: OpRole.Backward})
        return

    def _insert_cast_ops(self, block, insert_idx, cast_ops):
        """
        _add_cast_ops
        """
        for fp16_name, fp32_name in cast_ops.items():
            block._insert_op(
                insert_idx,
                type="cast",
                inputs={"X": fp32_name},
                outputs={"Out": fp16_name},
                attrs={
                    "in_dtype": core.VarDesc.VarType.FP32,
                    "out_dtype": core.VarDesc.VarType.FP16
                })
        return

    def _insert_fill_constant_ops(self, block, insert_idx, fill_constant_vars):
        """
        _add_fill_constant_ops
        """
        for broadcast_name in fill_constant_vars:
            broadcast_var = block.var(broadcast_name)
            block._insert_op(
                insert_idx,
                type="fill_constant",
                outputs={"Out": broadcast_var.name},
                attrs={
                    "shape": broadcast_var.shape,
                    "dtype": broadcast_var.dtype,
                    "value": 0.0,
                })
        return

    def _insert_sync_comm_ops(self, block, insert_idx, comm_dep_vars):
        """
        _insert_sync_comm_ops
        """
        # TODO(mapingshuo) fix OP_ROLE_KEY
        for i in range(self._nrings):
            block._insert_op(
                insert_idx,
                type='c_sync_comm_stream',
                inputs={'X': comm_dep_vars},
                outputs={'Out': comm_dep_vars},
                attrs={'ring_id': i,
                       OP_ROLE_KEY: OpRole.Forward})
        return self._nrings

    def _insert_sync_calc_op(self, block, insert_idx, calc_dep_vars):
        """
        _insert_sync_calc_op
        """
        # TODO(mapingshuo) fix OP_ROLE_KEY
        block._insert_op(
            insert_idx,
            type='c_sync_calc_stream',
            inputs={'X': calc_dep_vars},
            outputs={'Out': calc_dep_vars},
            attrs={OP_ROLE_KEY: OpRole.Forward})
        return

    def _add_broadcast_allreduce(self, block):
        """
        _add_broadcast_allreduce
        """
        ring_id = -1

        if len(self._sub_progs) < 1:
            return

        if self._sub_progs[-1]._allreduce_vars:
            self._insert_sync_comm_ops(block, self._sub_progs[-1]._end_idx,
                                       self._sub_progs[-1]._allreduce_vars)
            self._insert_allreduce_ops(block, self._sub_progs[-1]._end_idx,
                                       self._sub_progs[-1]._allreduce_vars)

        for idx, subprog in reversed(list(enumerate(self._sub_progs))):
            print("subprog_{}: ({}-{})".format(idx, subprog._start_idx,
                                               subprog._end_idx))

            allreduce_vars = self._sub_progs[
                idx - 1]._allreduce_vars if idx > 0 else []
            broadcast_vars = self._sub_progs[idx +
                                             1]._broadcast_vars if idx < len(
                                                 self._sub_progs) - 1 else []
            fill_constant_vars = self._sub_progs[
                idx + 2]._fill_constant_vars if idx < len(
                    self._sub_progs) - 2 else []
            cast_ops = self._sub_progs[idx + 2]._cast_ops if idx < len(
                self._sub_progs) - 2 else {}

            for op_idx in reversed(range(subprog._start_idx, subprog._end_idx)):
                op = block.ops[op_idx]
                for input_name in op.desc.input_arg_names():
                    if input_name in subprog._param2broadcast and \
                        input_name != subprog._param2broadcast[input_name]:
                        op._rename_input(input_name,
                                         subprog._param2broadcast[input_name])

            for param_name, broadcast_name in subprog._param2broadcast.items():
                if param_name != broadcast_name:
                    block.create_var(
                        name=broadcast_name,
                        shape=self._main_program.global_block().var(
                            param_name).shape,
                        dtype=self._main_program.global_block().var(param_name)
                        .dtype,
                        persistable=False)

            # step1: remove cast ops
            block._sync_with_cpp()
            subprog._end_idx += self._remove_cast_op(block, subprog, 0)

            # step2: add Sync ops
            comm_dep_vars = allreduce_vars + [x[0] for x in broadcast_vars]
            if len(comm_dep_vars) > 0:
                self._insert_sync_comm_ops(
                    block,
                    subprog._end_idx,
                    comm_dep_vars, )
            calc_dep_vars = fill_constant_vars + [
                k for k, v in cast_ops.items()
            ] + self._sub_progs[idx]._allreduce_vars

            if len(calc_dep_vars) > 0:
                self._insert_sync_calc_op(block, subprog._end_idx,
                                          [calc_dep_vars[-1]])

            # step3: insert `fill_constant` ops 
            self._insert_fill_constant_ops(block, subprog._end_idx,
                                           fill_constant_vars)

            # step4: add `cast` ops     
            self._insert_cast_ops(block, subprog._end_idx, cast_ops)

            # step5: add broadcast ops
            self._insert_broadcast_ops(block, subprog._start_idx,
                                       broadcast_vars)

            # step6: add all_reduce ops
            self._insert_allreduce_ops(block, subprog._start_idx,
                                       allreduce_vars)

            block._sync_with_cpp()

        if self._sub_progs[0]._broadcast_vars:
            self._insert_sync_comm_ops(
                block, self._sub_progs[0]._start_idx,
                [x[0] for x in self._sub_progs[0]._broadcast_vars])
            self._insert_broadcast_ops(block, self._sub_progs[0]._start_idx,
                                       self._sub_progs[0]._broadcast_vars)

        fill_constant_vars = reduce(
            lambda x, y: x._fill_constant_vars + y._fill_constant_vars,
            self._sub_progs[:2])

        # Join
        cast_ops = {}
        for x in self._sub_progs[:2]:
            for k, v in x._cast_ops.items():
                cast_ops[k] = v

        calc_deps_vars = fill_constant_vars + [k for k, v in cast_ops.items()]
        if fill_constant_vars or cast_ops:
            self._insert_sync_calc_op(block, self._sub_progs[0]._start_idx,
                                      [calc_deps_vars[-1]])

        if fill_constant_vars:
            self._insert_fill_constant_ops(block, self._sub_progs[0]._start_idx,
                                           fill_constant_vars)

        if cast_ops:
            self._insert_cast_ops(block, self._sub_progs[0]._start_idx,
                                  cast_ops)

        return

    def _prune_startup_program(self, block):
        for idx, op in reversed(list(enumerate(block.ops))):
            for output_name in op.desc.output_arg_names():
                var_device_id = self._var_device_id(output_name)
                if var_device_id == -1 or var_device_id == self.role_maker._worker_index(
                ):
                    continue
                print("%d: startup_block remove op %s" %
                      (self.role_maker._worker_index(), op.type))
                block._remove_op(idx)
                break
        for var_name in list(block.vars.keys()):
            var_device_id = self._var_device_id(var_name)
            if var_device_id == -1 or var_device_id == self.role_maker._worker_index(
            ):
                continue
            print("%d: startup_block remove var %s" %
                  (self.role_maker._worker_index(), var_name))
            block._remove_var(var_name)
        block._sync_with_cpp()

    def _find_broadcast_params(self, params, param2device):
        broadcast_vars = set([])
        fp16_params = set([])
        fp16_to_fp32 = {}
        main_block = self._main_program.global_block()

        param_usage = {x: 0 for x in params}
        for op in main_block.ops:
            if is_optimizer_op(op):
                continue
            for input_name in op.desc.input_arg_names():
                if input_name in params:
                    param_usage[input_name] += 1

        for op in main_block.ops:
            if not self._is_fp16_cast_op(main_block, op):
                continue
            input_name = op.input_arg_names[0]
            output_name = op.output_arg_names[0]
            broadcast_vars.add(output_name)
            fp16_params.add(output_name)
            fp16_to_fp32[output_name] = input_name
            param_usage[input_name] -= 1
            param2device[output_name] = param2device[input_name]

        for param, usage in param_usage.items():
            if usage > 0:
                broadcast_vars.add(param)
        return fp16_params, broadcast_vars, fp16_to_fp32

    def _set_up(self, params_grads):
        # step 1: initialize nccl
        print("work idx: ", self.role_maker._worker_index())
        endpoints = self.role_maker._get_trainer_endpoints()
        current_endpoint = endpoints[self.role_maker._worker_index()]
        self._collective_helper = CollectiveHelper(self.role_maker,
                                                   self._nrings)
        for ring_id in range(self._nrings):
            self._collective_helper._init_communicator(
                self._startup_program, current_endpoint, endpoints,
                self.role_maker._worker_index(), ring_id, None)
        startup_block = self._startup_program.global_block()
        startup_block._sync_with_cpp()

        # step 2: split params
        self._params = set([x[0].name for x in params_grads])
        self._param2device = self._split_params([x[0] for x in params_grads])

        # step 3: get broadcast vars
        self._fp16_params, self._broadcast_vars, self._fp16_to_params = self._find_broadcast_params(
            self._params, self._param2device)

    def _wait(self, ):
        endpoints = self.role_maker._get_trainer_endpoints()
        current_endpoint = endpoints[self.role_maker._worker_index()]
        if self.role_maker._worker_index() == 0:
            self._collective_helper._wait(current_endpoint, endpoints, '6174')

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):

        # if self.user_defined_strategy.sharding_configs["allreduce"]:
        #     return self.minimize_impl_allreduce(loss, startup_program,
        #                                         parameter_list, no_grad_set)

        # ckpts = list(self.user_defined_strategy.sharding_configs["checkpoints"])

        # optimizer = self.inner_opt
        # if len(ckpts) > 0:
        #     print("add recompute")
        #     print(ckpts)
        #     optimizer = fluid.optimizer.RecomputeOptimizer(optimizer)
        #     optimizer._set_checkpoints(ckpts)

        # if self.user_defined_strategy.sharding_configs["amp"]:
        #     optimizer = amp_decorate(optimizer, use_dynamic_loss_scaling=True)

        self._nrings = self.user_defined_strategy.sharding_configs["nrings"]
        self._fuse_broadcast_MB_bytes = self.user_defined_strategy.sharding_configs[
            "fuse_broadcast_MB_bytes"]

        print("doing sharding inner_opt optimize...")
        if self.inner_opt is None:
            raise ValueError(
                "self.inner_opt of ShardingOptimizer should not be None.")
        optimize_ops, params_grads = self.inner_opt.minimize(
            loss, startup_program, parameter_list, no_grad_set)

        print("doing sharding optimize...")
        if startup_program is None:
            startup_program = default_startup_program()
        main_block = loss.block
        startup_block = startup_program.global_block()
        self._main_program = main_block.program
        self._startup_program = startup_program

        # step1: set_up
        self._set_up(params_grads)

        # step2: split_program
        self._split_program(main_block)

        # step3: add broadcast and reduce ops
        print("insert broadcast and allreduce")
        self._add_broadcast_allreduce(main_block)
        main_block._sync_with_cpp()
        startup_block._sync_with_cpp()

        # step4: insert reduce_sum for grad
        self._insert_scale_loss_grad_ops(
            main_block, scale=1.0 / self.role_maker._worker_num())
        main_block._sync_with_cpp()

        # step5: remove unneeded ops and vars from block
        print("main_block remove ops and vars")
        self._prune_main_program(main_block)
        print("startup_block remove ops and vars")
        self._prune_startup_program(startup_block)

        # check op dependecy for broadcast
        self._check_broadcast(main_block)
        self._check_allreduce_sum(main_block)
        self._wait()
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
                        assert (last_sync_calc_op_idx != -1)
                        assert (broadcast_vars[var_name]["fill_constant_pos"] <
                                last_sync_calc_op_idx)
                        assert (last_sync_calc_op_idx < idx)
                    continue
            for input_name in op.desc.input_arg_names():
                if input_name in broadcast_vars:
                    assert (broadcast_vars[input_name]["broadcast_pos"] != -1)
                    assert (broadcast_vars[input_name]["broadcast_pos"] <
                            last_sync_comm_op_idx)
                    assert (last_sync_comm_op_idx < idx)
        print("check broadcast done")
        return

    def _check_allreduce_sum(self, block):
        """
        if a Var is allreduced, the op order should be:
            - 0: op that generate Var
            - 1: sync_calc
            - 2: allreduce_sum op
            - 3: sync_comm
            - 4: op that use Var
        """
        var_status = {}
        for op in block.ops:
            if op.type == "c_allreduce_sum":
                var_name = op.desc.input_arg_names()[0]
                var_status[var_name] = -1

        for op in block.ops:
            if op.type == "c_sync_calc_stream":
                for var_name in var_status:
                    if var_name in var_status and var_status[var_name] == 0:
                        var_status[var_name] = 1
            elif op.type == "c_allreduce_sum":
                var_name = op.desc.input_arg_names()[0]
                if var_status[var_name] == -1:
                    raise ValueError("{} is not generated, but you are"
                                     "trying to all-reduce it".format(var_name))
                if var_status[var_name] == 0:
                    raise ValueError("There should be a sync_calc op "
                                     "after generate Var: {} and before the"
                                     "c_allreduce_sum op".format(var_name))
                assert (var_status[var_name] == 1)
                var_status[var_name] = 2
            elif op.type == "c_sync_comm_stream":
                for var_name in op.desc.input_arg_names():
                    if var_name in var_status and var_status[var_name] == 2:
                        var_status[var_name] = 3
            else:
                for input_name in op.desc.input_arg_names():
                    if input_name in var_status:
                        if var_status[input_name] != 3:
                            raise ValueError(
                                "There should be a sync_comm op "
                                "after allreduce the Var: {}".format(var_name))
                for output_name in op.desc.output_arg_names():
                    if output_name in var_status and \
                      var_status[output_name] == -1:
                        var_status[output_name] = 0
        print("finish check allreduce")

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

    # def _insert_allreduce_ops_tmp(self, block):
    #     ring_id = -1
    #     grad = None
    #     for idx, op in reversed(list(enumerate(block.ops))):
    #         if is_backward_op(op) and \
    #                 OP_ROLE_VAR_KEY in op.attr_names:
    #             op_role_var = op.all_attrs()[OP_ROLE_VAR_KEY]

    #             if len(op_role_var) == 0:
    #                 continue
    #             assert len(op_role_var) % 2 == 0

    #             offset = idx
    #             for i in range(0, len(op_role_var), 2):
    #                 # param = block.vars[op_role_var[i]]
    #                 grad = block.vars[op_role_var[i + 1]]
    #                 # TODO(mapingshuo): what is is_distributed
    #                 # if param.is_distributed:
    #                 #     continue

    #                 if offset == idx:
    #                     offset += 1
    #                     block._insert_op(
    #                         offset,
    #                         type='c_sync_calc_stream',
    #                         inputs={'X': grad},
    #                         outputs={'Out': grad},
    #                         attrs={OP_ROLE_KEY: OpRole.Backward})
    #                     offset += 1
    #                 # As we search ops reversedly, we should insert c_allreduce_sum
    #                 # op in the same way to keep the ring_id alternate
    #                 print("add allreduce op for {}".format(grad.name))
    #                 ring_id = (ring_id + 1) % self._nrings
    #                 block._insert_op(
    #                     offset,
    #                     type='c_allreduce_sum',
    #                     inputs={'X': grad},
    #                     outputs={'Out': grad},
    #                     attrs={
    #                         'ring_id': ring_id,
    #                         OP_ROLE_KEY: OpRole.Backward
    #                     })

    #     if grad is None:
    #         return

    #     for idx, op in enumerate(block.ops):
    #         if is_optimizer_op(op):
    #             for ring_id in range(self._nrings):
    #                 block._insert_op(
    #                     idx + ring_id,
    #                     type='c_sync_comm_stream',
    #                     inputs={'X': grad},
    #                     outputs={'Out': grad},
    #                     attrs={
    #                         'ring_id': ring_id,
    #                         OP_ROLE_KEY: OpRole.Backward
    #                     })
    #             break

    # def minimize_impl_allreduce(self,
    #                             loss,
    #                             startup_program=None,
    #                             parameter_list=None,
    #                             no_grad_set=None):

    #     self._nrings = self.user_defined_strategy.sharding_configs["nrings"]

    #     ckpts = list(self.user_defined_strategy.zero_configs["checkpoints"])
    #     optimizer = self.inner_opt
    #     if len(ckpts) > 0:
    #         print("add recompute")
    #         print(ckpts)
    #         optimizer = fluid.optimizer.RecomputeOptimizer(optimizer)
    #         optimizer._set_checkpoints(ckpts)

    #     if self.user_defined_strategy.sharding_configs["amp"]:
    #         optimizer = amp_decorate(optimizer, use_dynamic_loss_scaling=True)

    #     optimize_ops, params_grads = optimizer.minimize(
    #         loss, startup_program, parameter_list, no_grad_set)

    #     if startup_program is None:
    #         startup_program = default_startup_program()

    #     print("work idx: ", self.role_maker._worker_index())
    #     endpoints = self.role_maker._get_trainer_endpoints()
    #     current_endpoint = endpoints[self.role_maker._worker_index()]

    #     collective_helper = CollectiveHelper(self.role_maker, self._nrings)
    #     for ring_id in range(self._nrings):
    #         collective_helper._init_communicator(
    #             startup_program, current_endpoint, endpoints,
    #             self.role_maker._worker_index(), ring_id, '6174')
    #     main_block = loss.block
    #     startup_block = startup_program.global_block()
    #     self._broadcast_params(startup_block)

    #     self._insert_scale_loss_grad_ops(
    #         main_block, scale=1.0 / self.role_maker._worker_num())
    #     self._insert_allreduce_ops_tmp(main_block)
    #     print("insert allreduce done")
    #     return optimize_ops, params_grads
