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

from paddle.fluid import core
from functools import reduce
from paddle.distributed.fleet.meta_optimizers.common import is_optimizer_op, is_loss_grad_op
from paddle.distributed.fleet.meta_optimizers.common import OpRole, OP_ROLE_KEY, OP_ROLE_VAR_KEY

import re


class ProgramSegment(object):
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


class FP16Utils(object):
    def __init__(self):
        pass

    def is_fp16_cast_op(block, op, params):
        if op.type != "cast":
            return False
        if is_optimizer_op(op):
            return False
        assert (len(op.desc.input_arg_names()) == 1)
        assert (len(op.desc.output_arg_names()) == 1)
        input_name, output_name = op.desc.input_arg_names()[
            0], op.desc.output_arg_names()[0]
        if input_name not in params:
            return False
        input_var = block.var(input_name)
        output_var = block.var(output_name)
        if input_var.dtype != core.VarDesc.VarType.FP32 or \
            output_var.dtype != core.VarDesc.VarType.FP16:
            return False
        return True

    def is_fp32_cast_op(block, op):
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

    def remove_cast_op(block, params, segment, offset):
        inserted_op_num = 0
        for op_idx in reversed(
                range(offset + segment._start_idx, offset + segment._end_idx)):
            op = block.ops[op_idx]
            if FP16Utils.is_fp16_cast_op(block, op, params):
                block._remove_op(op_idx)
                inserted_op_num -= 1
        block._sync_with_cpp()
        return inserted_op_num


class DeviceVariables(object):
    def __init__(self, ):
        self.params = set([])
        self.worker_idx = -1
        self.worker_num = -1
        self.param2device = {}

    def setup(self, params_grads, worker_idx, worker_num):
        # param names of all devices
        self.params = set([x[0].name for x in params_grads])
        # _param(str) -> device_id(int) 
        self.worker_idx = worker_idx
        self.worker_num = worker_num
        self.param2device = self._split_params(params_grads, worker_idx,
                                               worker_num)

    def _split_params(self, params_grads, worker_idx, worker_num):
        param2device = {}
        total_param_mem = 0.0
        param2mem = []
        for param in [x[0] for x in params_grads]:
            mem = get_var_size(param)
            total_param_mem += mem
            param2mem.append((param.name, mem))
        device2params = {x: [] for x in range(worker_num)}
        device_idx = 0
        mem_accu = 0.0
        for param_name, mem in param2mem:
            if mem_accu > total_param_mem * 1.0 * (device_idx + 1) / worker_num:
                device_idx += 1
            device2params[device_idx].append(param_name)
            param2device[param_name] = device_idx
            mem_accu += mem
        print(device2params)
        return param2device

    # def _is_opti_var(self, var_name, params):
    #     if var_name in self.params:
    #         return True
    #     for suffix in [
    #             "_moment1_0", "_moment2_0", "_beta1_pow_acc_0",
    #             "_beta2_pow_acc_0", "_velocity_0"
    #     ]:
    #         base_name = re.sub(suffix, '', var_name)
    #         if base_name in self.params:
    #             return True
    #     return False

    def _var_device_id(self, var_name):
        if var_name in self.param2device:
            return self.param2device[var_name]
        for suffix in [
                "_moment1_0", "_moment2_0", "_beta1_pow_acc_0",
                "_beta2_pow_acc_0", "_velocity_0"
        ]:
            base_name = re.sub(suffix, '', var_name)
            if base_name in self.param2device:
                return self.param2device[base_name]
        return -1

    def find_broadcast_params(self, block):
        print("----find_broadcast_params-----")
        broadcast_vars = set([])
        fp16_params = set([])
        fp16_to_fp32 = {}

        param_usage = {x: 0 for x in self.params}
        for op in block.ops:
            if is_optimizer_op(op):
                continue
            for input_name in op.desc.input_arg_names():
                if input_name in self.params:
                    param_usage[input_name] += 1

        for op in block.ops:
            print("-" * 20)
            print(op.type)
            if not FP16Utils.is_fp16_cast_op(block, op, self.params):
                continue
            print("is_fp16_cast_op")
            input_name = op.input_arg_names[0]
            output_name = op.output_arg_names[0]
            broadcast_vars.add(output_name)
            fp16_params.add(output_name)
            fp16_to_fp32[output_name] = input_name
            param_usage[input_name] -= 1
            self.param2device[output_name] = self.param2device[input_name]

        for param, usage in param_usage.items():
            if usage > 0:
                broadcast_vars.add(param)
        return broadcast_vars

    def has_param(self, var_name):
        return var_name in self.param2device and \
            self._var_device_id(var_name) == self.worker_idx

    def has_opt_var(self, var_name):
        return self._var_device_id(var_name) == self.worker_idx

    def has_var(self, var_name):
        return self._var_device_id(var_name) == -1 or \
            self._var_device_id(var_name) == self.worker_idx

    def device(self, var_name):
        return self._var_device_id(var_name)


class WeightDecayHelper(object):
    def __init__(self):
        pass

    def _is_weight_decay_op(self, op):
        return op.desc.has_attr("op_namescope") \
            and op.desc.attr("op_namescope").startswith("/regularization")

    def prune_weight_decay(self, block, device_vars):
        for idx, op in reversed(list(enumerate(block.ops))):
            if not self._is_weight_decay_op(op):
                continue
            if OP_ROLE_VAR_KEY not in op.attr_names:
                raise ValueError(
                    "The Weight Dacay op should hold op_role_var attribute"
                    "but the {} op does not hold op_role_var".format(op.type))
            op_role_var = op.all_attrs()[OP_ROLE_VAR_KEY]
            if not device_vars.has_param(op_role_var[0]):
                block._remove_op(idx)
        block._sync_with_cpp()


class GradientClipHelper(object):
    def __init__(self):
        pass

    def _is_gradient_clip_op(self, op):
        return op.desc.has_attr("op_namescope") \
            and op.desc.attr("op_namescope").startswith("/gradient_clip")

    def prune_gradient_clip(self, block, device_vars):
        deperated_vars = set()
        deperate_op_idx = set()
        for idx, op in enumerate(block.ops):
            if not self._is_gradient_clip_op(op):
                continue
            if op.type == "sum":
                continue
            deperate_op = False
            for input_name in op.desc.input_arg_names():
                if input_name in deperated_vars:
                    deperate_op = True
                param_name = input_name.strip("@GRAD")
                if device_vars.has_param(param_name):
                    deperate_op = True

            if deperate_op:
                deperate_op_idx.add(idx)
                for output_name in op.desc.output_arg_names():
                    deperated_vars.add(output_name)

        if not deperated_vars:
            # got no gradient_clip op
            return

        for idx, op in reversed(list(enumerate(block.ops))):
            if not self._is_gradient_clip_op(op):
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


def check_broadcast(block):
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


def check_allreduce_sum(block):
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
                        raise ValueError("There should be a sync_comm op "
                                         "after allreduce the Var: {}".format(
                                             var_name))
            for output_name in op.desc.output_arg_names():
                if output_name in var_status and \
                    var_status[output_name] == -1:
                    var_status[output_name] = 0
    print("finish check allreduce")


def insert_sync_calc_op(block, insert_idx, calc_dep_vars):
    """
    _insert_sync_calc_op
    """
    op_role = block.ops[insert_idx].attr('op_role')
    block._insert_op(
        insert_idx,
        type='c_sync_calc_stream',
        inputs={'X': calc_dep_vars},
        outputs={'Out': calc_dep_vars},
        attrs={OP_ROLE_KEY: op_role})
    return


def insert_sync_comm_ops(block, insert_idx, nrings, comm_dep_vars):
    """
    _insert_sync_comm_ops
    """
    op_role = block.ops[insert_idx].attr('op_role')
    for i in range(nrings):
        block._insert_op(
            insert_idx,
            type='c_sync_comm_stream',
            inputs={'X': comm_dep_vars},
            outputs={'Out': comm_dep_vars},
            attrs={'ring_id': i,
                   OP_ROLE_KEY: op_role})
    return nrings


def insert_fill_constant_ops(block, insert_idx, fill_constant_vars):
    """
    _add_fill_constant_ops
    """
    op_role = block.ops[insert_idx].attr('op_role')
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
                OP_ROLE_KEY: op_role
            })
    return


def insert_cast_ops(block, insert_idx, cast_ops):
    """
    _add_cast_ops
    """
    op_role = block.ops[insert_idx].attr('op_role')
    for fp16_name, fp32_name in cast_ops.items():
        block._insert_op(
            insert_idx,
            type="cast",
            inputs={"X": fp32_name},
            outputs={"Out": fp16_name},
            attrs={
                "in_dtype": core.VarDesc.VarType.FP32,
                "out_dtype": core.VarDesc.VarType.FP16,
                OP_ROLE_KEY: op_role
            })
    return


def insert_allreduce_ops(block, insert_idx, nrings, allreduce_vars):
    """
    _add_allreduce_ops
    """
    ring_id = -1
    for var in allreduce_vars:
        ring_id = (ring_id + 1) % nrings
        block._insert_op(
            insert_idx,
            type='c_allreduce_sum',
            inputs={'X': var},
            outputs={'Out': var},
            attrs={'ring_id': ring_id,
                   OP_ROLE_KEY: OpRole.Backward})
    return


def insert_broadcast_ops(block, insert_idx, nrings, broadcast2root):
    """
    _add_broadcast_ops
    """
    ring_id = -1
    op_role = block.ops[insert_idx].attr('op_role')
    for broadcast_name, root_device in broadcast2root:
        ring_id = (ring_id + 1) % nrings
        block._insert_op(
            insert_idx,
            type='c_broadcast',
            inputs={'X': broadcast_name},
            outputs={'Out': broadcast_name},
            attrs={
                'ring_id': ring_id,
                'root': root_device,
                OP_ROLE_KEY: op_role
            })
    return


DtypeToSize = {
    core.VarDesc.VarType.FP16: 2,
    core.VarDesc.VarType.FP32: 4,
    core.VarDesc.VarType.FP64: 8,
    core.VarDesc.VarType.INT16: 2,
    core.VarDesc.VarType.INT32: 4,
    core.VarDesc.VarType.INT64: 8,
    core.VarDesc.VarType.BOOL: 1,
    core.VarDesc.VarType.UINT8: 1,
}


def get_var_size(param):
    """
    input:
        - param: var
    return:
        var size in Bytes
    """
    assert -1 not in param.shape
    return reduce(lambda x, y: x * y,
                  param.shape) * DtypeToSize[param.dtype] / 1024.0 / 1024.0


def insert_scale_loss_grad_ops(block, scale=1.0):
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
