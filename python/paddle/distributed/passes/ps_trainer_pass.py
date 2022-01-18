# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from ..ps.utils.public import *
from paddle.framework import core
from .pass_base import PassBase, register_pass
from paddle.fluid.transpiler.details.program_utils import delete_ops

OP_NAME_SCOPE = "op_namescope"
CLIP_OP_NAME_SCOPE = "gradient_clip"
STEP_COUNTER = "@PS_STEP_COUNTER@"
OP_ROLE_VAR_ATTR_NAME = core.op_proto_and_checker_maker.kOpRoleVarAttrName()
RPC_OP_ROLE_ATTR_NAME = core.op_proto_and_checker_maker.kOpRoleAttrName()
RPC_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.RPC
LR_SCHED_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.LRSched
OPT_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.Optimize
op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()

SPARSE_OP_TYPE_DICT = {"lookup_table": "W", "lookup_table_v2": "W"}
SPARSE_GRAD_OP_TYPE_DICT = {
    "lookup_table_grad": "W",
    "lookup_table_v2_grad": "W"
}
DEVICE_LIST = ["cpu", "gpu", "xpu"]
COMMUNICATE_OPS_TYPE = ["send", "recv", "fetch_barrier", "send_barrier"]
DEFAULT_DEVICE = 'cpu'

__all__ = [
    'append_send_ops_pass', 'ps_gpu_pass', 'ps_transpile_pass',
    'delete_optimizer_pass'
    'distributed_ops_pass', 'split_heter_worker_ops_pass',
    'split_trainer_ops_pass', 'delete_extra_optimizer_pass',
    'fake_init_ops_pass'
]


@register_pass("append_send_ops_pass")
class AppendSendOpsPass(PassBase):  # 该 pass 被多种模式复用
    def __init__(self):
        super(AppendSendOpsPass, self).__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _append_send_op(program, union_vars, queue, is_sparse, table_id):
        if queue == STEP_COUNTER:
            send_input_vars = []
        else:
            send_input_vars = [
                program.global_block().vars[union_var]
                for union_var in union_vars
            ]

        dummy_output = []
        if mode in [DistributedMode.SYNC, DistributedMode.HALF_ASYNC]:
            dummy_output = program.global_block().create_var(
                name=framework.generate_control_dev_var_name())

        program.global_block().append_op(
            type="send",
            inputs={"X": send_input_vars},
            outputs={"Out": dummy_output},
            attrs={
                "send_varnames": [queue],
                "is_sparse": is_sparse,
                "table_id": table_id,
                RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
            })

        return dummy_output

    def _append_barrier_op(program, dummys):
        program.global_block().append_op(
            type="send_barrier",
            inputs={"X": dummys},
            outputs={"Out": []},
            attrs={
                "trainer_id": trainer_id,
                "half_async": True,
                RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
            })

    def _apply_single_impl(self, main_program, startup_program, context):
        ps_mode = context['ps_mode']
        if ps_mode == DistributedMode.GEO:
            send_ctx = get_geo_trainer_send_context(context)  # geo 模式
        else:
            send_ctx = get_the_one_send_context(context)  # async、sync 等各种模式
        dummys = []
        for merged_name, send in send_ctx.items():
            if send.is_sparse() and ps_mode != DistributedMode.GEO:
                continue
            is_sparse = 1 if send.is_sparse() else 0
            is_sparse = 2 if send.is_distributed() else is_sparse
            dummys.append(
                self._append_send_op(main_program,
                                     send.origin_varnames(), merged_name,
                                     is_sparse, send.table_id()))

        if ps_mode in [DistributedMode.SYNC, DistributedMode.HALF_ASYNC]:
            self._append_barrier_op(main_program, dummys)


@register_pass("distributed_ops_pass")
class DistributedOpsPass(PassBase):
    def __init__(self):
        super(DistributedOpsPass, self).__init__()
        w_2_table_id = {}
        emb_size = {}

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _push_sparse_fuse(_program, push_sparse_ops, use_ps_gpu, send_ctx):
        if use_ps_gpu:
            return
        if len(push_sparse_ops) == 0:
            return
        show = None
        clk = None
        use_entry = False
        for param, ops in push_sparse_ops.items():
            op_first = ops[0]
            break
        if op_first.has_attr("entry"):
            entry = op_first.attr("entry")
            entry = entry.split(':')
            if len(entry) == 3 and entry[0] == 'show_click_entry':
                show_var_name = entry[1]
                click_var_name = entry[2]
                if show_var_name in program.global_block(
                ).vars and click_var_name in program.global_block().vars:
                    show = program.global_block().vars[show_var_name]
                    clk = program.global_block().vars[click_var_name]
                    use_entry = True
                else:
                    warnings.warn(
                        'ShowClickEntry configured, but cannot find show/click var, will not use'
                    )

        if not use_entry:
            print('ShowClickEntry not configured, will not use')
            show = program.global_block().create_var(
                name="show",
                dtype=core.VarDesc.VarType.INT64,
                persistable=False,
                stop_gradient=True)
            program.global_block()._insert_op(
                index=0,
                type='fill_constant',
                inputs={},
                outputs={'Out': show},
                attrs={
                    'shape': [1],
                    'dtype': show.dtype,
                    'value': 1,
                })

            clk = program.global_block().create_var(
                name="clk",
                dtype=core.VarDesc.VarType.INT64,
                persistable=False,
                stop_gradient=True)
            program.global_block()._insert_op(
                index=0,
                type='fill_constant',
                inputs={},
                outputs={'Out': clk},
                attrs={
                    'shape': [1],
                    'dtype': clk.dtype,
                    'value': 0,
                })

        for param, ops in push_sparse_ops.items():
            all_ops = program.global_block().ops
            op_idxs = [all_ops.index(op) for op in ops]
            inputs = [
                program.global_block().vars[op.input("Ids")[0]] for op in ops
            ]
            w = program.global_block().vars[ops[0].output("W@GRAD")[0]]
            table_id = w_2_table_id[param]

            padding_idx = ops[0].attr("padding_idx")
            is_distributed = ops[0].attr("is_distributed")
            op_type = ops[0].type
            outputs = [
                program.global_block().vars[op.input("Out@GRAD")[0]]
                for op in ops
            ]

            for idx in op_idxs[::-1]:
                program.global_block()._remove_op(idx)

            program.global_block().append_op(
                type="distributed_push_sparse",
                inputs={
                    "Ids": inputs,
                    'W': w,
                    "Outputs": outputs,
                    "Shows": show,
                    "Clicks": clk
                },
                outputs={"Outputs": outputs},
                attrs={
                    "is_distributed": is_distributed,
                    "padding_idx": padding_idx,
                    "table_id": table_id,
                    "size": emb_size[param]
                })

    def _pull_sparse_fuse(_program, pull_sparse_ops, use_ps_gpu):
        def dag_check_up_and_reorder(program, inputs, outputs):
            global_block = program.global_block()
            min_output_index = len(global_block.ops)
            max_input_index = -1
            input_indexes = [0] * len(global_block.ops)
            output_indexes = [0] * len(global_block.ops)
            for idx, op in enumerate(global_block.ops):
                for i in range(0, len(op.output_names)):
                    if input_indexes[idx] == 1:
                        break
                    outs = op.output(op.output_names[i])
                    for in_id, in_var in enumerate(inputs):
                        if in_var.name in outs:
                            input_indexes[idx] = 1
                            max_input_index = max(max_input_index, idx)
                            break

                for i in range(0, len(op.input_names)):
                    if output_indexes[idx] == 1:
                        break
                    ins = op.input(op.input_names[i])
                    for out_id, out_var in enumerate(outputs):
                        if out_var.name in ins:
                            output_indexes[idx] = 1
                            min_output_index = min(min_output_index, idx)

            for i in range(len(global_block.ops)):
                if input_indexes[i] == 1 and output_indexes[i] == 1:
                    warnings.warn(
                        "unable to re-arrange dags order to combine distributed embedding ops because a op both needs embedding table's output as input and produces ids as the same embedding table's input"
                    )
                    return

            if min_output_index < max_input_index:
                move_ops = []
                for i in range(min_output_index + 1, len(input_indexes)):
                    if input_indexes[i] == 1:
                        move_ops.append((global_block.ops[i], i))
                for i, op in enumerate(move_ops):
                    queue = list()
                    visited = set()
                    queue.append(op[1])
                    visited.add(op[0])
                    start = 0
                    while start < len(queue):
                        pos = queue[start]
                        op = global_block.ops[pos]
                        op_inputs = []
                        for k in range(0, len(op.input_names)):
                            ins = op.input(op.input_names[k])
                            op_inputs.append(ins)
                        for j in range(pos - 1, min_output_index - 1, -1):
                            op1 = global_block.ops[j]
                            if op1 in visited:
                                continue
                            found = False
                            for k in range(0, len(op1.output_names)):
                                outs = op1.output(op1.output_names[k])
                                for t in range(len(op_inputs)):
                                    for y in op_inputs[t]:
                                        if y in outs:
                                            found = True
                                            break
                                    if found:
                                        break
                                if found:
                                    break
                            if found:
                                if output_indexes[j] == True:
                                    warnings.warn(
                                        "unable to re-arrange dags order to combine distributed embedding ops"
                                    )
                                    return
                                queue.append(j)
                                visited.add(global_block.ops[j])
                        start = start + 1

                    queue.sort()
                    for index in queue:
                        desc = global_block.desc._insert_op(min_output_index)
                        desc.copy_from(global_block.ops[index].desc)
                        global_block.desc._remove_op(index + 1, index + 2)
                        global_block.ops[index].desc = desc
                        insert_op = global_block.ops.pop(index)
                        input_state = input_indexes.pop(index)
                        output_state = output_indexes.pop(index)
                        global_block.ops.insert(min_output_index, insert_op)
                        input_indexes.insert(min_output_index, input_state)
                        output_indexes.insert(min_output_index, output_state)
                        min_output_index = min_output_index + 1

                assert global_block.desc.op_size() == len(global_block.ops)
                for i in range(len(global_block.ops)):
                    assert global_block.desc.op(i) == global_block.ops[i].desc

        for param, ops in pull_sparse_ops.items():
            all_ops = program.global_block().ops
            op_device = ""
            if config.is_heter_ps_mode:
                op_device = ops[0].attr("op_device")
            inputs = [
                program.global_block().vars[op.input("Ids")[0]] for op in ops
            ]
            w = program.global_block().vars[ops[0].input("W")[0]]
            emb_size[param] = w.shape[1]

            grad_name = config.param_name_to_grad_name[w.name]

            table_id = -1

            for name, ctx in send_ctx.items():
                if grad_name in ctx.origin_varnames():
                    table_id = ctx.table_id()

            if table_id == -1:
                raise ValueError(
                    "can not find suitable sparse table, please check")

            w_2_table_id[param] = table_id
            padding_idx = ops[0].attr("padding_idx")
            is_distributed = ops[0].attr("is_distributed")
            op_type = ops[0].type

            outputs = [
                program.global_block().vars[op.output("Out")[0]] for op in ops
            ]

            dag_check_up_and_reorder(program, inputs, outputs)

            op_idxs = [all_ops.index(op) for op in ops]

            for idx in op_idxs[::-1]:
                program.global_block()._remove_op(idx)

            inputs_idxs = [-1] * len(inputs)
            outputs_idxs = [len(program.global_block().ops) + 1] * len(outputs)

            for idx, op in enumerate(program.global_block().ops):
                for i in range(0, len(op.output_names)):
                    outs = op.output(op.output_names[i])
                    for in_id, in_var in enumerate(inputs):
                        if in_var.name in outs:
                            inputs_idxs[in_id] = max(idx, inputs_idxs[in_id])
                for i in range(0, len(op.input_names)):
                    ins = op.input(op.input_names[i])
                    for out_id, out_var in enumerate(outputs):
                        if out_var.name in ins:
                            outputs_idxs[out_id] = min(idx,
                                                       outputs_idxs[out_id])

            if min(outputs_idxs) - max(inputs_idxs) >= 1:
                if max(inputs_idxs) == -1:
                    distributed_idx = min(op_idxs)
                else:
                    distributed_idx = max(inputs_idxs) + 1

                if use_ps_gpu:
                    program.global_block()._insert_op(
                        index=distributed_idx,
                        type="pull_box_sparse",
                        inputs={"Ids": inputs,
                                'W': w},
                        outputs={"Out": outputs},
                        attrs={
                            "size": w.shape[1],
                            "is_distributed": True,
                            "is_sparse": True
                        })
                else:
                    program.global_block()._insert_op(
                        index=distributed_idx,
                        type="distributed_lookup_table",
                        inputs={"Ids": inputs,
                                'W': w},
                        outputs={"Outputs": outputs},
                        attrs={
                            "is_distributed": is_distributed,
                            "padding_idx": padding_idx,
                            "table_id": table_id,
                            "lookup_table_version": op_type,
                            "op_device": op_device
                        })
            else:
                for i in range(len(inputs_idxs)):
                    distributed_idx = op_idxs[i]

                    program.global_block()._insert_op(
                        index=distributed_idx,
                        type="distributed_lookup_table",
                        inputs={"Ids": [inputs[i]],
                                'W': w},
                        outputs={"Outputs": [outputs[i]]},
                        attrs={
                            "is_distributed": is_distributed,
                            "padding_idx": padding_idx,
                            "table_id": table_id,
                            "lookup_table_version": op_type,
                            "op_device": op_device
                        })

    def _get_pull_sparse_ops(_program):
        pull_sparse_ops = {}
        pull_sparse_ids = {}
        push_sparse_ops = {}
        ops = {}
        for op in _program.global_block().ops:
            if op.type in SPARSE_OP_TYPE_DICT.keys() \
                    and op.attr('remote_prefetch') is True:
                param_name = op.input(SPARSE_OP_TYPE_DICT[op.type])[0]
                if config.is_heter_ps_mode:
                    # trick for matchnet, need to modify
                    param_name += op.input("Ids")[0][0]
                ops = pull_sparse_ops.get(param_name, [])
                ops.append(op)
                pull_sparse_ops[param_name] = ops
                ids = pull_sparse_ids.get(param_name, [])
                ids.append(op.input("Ids")[0])
                pull_sparse_ids[param_name] = ids
        for op in _program.global_block().ops:
            if op.type in SPARSE_GRAD_OP_TYPE_DICT.keys():
                param_name = op.input(SPARSE_GRAD_OP_TYPE_DICT[op.type])[0]
                if param_name in pull_sparse_ids and op.input("Ids")[
                        0] in pull_sparse_ids[param_name]:
                    ops = push_sparse_ops.get(param_name, [])
                    ops.append(op)
                    push_sparse_ops[param_name] = ops

        return pull_sparse_ops, push_sparse_ops

    def _apply_single_impl(self, main_program, startup_program, context):
        pull_sparse_ops, push_sparse_ops = self._get_pull_sparse_ops(program)
        send_ctx = get_the_one_send_context(
            context, split_dense_table=context['is_heter_ps_mode'])
        self._pull_sparse_fuse(main_program, pull_sparse_ops,
                               context['use_ps_gpu'], send_ctx)
        self._push_sparse_fuse(main_program, push_sparse_ops,
                               context['use_ps_gpu'])


@register_pass("delete_optimizer_pass")
class DeleteOptimizesPass(PassBase):
    def __init__(self):
        super(DeleteOptimizesPass, self).__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _delete_optimizer_op_and_vars(_program, optimize_ops):
        optimize_vars = []
        optimize_op_role_vars = []
        optimize_need_delete_vars = []

        for op in optimize_ops:
            optimize_vars.extend(op.input_arg_names)
            optimize_op_role_vars.extend(op.attr("op_role_var"))

        optimize_vars = list(set(optimize_vars))
        optimize_op_role_vars = list(set(optimize_op_role_vars))

        for var in optimize_vars:
            if var not in optimize_op_role_vars:
                optimize_need_delete_vars.append(var)
        need_delete_optimize_vars = list(set(optimize_need_delete_vars))

        delete_ops(_program.global_block(), optimize_ops)
        for var in need_delete_optimize_vars:
            if _program.global_block().has_var(var):
                _program.global_block()._remove_var(var)

    def _add_lr_var(main_program, context):
        # Todo: hard code for pe
        lr_var = context['origin_main_program'].global_block().vars[
            "learning_rate_0"]
        main_program.global_block().create_var(
            name=lr_var.name,
            shape=lr_var.shape,
            dtype=lr_var.dtype,
            type=lr_var.type,
            lod_level=lr_var.lod_level,
            persistable=True)

    def _apply_single_impl(self, main_program, startup_program, context):
        optimizer_ops = get_optimize_ops(main_program)
        lr_ops = get_lr_ops(main_program)
        optimizer_ops.extend(lr_ops)
        self._delete_optimizer_op_and_vars(main_program, optimizer_ops)

        if hasattr(context['origin_main_program'], 'lr_sheduler'):
            self._add_lr_var(main_program, context)


@register_pass("fake_init_ops_pass")
class FakeInitOpsPass(PassBase):
    def __init__(self):
        super(FakeInitOpsPass, self).__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _get_sparse_table_names():
        dist_varnames = get_sparse_tablenames(context['origin_program'], True)
        sparse_varnames = get_sparse_tablenames(context['origin_program'],
                                                False)
        return list(set(dist_varnames + sparse_varnames))

    def _fake_init_sparsetable(program, sparse_table_names):
        # delete table init op
        for table_name in sparse_table_names:
            table_var = program.global_block().vars[table_name]
            table_param_init_op = []
            for op in program.global_block().ops:
                if table_name in op.output_arg_names:
                    table_param_init_op.append(op)
            init_op_num = len(table_param_init_op)
            if init_op_num != 1:
                raise ValueError("table init op num should be 1, now is " + str(
                    init_op_num))
            table_init_op = table_param_init_op[0]
            program.global_block().append_op(
                type="fake_init",
                inputs={},
                outputs={"Out": table_var},
                attrs={"shape": table_init_op.attr('shape')})
            delete_ops(program.global_block(), table_param_init_op)

    def _apply_single_impl(self, main_program, startup_program, context):
        sparse_tables = self._get_sparse_table_names(context)
        self._fake_init_sparsetable(startup_program, sparse_tables)


@register_pass("ps_gpu_pass")
class PsGpuPass(PassBase):
    def __init__(self):
        super(PsGpuPass, self).__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        pass


@register_pass("ps_transpile_pass")
class PsTranspilePass(PassBase):
    def __init__(self):
        super(PsTranspilePass, self).__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        pass


@register_pass("split_heter_worker_ops_pass")
class SplitHeterWorkerOpsPass(PassBase):
    def __init__(self):
        super(SplitHeterWorkerOpsPass, self).__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        pass


@register_pass("split_trainer_ops_pass")
class SplitTrainerOpsPass(PassBase):
    def __init__(self):
        super(SplitTrainerOpsPass, self).__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        pass


@register_pass("set_heter_pipeline_opt_pass")
class SetHeterPipelineOptPass(PassBase):
    def __init__(self):
        super(SetHeterPipelineOptPass, self).__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        pass


@register_pass("delete_extra_optimizer_pass")
class DeleteExtraOptimizerPass(PassBase):
    def __init__(self):
        super(DeleteExtraOptimizerPass, self).__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        pass
