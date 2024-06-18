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

import os

from _collections import defaultdict

import paddle
from paddle.base import framework
from paddle.distributed.passes.pass_base import PassBase, register_pass
from paddle.framework import core
from paddle.static import Parameter, Program

from ..ps.utils.public import *  # noqa: F403


@register_pass("append_send_ops_pass")
class AppendSendOpsPass(PassBase):  # 该 pass 被多种模式复用
    def __init__(self):
        super().__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _append_send_op(
        self, program, union_vars, queue, is_sparse, table_id, ps_mode
    ):
        if queue == STEP_COUNTER:
            send_input_vars = []
        else:
            send_input_vars = [
                program.global_block().vars[union_var]
                for union_var in union_vars
            ]

        dummy_output = []
        if ps_mode in [DistributedMode.SYNC, DistributedMode.HALF_ASYNC]:
            dummy_output = program.global_block().create_var(
                name=framework.generate_control_dev_var_name()
            )
        program.global_block().append_op(
            type="send",
            inputs={"X": send_input_vars},
            outputs={"Out": dummy_output},
            attrs={
                "send_varnames": [queue],
                "is_sparse": is_sparse,
                "table_id": table_id,
                RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE,
            },
        )

        return dummy_output

    def _append_barrier_op(self, program, dummys, trainer_id):
        program.global_block().append_op(
            type="send_barrier",
            inputs={"X": dummys},
            outputs={"Out": []},
            attrs={
                "trainer_id": trainer_id,
                "half_async": True,
                RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE,
            },
        )

    def _apply_single_impl(self, main_program, startup_program, pass_ctx):
        attrs = pass_ctx._attrs
        ps_mode = attrs['ps_mode']
        # if ps_mode == DistributedMode.GEO:
        #   send_ctx = get_geo_trainer_send_context(attrs)  # geo 模式, 没必要
        send_ctx = get_the_one_send_context(
            attrs, split_dense_table=attrs['is_heter_ps_mode']
        )  # async、sync 等各种模式

        dummys = []
        for merged_name, send in send_ctx.items():  # embedding_0.w_0@GRAD
            if send.is_sparse() and ps_mode != DistributedMode.GEO:
                continue
            if (not send.is_sparse()) and ps_mode == DistributedMode.GEO:
                continue
            if send.program_id() != id(attrs['loss'].block.program):
                continue
            if len(send.remote_sparse_ids()) > 0:
                continue
            is_sparse = 1 if send.is_sparse() else 0
            is_sparse = 2 if send.is_distributed() else is_sparse
            dummys.append(
                self._append_send_op(
                    main_program,
                    send.origin_varnames(),
                    merged_name,
                    is_sparse,
                    send.table_id(),
                    ps_mode,
                )
            )
        if ps_mode in [DistributedMode.SYNC, DistributedMode.HALF_ASYNC]:
            trainer_id = get_role_id(attrs['role_maker'])
            self._append_barrier_op(main_program, dummys, trainer_id)


@register_pass("distributed_ops_pass")
class DistributedOpsPass(PassBase):
    def __init__(self):
        super().__init__()
        self.w_2_table_id = {}
        self.emb_size = {}

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _push_sparse_fuse(self, _program, push_sparse_ops, attrs, use_cvm_op):
        if attrs['use_ps_gpu']:
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
                if (
                    show_var_name in _program.global_block().vars
                    and click_var_name in _program.global_block().vars
                ):
                    show = _program.global_block().vars[show_var_name]
                    clk = _program.global_block().vars[click_var_name]
                    use_entry = True
                else:
                    warnings.warn(
                        'ShowClickEntry configured, but cannot find show/click var, will not use'
                    )

        if not use_entry:
            print('ShowClickEntry not configured, will not use')
            show = _program.global_block().create_var(
                name="show",
                dtype=core.VarDesc.VarType.FP32,
                persistable=False,
                stop_gradient=True,
            )
            _program.global_block()._insert_op(
                index=0,
                type='fill_constant',
                inputs={},
                outputs={'Out': show},
                attrs={
                    'shape': [1],
                    'dtype': show.dtype,
                    'value': 1,
                },
            )

            clk = _program.global_block().create_var(
                name="clk",
                dtype=core.VarDesc.VarType.FP32,
                persistable=False,
                stop_gradient=True,
            )
            _program.global_block()._insert_op(
                index=0,
                type='fill_constant',
                inputs={},
                outputs={'Out': clk},
                attrs={
                    'shape': [1],
                    'dtype': clk.dtype,
                    'value': 0,
                },
            )

        for param, ops in push_sparse_ops.items():
            all_ops = _program.global_block().ops
            op_idxs = [all_ops.index(op) for op in ops]
            inputs = [
                _program.global_block().vars[op.input("Ids")[0]] for op in ops
            ]
            w = _program.global_block().vars[ops[0].output("W@GRAD")[0]]
            table_id = self.w_2_table_id[param]

            padding_idx = ops[0].attr("padding_idx")
            is_distributed = ops[0].attr("is_distributed")
            op_type = ops[0].type

            slots = [op.attr("slot") for op in ops]
            print('debug zcb slots: ', slots)
            outputs = [
                _program.global_block().vars[op.input("Out@GRAD")[0]]
                for op in ops
            ]

            for idx in op_idxs[::-1]:
                _program.global_block()._remove_op(idx)

            _program.global_block().append_op(
                type="distributed_push_sparse",
                inputs={
                    "Ids": inputs,
                    'W': w,
                    "Outputs": outputs,
                    "Shows": show,
                    "Clicks": clk,
                },
                outputs={"Outputs": outputs},
                attrs={
                    "is_distributed": is_distributed,
                    "padding_idx": padding_idx,
                    "table_id": table_id,
                    "size": self.emb_size[param],
                    "use_cvm_op": use_cvm_op,
                    "slots": slots,
                },
            )

    def _pull_sparse_fuse(self, _program, pull_sparse_ops, attrs, send_ctx):
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
                    queue = []
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
                                if output_indexes[j]:
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

        if attrs['use_ps_gpu']:
            gpups_inputs_idxs = []
            gpups_outputs_idxs = []
            gpups_inputs = []
            gpups_outputs = []
            gpups_w_size = []
            gpups_min_distributed_idx = len(_program.global_block().ops) + 1

        for param, ops in pull_sparse_ops.items():
            all_ops = _program.global_block().ops
            op_device = ""
            if attrs['is_heter_ps_mode']:
                op_device = ops[0].attr("op_device")
            inputs = [
                _program.global_block().vars[op.input("Ids")[0]] for op in ops
            ]
            w = _program.global_block().vars[ops[0].input("W")[0]]
            self.emb_size[param] = w.shape[1]

            grad_name = attrs['param_name_to_grad_name'][w.name]

            table_id = -1

            for name, ctx in send_ctx.items():
                if grad_name in ctx.origin_varnames():
                    table_id = ctx.table_id()

            if table_id == -1:
                raise ValueError(
                    "can not find suitable sparse table, please check"
                )

            self.w_2_table_id[param] = table_id
            padding_idx = ops[0].attr("padding_idx")
            is_distributed = ops[0].attr("is_distributed")
            op_type = ops[0].type

            outputs = [
                _program.global_block().vars[op.output("Out")[0]] for op in ops
            ]

            dag_check_up_and_reorder(_program, inputs, outputs)

            op_idxs = [all_ops.index(op) for op in ops]

            for idx in op_idxs[::-1]:
                _program.global_block()._remove_op(idx)

            inputs_idxs = [-1] * len(inputs)
            outputs_idxs = [len(_program.global_block().ops) + 1] * len(outputs)

            for idx, op in enumerate(_program.global_block().ops):
                for i in range(0, len(op.output_names)):
                    outs = op.output(op.output_names[i])
                    for in_id, in_var in enumerate(inputs):
                        if in_var.name in outs:
                            inputs_idxs[in_id] = max(idx, inputs_idxs[in_id])
                for i in range(0, len(op.input_names)):
                    ins = op.input(op.input_names[i])
                    for out_id, out_var in enumerate(outputs):
                        if out_var.name in ins:
                            outputs_idxs[out_id] = min(
                                idx, outputs_idxs[out_id]
                            )

            if attrs['use_ps_gpu']:
                gpups_inputs_idxs.extend(inputs_idxs)
                gpups_outputs_idxs.extend(outputs_idxs)
                gpups_inputs.extend(inputs)
                gpups_outputs.extend(outputs)
                gpups_w_size.extend([w.shape[1]] * len(inputs))
                gpups_min_distributed_idx = min(
                    *op_idxs, gpups_min_distributed_idx
                )
                continue

            if min(outputs_idxs) - max(inputs_idxs) >= 1:
                if max(inputs_idxs) == -1:
                    distributed_idx = min(op_idxs)
                else:
                    distributed_idx = max(inputs_idxs) + 1

                _program.global_block()._insert_op(
                    index=distributed_idx,
                    type="distributed_lookup_table",
                    inputs={"Ids": inputs, 'W': w},
                    outputs={"Outputs": outputs},
                    attrs={
                        "is_distributed": is_distributed,
                        "padding_idx": padding_idx,
                        "table_id": table_id,
                        "lookup_table_version": op_type,
                        "op_device": op_device,
                    },
                )
            else:
                for i in range(len(inputs_idxs)):
                    distributed_idx = op_idxs[i]

                    _program.global_block()._insert_op(
                        index=distributed_idx,
                        type="distributed_lookup_table",
                        inputs={"Ids": [inputs[i]], 'W': w},
                        outputs={"Outputs": [outputs[i]]},
                        attrs={
                            "is_distributed": is_distributed,
                            "padding_idx": padding_idx,
                            "table_id": table_id,
                            "lookup_table_version": op_type,
                            "op_device": op_device,
                        },
                    )

        if attrs['use_ps_gpu'] and len(gpups_inputs) > 0:
            if max(gpups_inputs_idxs) > 0:
                raise ValueError("There can't be ops before embedding in gpups")

            _program.global_block()._insert_op(
                index=gpups_min_distributed_idx,
                type="pull_gpups_sparse",
                inputs={
                    "Ids": gpups_inputs,
                },
                outputs={"Out": gpups_outputs},
                attrs={
                    "size": gpups_w_size,
                    "is_distributed": True,
                    "is_sparse": True,
                },
            )
            PSGPU = core.PSGPU()
            try:
                gpu_slot = [int(var.name) for var in gpups_inputs]
            except ValueError:
                raise ValueError(
                    "The slot name in gpups Should be able to convert to integer."
                )
            PSGPU.set_slot_vector(gpu_slot)
            gpu_mf_sizes = [x - 3 for x in gpups_w_size]
            PSGPU.set_slot_dim_vector(gpu_mf_sizes)

    def _get_pull_sparse_ops(self, _program, attrs):
        pull_sparse_ops = {}
        pull_sparse_ids = {}
        push_sparse_ops = {}
        ops = {}
        use_cvm_op = False
        for op in _program.global_block().ops:
            if (
                op.type in SPARSE_OP_TYPE_DICT.keys()
                and op.attr('remote_prefetch') is True
            ):
                param_name = op.input(SPARSE_OP_TYPE_DICT[op.type])[0]
                if attrs['is_heter_ps_mode'] and not attrs['is_fl_ps_mode']:
                    # TODO: trick for matchnet, need to modify for heter_ps
                    param_name += op.input("Ids")[0][0]
                if param_name in attrs['local_sparse']:  # for recall/ncf model
                    continue
                ops = pull_sparse_ops.get(param_name, [])
                ops.append(op)
                pull_sparse_ops[param_name] = ops
                ids = pull_sparse_ids.get(param_name, [])
                ids.append(op.input("Ids")[0])
                pull_sparse_ids[param_name] = ids
            if op.type == 'cvm':
                use_cvm_op = True

        for op in _program.global_block().ops:
            if op.type in SPARSE_GRAD_OP_TYPE_DICT.keys():
                param_name = op.input(SPARSE_GRAD_OP_TYPE_DICT[op.type])[0]
                if (
                    param_name in pull_sparse_ids
                    and op.input("Ids")[0] in pull_sparse_ids[param_name]
                ):
                    ops = push_sparse_ops.get(param_name, [])
                    ops.append(op)
                    push_sparse_ops[param_name] = ops

        return pull_sparse_ops, push_sparse_ops, use_cvm_op

    def _apply_single_impl(self, main_program, startup_program, pass_ctx):
        attrs = pass_ctx._attrs
        (
            pull_sparse_ops,
            push_sparse_ops,
            use_cvm_op,
        ) = self._get_pull_sparse_ops(main_program, attrs)
        print(
            "is_heter_ps_mode in distributed_ops_pass {}?".format(
                attrs['is_heter_ps_mode']
            )
        )
        send_ctx = get_the_one_send_context(
            attrs, split_dense_table=attrs['is_heter_ps_mode']
        )
        self._pull_sparse_fuse(main_program, pull_sparse_ops, attrs, send_ctx)
        self._push_sparse_fuse(main_program, push_sparse_ops, attrs, use_cvm_op)


@register_pass("delete_optimizer_pass")
class DeleteOptimizesPass(PassBase):
    def __init__(self):
        super().__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _delete_optimizer_op_and_vars(
        self, _program, remote_optimize_ops, local_optimize_ops
    ):
        local_optimize_vars = []
        remote_optimize_vars = []
        remote_optimize_op_role_vars = []
        optimize_need_delete_vars = []

        for op in local_optimize_ops:
            local_optimize_vars.extend(op.input_arg_names)

        for op in remote_optimize_ops:
            remote_optimize_vars.extend(op.input_arg_names)
            remote_optimize_op_role_vars.extend(op.attr("op_role_var"))

        remote_optimize_vars = list(
            set(remote_optimize_vars)
        )  # param + grad + optimizer_state + learning_rate
        remote_optimize_op_role_vars = list(
            set(remote_optimize_op_role_vars)
        )  # param + grad
        print(
            f"remote_optimize_vars: {remote_optimize_vars}, remote_optimize_op_role_vars: {remote_optimize_op_role_vars}, local_optimize_vars: {local_optimize_vars}"
        )
        for var in remote_optimize_vars:
            if var in local_optimize_vars:
                continue
            if var not in remote_optimize_op_role_vars:
                optimize_need_delete_vars.append(var)
        need_delete_optimize_vars = list(set(optimize_need_delete_vars))

        delete_ops(_program.global_block(), remote_optimize_ops)
        for var in need_delete_optimize_vars:
            if _program.global_block().has_var(var):
                _program.global_block()._remove_var(var)

    def _add_lr_var(self, main_program, attrs):
        # Todo: hard code for pe
        lr_var = (
            attrs['origin_main_program'].global_block().vars["learning_rate_0"]
        )
        main_program.global_block().create_var(
            name=lr_var.name,
            shape=lr_var.shape,
            dtype=lr_var.dtype,
            type=lr_var.type,
            lod_level=lr_var.lod_level,
            persistable=True,
        )

    def _apply_single_impl(self, main_program, startup_program, pass_ctx):
        attrs = pass_ctx._attrs
        all_optimize_ops = get_optimize_ops(main_program)
        remote_optimize_ops = get_optimize_ops(
            main_program, attrs['remote_sparse']
        )
        lr_ops = get_lr_ops(main_program)
        remote_optimize_ops.extend(lr_ops)
        local_optimize_ops = list(
            set(all_optimize_ops) - set(remote_optimize_ops)
        )
        self._delete_optimizer_op_and_vars(
            main_program, remote_optimize_ops, local_optimize_ops
        )

        if hasattr(attrs['origin_main_program'], 'lr_scheduler'):
            self._add_lr_var(main_program, attrs)


@register_pass("delete_extra_optimizer_pass")
class DeleteExtraOptimizerPass(PassBase):
    def __init__(self):
        super().__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, pass_ctx):
        attrs = pass_ctx._attrs
        remote_optimize_vars = []
        remote_optimize_op_role_vars = []
        optimize_need_delete_vars = []
        all_optimize_ops = get_optimize_ops(main_program)
        remote_optimize_ops = get_optimize_ops(
            main_program, attrs['remote_sparse']
        )
        local_optimize_ops = list(
            set(all_optimize_ops) - set(remote_optimize_ops)
        )

        local_optimize_vars = []
        for op in local_optimize_ops:
            local_optimize_vars.extend(op.input_arg_names)

        for op in remote_optimize_ops:
            remote_optimize_vars.extend(op.input_arg_names)
            remote_optimize_op_role_vars.extend(op.attr("op_role_var"))

        remote_optimize_vars = list(set(remote_optimize_vars))
        remote_optimize_op_role_vars = list(set(remote_optimize_op_role_vars))
        for var in remote_optimize_vars:
            if var in local_optimize_vars:
                continue
            if 'learning_rate_0' == var:
                continue
            if var not in remote_optimize_op_role_vars:
                optimize_need_delete_vars.append(var)
        need_delete_optimize_vars = list(set(optimize_need_delete_vars))

        init_ops = []
        for var in need_delete_optimize_vars:
            param_init_op = []
            for op in startup_program.global_block().ops:
                if var in op.output_arg_names:
                    param_init_op.append(op)
            init_ops.extend(param_init_op)
        delete_ops(startup_program.global_block(), init_ops)

        for var in need_delete_optimize_vars:
            if startup_program.global_block().has_var(var):
                startup_program.global_block()._remove_var(var)


@register_pass("fake_init_ops_pass")
class FakeInitOpsPass(PassBase):
    def __init__(self):
        super().__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _get_sparse_table_names(self, attrs):
        dist_varnames = get_sparse_tablenames(
            attrs['origin_main_programs'], True
        )
        sparse_varnames = get_sparse_tablenames(
            attrs['origin_main_programs'], False
        )
        return list(set(dist_varnames + sparse_varnames))

    def _fake_init_sparsetable(
        self, startup_program, sparse_table_names, attrs
    ):
        # delete table init op
        for table_name in sparse_table_names:
            table_var = startup_program.global_block().vars[table_name]
            if (
                str(table_var).split(":")[0].strip().split()[-1]
                in attrs['local_sparse']
            ):
                continue
            table_param_init_op = []
            for op in startup_program.global_block().ops:
                if table_name in op.output_arg_names:
                    table_param_init_op.append(op)
            init_op_num = len(table_param_init_op)
            if init_op_num != 1:
                raise ValueError(
                    "table init op num should be 1, now is " + str(init_op_num)
                )
            table_init_op = table_param_init_op[0]
            startup_program.global_block().append_op(
                type="fake_init",
                inputs={},
                outputs={"Out": table_var},
                attrs={"shape": table_init_op.attr('shape')},
            )
            delete_ops(startup_program.global_block(), table_param_init_op)

    def _apply_single_impl(self, main_program, startup_program, pass_ctx):
        attrs = pass_ctx._attrs
        sparse_tables = self._get_sparse_table_names(attrs)
        self._fake_init_sparsetable(startup_program, sparse_tables, attrs)


@register_pass("ps_gpu_pass")
class PsGpuPass(PassBase):
    def __init__(self):
        super().__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _add_push_box_sparse_op(self, program):
        insert_index = -1
        for idx, op in list(enumerate(program.global_block().ops)):
            if op.type == "lookup_table_grad":
                insert_index = idx
        for op in program.global_block().ops:
            if op.type != "pull_box_sparse" and op.type != "pull_gpups_sparse":
                continue
            grad_op_desc, op_grad_to_var = core.get_grad_op_desc(
                op.desc, set(), []
            )
            for op_desc in grad_op_desc:
                new_op_desc = program.global_block().desc._insert_op(
                    insert_index + 1
                )
                new_op_desc.copy_from(op_desc)
                new_op_desc._set_attr(op_role_attr_name, backward)
                new_op = paddle.static.Operator(
                    program.global_block(), new_op_desc
                )
                program.global_block().ops.insert(insert_index + 1, new_op)
                program.global_block()._sync_with_cpp()

    def _remove_optimizer_var(self, program):
        embedding_w = {}
        for idx, op in list(enumerate(program.global_block().ops)):
            if op.type == "lookup_table_grad":
                for name in op.input("W"):
                    embedding_w[name] = 1

        optimize_vars = []
        optimize_op_role_vars = []
        optimize_need_delete_vars = []
        for op in get_optimize_ops(program):
            # print("op=%s, input_names=%s" % (op, op.input_names))
            if "Param" not in op.input_names:
                continue
            for name in op.input("Param"):
                if name in embedding_w:
                    optimize_op_role_vars.extend(op.attr("op_role_var"))
                    for key_name in op.input_names:
                        if key_name == "LearningRate":
                            continue
                        for var in op.input(key_name):
                            optimize_vars.append(var)

        optimize_vars = list(set(optimize_vars))
        optimize_op_role_vars = list(set(optimize_op_role_vars))

        for var in optimize_vars:
            if var not in optimize_op_role_vars:
                optimize_need_delete_vars.append(var)
        need_delete_optimize_vars = list(set(optimize_need_delete_vars))

        for name in need_delete_optimize_vars:
            if program.global_block().has_var(name):
                program.global_block()._remove_var(name)

    def _remove_lookup_table_grad_op_and_var(self, program):
        lookup_table_grad_var = {}
        remove_op_index = []
        remove_var = []
        for idx, op in list(enumerate(program.global_block().ops)):
            if op.type == "lookup_table_grad":
                for name in op.output("W@GRAD"):
                    lookup_table_grad_var[name] = 1
                    remove_op_index.append(idx)
                    remove_var.append(name)
                for name in op.input("W"):
                    lookup_table_grad_var[name] = 1

        for idx, op in list(enumerate(program.global_block().ops)):
            if op.type == "pull_box_sparse" or op.type == "pull_gpups_sparse":
                continue
            for key_name in op.input_names:
                for var in op.input(key_name):
                    if var in lookup_table_grad_var:
                        remove_op_index.append(idx)
                        break

        remove_op_index = list(set(remove_op_index))
        remove_op_index.sort(reverse=True)
        for idx in remove_op_index:
            program.global_block()._remove_op(idx)
        for name in remove_var:
            program.global_block()._remove_var(name)

    def _apply_single_impl(self, main_program, startup_program, pass_ctx):
        attrs = pass_ctx._attrs
        self._add_push_box_sparse_op(main_program)
        self._remove_optimizer_var(main_program)
        self._remove_lookup_table_grad_op_and_var(main_program)


@register_pass("ps_transpile_pass")
class PsTranspilePass(PassBase):
    def __init__(self):
        super().__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, pass_ctx):
        if core._is_compiled_with_gpu_graph() is False:
            from ..transpiler.collective import MultiThread

            t = MultiThread()
            print("ps_transpile_pass use MultiThread for non_gpu_graph mode")
        else:
            from ..transpiler.collective import SingleProcessMultiThread

            t = SingleProcessMultiThread()
            print(
                "ps_transpile_pass use SingleProcessMultiThread for gpu_graph mode"
            )

        attrs = pass_ctx._attrs
        env = get_dist_env()
        t.transpile(
            startup_program=startup_program,
            main_program=main_program,
            rank=env["trainer_id"],
            endpoints=env["trainer_endpoints"],
            current_endpoint=env['current_endpoint'],
            wait_port=False,
        )


@register_pass("split_heter_worker_ops_pass")
class SplitHeterWorkerOpsPass(PassBase):
    def __init__(self):
        super().__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _create_heter_program(
        self,
        program,
        attrs,
        heter_program,
        program_block_ops_list,
        heter_ops,
        block_var_detail,
    ):
        # This function mainly includes the following contents:
        # 1. For every heter block:
        #     a) copy heter device op from origin program
        #     b) create variables which belong to heter op:
        #         -> if variable is persistable, clone it in global_scope
        #         -> if variable is temp, create it in heter block
        #     c) create communicate related op as follow:
        #         joint_var.0_1 -> slice -> reshape -> origin_var
        #         origin_var -> origin_program
        #         reshape -> concat -> joint_var.1_2
        #     d) copy send op from origin program for var@grad which located in current heter block
        #     e) re-check every op in current block if its device is not current heter device
        # 2. Create send op for step counter in last heter-block
        # 3. Create Listen&Serv OP and Send&Recv OP for distributed training
        # 4. update CompileTimeStrategy for heter_program

        optimizer_block = []
        grad_to_block_id = []
        send_grad_var_list = []

        pre_block_idx = heter_program.num_blocks - 1
        role_maker = attrs['role_maker']
        current_device = role_maker._heter_device_type().lower()
        stage_id = int(role_maker._get_stage_id())

        heter_block_ops_forward = program_block_ops_list[stage_id - 1][
            "forward"
        ]
        heter_block_ops_backward = program_block_ops_list[stage_id - 1][
            "backward"
        ]

        heter_block = heter_program._create_block(pre_block_idx)
        optimizer_block.append(heter_block)
        for _, op in enumerate(heter_block_ops_forward):
            block_append_op(heter_program, program, heter_block, op)

        entrance_vars = block_var_detail[stage_id - 1]["forward"]["entrance"]
        add_vars_by_var_list(entrance_vars, program, heter_program, heter_block)
        exit_vars = block_var_detail[stage_id - 1]["forward"]["exit"]
        add_vars_by_var_list(exit_vars, program, heter_program, heter_block)

        first_op_index_fp = len(heter_block.ops)

        if stage_id < len(program_block_ops_list):
            heter_block_bp = heter_program._create_block(pre_block_idx)
            optimizer_block.append(heter_block_bp)

            for _, op in enumerate(heter_block_ops_backward):
                block_append_op(heter_program, program, heter_block_bp, op)

            bp_entrance_vars = block_var_detail[stage_id - 1]["backward"][
                "entrance"
            ]
            add_vars_by_var_list(
                bp_entrance_vars, program, heter_program, heter_block_bp
            )
            bp_exit_vars = block_var_detail[stage_id - 1]["backward"]["exit"]
            add_vars_by_var_list(
                bp_exit_vars, program, heter_program, heter_block_bp
            )
            backward_comm_info = get_communicate_var_info(
                program, stage_id, bp_entrance_vars, type="backward"
            )

            grad_to_block_id.append(
                backward_comm_info["block_input_var_name"]
                + ":"
                + str(heter_block_bp.idx)
            )

        else:
            for _, op in enumerate(heter_block_ops_backward):
                block_append_op(heter_program, program, heter_block, op)

            bp_entrance_vars = block_var_detail[stage_id - 1]["backward"][
                "entrance"
            ]
            add_vars_by_var_list(
                bp_entrance_vars, program, heter_program, heter_block
            )
            bp_exit_vars = block_var_detail[stage_id - 1]["backward"]["exit"]
            add_vars_by_var_list(
                bp_exit_vars, program, heter_program, heter_block
            )

            heter_block_bp = heter_block

        forward_comm_info = get_communicate_var_info(
            program, stage_id, entrance_vars, type="forward"
        )

        grad_to_block_id.append(
            forward_comm_info["block_input_var_name"]
            + ":"
            + str(heter_block.idx)
        )

        first_op_index_bp = len(heter_block_bp.ops)

        if stage_id <= len(block_var_detail) - 1:
            static_var = insert_communicate_op(
                program,
                role_maker,
                heter_block,
                stage_id,
                first_op_index_fp,
                block_var_detail,
                current_device,
            )
        static_var_bp = insert_communicate_op(
            program,
            role_maker,
            heter_block_bp,
            stage_id,
            first_op_index_bp,
            block_var_detail,
            current_device,
            False,
        )

        # add send op
        send_grad_var_list = add_send_op(
            program,
            heter_block_bp,
            block_var_detail[stage_id - 1]["backward"]["persistables"],
        )

        # add step conter
        send_input_vars = []
        dummy_output = []
        pserver_endpoints = get_ps_endpoints(role_maker)
        attrs = {
            "message_to_block_id": grad_to_block_id,
            "optimize_blocks": optimizer_block,
            # runtime attribute
            "endpoint": get_heter_worker_endpoint(role_maker),
            "fanin": len(get_previous_stage_trainers(role_maker)),
            "pserver_id": get_role_id(role_maker),
            "distributed_mode": attrs['ps_mode'],
            "rpc_exec_thread_num": int(os.getenv("CPU_NUM", 32)),
            RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE,
        }
        # append the listen_and_serv op
        heter_program.global_block().append_op(
            type="heter_listen_and_serv",
            inputs={'X': []},
            outputs={},
            attrs=attrs,
        )
        # TODO check heter program

    def _apply_single_impl(self, main_program, startup_program, pass_ctx):
        """
        split heter worker program from origin-program
        1. find heter op (located on different device)
        2. find input&output of every heter-block
        3. create heter worker program, add listen&serv op
        """
        attrs = pass_ctx._attrs
        default_device = "cpu"
        program, heter_ops, _, program_block_ops = find_heter_ops(
            main_program, default_device
        )
        if len(heter_ops) == 0:
            warnings.warn(
                "Currently running in Heter Parameter Server mode, but no OP running on heterogeneous devices, Please check your code."
            )
            main_program = program
            return

        program_block_ops = union_forward_gradient_op(program_block_ops)
        block_vars_detail = find_block_joints(
            program, program_block_ops, heter_ops
        )
        heter_program = paddle.framework.Program()
        self._create_heter_program(
            program,
            attrs,
            heter_program,
            program_block_ops,
            heter_ops,
            block_vars_detail,
        )
        main_program = heter_program


@register_pass("split_trainer_ops_pass")
class SplitTrainerOpsPass(PassBase):
    def __init__(self):
        super().__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _replace_ops_by_communicate_op(
        self, program, attrs, heter_block_index, ops_list, block_var_detail
    ):
        all_op = program.global_block().ops
        start_op = ops_list[0]
        first_op_idx = -1
        for op in all_op:
            if str(op) == str(start_op):
                first_op_idx = all_op.index(op)
                break
        assert first_op_idx != -1
        delete_same_ops(program.global_block(), ops_list)

        entrance_var = []
        role_maker = attrs['role_maker']
        if heter_block_index == 1:
            next_heter_worker_endpoints = get_next_stage_trainers(role_maker)

            entrance_var = block_var_detail[heter_block_index]["forward"][
                "entrance"
            ]

            comm_info = get_communicate_var_info(
                program, heter_block_index + 1, entrance_var
            )
            program.global_block()._insert_op(
                index=first_op_idx,
                type="send_and_recv",
                inputs={"X": program.global_block().vars[entrance_var[0]]},
                outputs={"Out": []},
                attrs={
                    "mode": "forward",
                    "send_var_name": entrance_var + ["microbatch_id"],
                    "recv_var_name": [],
                    "message_name": comm_info["block_input_var_name"],
                    "next_endpoints": next_heter_worker_endpoints,
                    "previous_endpoints": [],
                    "trainer_id": get_role_id(role_maker),
                    RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE,
                },
            )

        return entrance_var

    def _remove_var_pair_by_grad(self, var_name, attrs):
        for index, pair in enumerate(attrs['merged_variables_pairs']):
            var = pair[0]
            var_grad = pair[1]
            if var_grad.merged_var.name == var_name:
                del attrs['merged_variables_pairs'][index]

        for index, pair in enumerate(attrs['merged_dense_pairs']):
            var = pair[0]
            var_grad = pair[1]
            if var_grad.merged_var.name == var_name:
                del attrs['merged_dense_pairs'][index]
                return

        for index, pair in enumerate(attrs['merged_sparse_pairs']):
            var = pair[0]
            var_grad = pair[1]
            if var_grad.merged_var.name == var_name:
                del attrs['merged_sparse_pairs'][index]
                return

    def _remove_trainer_send_op(
        self, program, attrs, heter_block_index, block_var_detail
    ):
        # if trainer do FF->BP->SEND, it has follow vars: var, var@GRAD
        # if trainer only do SEND, it has one var: var@GRAD
        # Delete Send op ,if trainer doesn't has pair var (var<->var@GRAD)
        persistables = (
            block_var_detail[heter_block_index]["forward"]["persistables"]
            + block_var_detail[heter_block_index]["backward"]["persistables"]
        )
        need_remove_send_op = []
        need_remove_grad_var = []
        for op in find_send_op(program):
            input_list, _ = find_op_input_output(
                program, program.global_block(), op
            )
            for var_name in input_list:
                origin_var_name = var_name.split("@GRAD")[0]
                if origin_var_name in persistables:
                    need_remove_send_op.append(op)
                    need_remove_grad_var.append(var_name)
        need_remove_send_op = list(set(need_remove_send_op))
        delete_ops(program.global_block(), need_remove_send_op)
        for grad_var_name in need_remove_grad_var:
            self._remove_var_pair_by_grad(grad_var_name, attrs)

    def _create_trainer_program(
        self,
        program,
        origin_program,
        attrs,
        program_block_ops_list,
        block_var_detail,
    ):
        # This function mainly includes the following contents:
        # 1. For every heter block in origin program
        #     a) delete heter op and related variables
        #     b) add send&recv op
        #     c) add communicate ops as follows:
        #         origin_var -> reshape -> concat -> joint_var.0_1
        #         send&recv op(send joint_var.0_1; recv joint_var.1_2)
        #         joint_var.1_2 -> slice -> reshape -> origin_var
        #     d) remove send op which related var@grad is not in trainer program
        # 2. check every op's device
        static_var = []
        for heter_block_index in range(1, len(program_block_ops_list)):
            ops_list = (
                program_block_ops_list[heter_block_index]["forward"]
                + program_block_ops_list[heter_block_index]["backward"]
            )
            static_var += self._replace_ops_by_communicate_op(
                program, attrs, heter_block_index, ops_list, block_var_detail
            )
            self._remove_trainer_send_op(
                program, attrs, heter_block_index, block_var_detail
            )

        optimizer_block = []
        grad_to_block_id = []

        bp_ops_list = program_block_ops_list[0]["backward"]
        delete_same_ops(program.global_block(), bp_ops_list)
        delete_trainer_useless_var(program, static_var)
        backward_block = create_backward_block(
            program, origin_program, bp_ops_list, block_var_detail
        )

        bp_entrance_vars = block_var_detail[0]["backward"]["entrance"]
        backward_comm_info = get_communicate_var_info(
            origin_program, 1, bp_entrance_vars, type="backward"
        )

        grad_to_block_id.append(
            backward_comm_info["block_input_var_name"]
            + ":"
            + str(backward_block.idx)
        )
        optimizer_block.append(backward_block)
        role_maker = attrs['role_maker']
        attrs = {
            "message_to_block_id": grad_to_block_id,
            "optimize_blocks": optimizer_block,
            # runtime attribute
            "endpoint": get_trainer_endpoint(
                role_maker
            ),  # get trainer endpoint
            "fanin": 0,  # get heter worker
            "pserver_id": get_role_id(role_maker),
            "distributed_mode": attrs['ps_mode'],
            "rpc_exec_thread_num": int(os.getenv("CPU_NUM", 32)),
            RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE,
        }
        # append the listen_and_serv op
        program.global_block()._insert_op(
            index=0,
            type="heter_listen_and_serv",
            inputs={'X': []},
            outputs={},
            attrs=attrs,
        )

        # TODO add check for bp block
        # check_op_device(program.global_block(), DEFAULT_DEVICE)

    def _apply_single_impl(self, main_program, startup_program, pass_ctx):
        """
        split cpu-trainer program from origin-program
        1. find heter op (located on different device)
        2. find input&output of every heter-block
        3. create cpu-trainer program, add send&recv op
        """
        attrs = pass_ctx._attrs
        default_device_ = 'cpu'
        program, heter_ops, default_ops, program_block_ops = find_heter_ops(
            main_program, default_device_
        )
        program_block_ops = union_forward_gradient_op(program_block_ops)

        block_vars_detail = find_block_joints(
            program, program_block_ops, heter_ops
        )
        trainer_program = program.clone()
        self._create_trainer_program(
            trainer_program,
            program,
            attrs,
            program_block_ops,
            block_vars_detail,
        )
        main_program = trainer_program


@register_pass("set_heter_pipeline_opt_pass")
class SetHeterPipelineOptPass(PassBase):
    def __init__(self):
        super().__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, pass_ctx):
        attrs = pass_ctx._attrs
        role_maker = attrs['role_maker']
        num_microbatches = attrs['user_defined_strategy'].pipeline_configs[
            'accumulate_steps'
        ]

        startup_program._heter_pipeline_opt = {
            "startup_program": startup_program,
            "pipeline_stage": int(role_maker._get_stage_id()) - 1,
            "heter_place": role_maker._heter_device(),
            "is_fl_mode": 1,
        }
        main_program._heter_pipeline_opt = {
            "trainer": "HeterPipelineTrainer",
            "device_worker": "HeterSection",
            "trainers": role_maker._get_stage_trainers(),  # trainer num in each stage
            "trainer_id": int(role_maker._role_id()),
            "pipeline_stage": int(role_maker._get_stage_id()) - 1,
            "num_pipeline_stages": int(role_maker._get_num_stage()),
            "section_program": main_program,
            "num_microbatches": num_microbatches,
            "heter_place": role_maker._heter_device(),
            "is_fl_mode": 1,
        }


@register_pass("split_fl_ops_pass")
class SplitFlOpsPass(PassBase):
    def __init__(self):
        super().__init__()
        self.PART_A_DEVICE_FlAG = 'gpu:0'
        self.PART_A_JOINT_OP_DEVICE_FlAG = 'gpu:2'
        self.PART_B_DEVICE_FlAG = 'gpu:1'
        self.PART_B_JOINT_OP_DEVICE_FlAG = 'gpu:3'

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _insert_encrypt_op(self):
        pass

    def _insert_decrypt_op(self):
        pass

    def _clear_op_device_flag(self, program):
        for block in program.blocks:
            for op in block.ops:
                device = op.attr(OP_DEVICE_KEY)
                op._set_attr(OP_DEVICE_KEY, '') if device != '' else None

    def _split_fl_program(self):
        self.partA_ops = []
        self.partB_ops = []
        party_program_map = defaultdict(Program)
        block = self.ori_main_program.block(0)
        for op in block.ops:
            device = op.attr(OP_DEVICE_KEY)
            if (
                device == self.PART_A_DEVICE_FlAG
                or device == ''
                or device == self.PART_A_JOINT_OP_DEVICE_FlAG
            ):
                program = party_program_map['a']
                self.partA_ops.append(op)
            elif (
                device == self.PART_B_DEVICE_FlAG
                or device == self.PART_B_JOINT_OP_DEVICE_FlAG
            ):
                program = party_program_map['b']
                self.partB_ops.append(op)
            op_desc = op.desc
            ap_op = program.global_block().desc.append_op()
            ap_op.copy_from(op_desc)
            ap_op._set_attr(OP_DEVICE_KEY, device)

        for key in ['a', 'b']:
            program = party_program_map[key]
            program._sync_with_cpp()

        return party_program_map

    def _insert_partA_communicate_op(self, block, idx):
        comm_info = f"forward_joint_{1}_{2}@fl_ps"
        block._insert_op(
            idx,
            type='send_and_recv',
            inputs={'X': self.partA_to_partB_tensor},
            outputs={'Out': []},
            attrs={
                'mode': 'forward',  # mode 直接关联前向和反向 channel 选择
                'send_var_name': self.partA_to_partB_tensor_name
                + ["microbatch_id"],
                'recv_var_name': [],
                'message_name': comm_info,
                'next_endpoints': get_next_stage_trainers(
                    self.role_maker
                ),  # partB_endpoints
                'previous_endpoints': get_previous_stage_trainers(
                    self.role_maker
                ),
                'trainer_id': get_role_id(self.role_maker),  # global id
                RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE,
            },
        )

    def _insert_partB_communicate_op(self, block, idx):
        comm_info = f"backward_joint_{2}_{1}@fl_ps"
        block._insert_op(
            idx,
            type='send_and_recv',
            inputs={'X': self.partB_to_partA_grad},
            outputs={'Out': []},
            attrs={
                'mode': 'backward',
                'send_var_name': self.partB_to_partA_grad_name
                + ["microbatch_id"],
                'recv_var_name': [],
                'message_name': comm_info,
                'next_endpoints': get_next_stage_trainers(
                    self.role_maker
                ),  # partA_endpoints
                'previous_endpoints': get_previous_stage_trainers(
                    self.role_maker
                ),
                'trainer_id': get_role_id(self.role_maker),  # global id
                RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE,
            },
        )

    def _create_var_for_block(self, vars, block):
        for var in vars:
            if block._find_var_recursive(str(var)):
                continue
            source_var = self.ori_main_block._var_recursive(str(var))
            if isinstance(var, Parameter):
                dest_var = block.create_parameter(
                    name=source_var.name,
                    shape=source_var.shape,
                    dtype=source_var.dtype,
                    type=source_var.type,
                    lod_level=source_var.lod_level,
                    stop_gradient=source_var.stop_gradient,
                    trainable=source_var.trainable,
                    optimize_attr=source_var.optimize_attr,
                    regularizer=source_var.regularizer,
                    error_clip=source_var.error_clip,
                )
            else:
                dest_var = block._clone_variable(source_var, False)
            dest_var.stop_gradient = source_var.stop_gradient
            if hasattr(source_var, 'is_distributed'):
                dest_var.is_distributed = source_var.is_distributed

    def _get_block_by_idx(self, op_list, program, block_idx):
        if block_idx < len(program.blocks):
            new_block = program.block(block_idx)
        else:
            new_block = program._create_block()
        for _, op in enumerate(op_list):
            ap_op = new_block.desc.append_op()
            ap_op.copy_from(op.desc)
            ap_op._set_attr(OP_DEVICE_KEY, op.attr(OP_DEVICE_KEY))
            vars = op.desc.input_arg_names() + op.desc.output_arg_names()
            self._create_var_for_block(vars, new_block)
        new_block._sync_with_cpp()
        return new_block

    def _find_joint_forward_op(self, block, flag):
        op_idx = 0
        for op in block.ops:
            if is_forward_op(op) and op.attr(OP_DEVICE_KEY) == flag:
                return op_idx
            else:
                op_idx += 1
        return op_idx

    def _find_joint_backward_op(self, block, flag):
        op_idx = 0
        for op in block.ops:
            if is_backward_op(op) and op.attr(OP_DEVICE_KEY) == flag:
                return op_idx
            else:
                op_idx += 1
        return op_idx

    def _get_partB_to_partA_grad(self, block, flag):
        op_idx = self._find_joint_backward_op(block, flag)
        op = block.ops[op_idx]
        vars1 = op.desc.input_arg_names()
        op_idx = self._find_joint_forward_op(block, flag)
        op = block.ops[op_idx]
        vars2 = op.desc.output_arg_names()
        self.partB_to_partA_grad_name = list(set(vars1) - set(vars2))
        self.partB_to_partA_grad = []
        for var_name in self.partB_to_partA_grad_name:
            self.partB_to_partA_grad.append(self.ori_main_block.var(var_name))

    def _find_dense_grad_vars(self, bp_op_list):
        program = self.ori_main_program
        bp_op_input, bp_op_output = find_ops_list_input_output(
            program, bp_op_list
        )
        return screen_persistables(program, bp_op_input) + screen_persistables(
            program, bp_op_output
        )

    def _get_partA_program(self, block):
        # 1. create block 0
        # 1.1 insert send op
        op_idx = self._find_joint_forward_op(
            block, self.PART_A_JOINT_OP_DEVICE_FlAG
        )
        op_list = []
        for i in range(len(block.ops)):
            op = block.ops[i]
            op_list.append(op)
            if i == op_idx:
                out_name = op.desc.output_arg_names()[0]
                self.partA_to_partB_tensor_name = op.desc.output_arg_names()
                self.partA_to_partB_tensor = self.ori_main_block.var(out_name)
                break
        first_block = self._get_block_by_idx(op_list, self.partA_program, 0)
        self._insert_partA_communicate_op(first_block, op_idx + 1)
        # logger.info('partA-first_block:{}'.format(first_block))

        # 2. create block 1
        bp_op_list = get_bp_op_list(block)
        push_sparse_op_list = get_distributed_push_sparse_op_list(block)
        # logger.info('bp_op_list: {}'.format(bp_op_list))
        second_block = self._get_block_by_idx(
            bp_op_list + push_sparse_op_list, self.partA_program, 1
        )
        # 2.1. insert partA recv op
        block_input_flag = f"backward_joint_{2}_{1}@fl_ps"
        grad_to_block_id = block_input_flag + ":" + str(second_block.idx)
        attrs = {
            "message_to_block_id": [grad_to_block_id],
            "optimize_blocks": [second_block],
            "endpoint": get_trainer_endpoint(self.role_maker),
            "fanin": 0,
            "pserver_id": get_role_id(self.role_maker),
            "distributed_mode": self.ps_mode,
            "rpc_exec_thread_num": int(os.getenv("CPU_NUM", 32)),
            RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE,
        }
        second_block._insert_op(
            index=0,
            type='heter_listen_and_serv',
            inputs={'X': []},
            outputs={},
            attrs=attrs,
        )
        # 2.2 insert push dense grad op
        send_ops = find_send_op(self.ori_main_program)  # push dense
        delete_same_ops(block, send_ops)
        dense_grad_vars = self._find_dense_grad_vars(bp_op_list)
        add_send_op(self.ori_main_program, second_block, dense_grad_vars)
        # logger.info('partA-second_block:{}'.format(second_block))

    def _get_partB_program(self, block):
        op_idx1 = self._find_joint_forward_op(
            block, self.PART_B_JOINT_OP_DEVICE_FlAG
        )  # elementwise_add op
        op_idx2 = self._find_joint_backward_op(
            block, self.PART_B_JOINT_OP_DEVICE_FlAG
        )
        op_cnt = 0
        op_list1 = []
        op_list2 = []
        op_list3 = []
        for op in block.ops:
            if op_cnt < op_idx1:
                op_list1.append(op)
            elif op_cnt <= op_idx2:
                op_list2.append(op)
            else:
                op_list3.append(op)
            op_cnt += 1

        # 1. create block 0
        first_block = self._get_block_by_idx(op_list1, self.partB_program, 0)

        # 2. create block 1
        second_block = self._get_block_by_idx(op_list2, self.partB_program, 1)
        # 2.1 insert send op
        self._insert_partB_communicate_op(second_block, len(op_list2))
        # 2.2 insert remain ops
        second_block = self._get_block_by_idx(op_list3, self.partB_program, 1)
        # 2.3 insert push dense grad op
        bp_op_list = get_bp_op_list(second_block)
        dense_grad_vars = self._find_dense_grad_vars(bp_op_list)
        add_send_op(self.ori_main_program, second_block, dense_grad_vars)

        # 3. insert partB recv op
        block_input_flag = f"forward_joint_{1}_{2}@fl_ps"
        grad_to_block_id = block_input_flag + ":" + str(second_block.idx)
        attrs = {
            "message_to_block_id": [grad_to_block_id],
            "optimize_blocks": [second_block],  # what to do?
            "endpoint": get_heter_worker_endpoint(self.role_maker),
            "fanin": len(get_previous_stage_trainers(self.role_maker)),
            "pserver_id": 1,  # TODO
            "distributed_mode": self.ps_mode,
            "rpc_exec_thread_num": int(os.getenv("CPU_NUM", 32)),
            RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE,
        }
        first_block._insert_op(
            index=len(op_list1),
            type="heter_listen_and_serv",
            inputs={'X': []},
            outputs={},
            attrs=attrs,
        )

        # logger.info('partB-first_block:{}'.format(first_block))
        # logger.info('partB-second_block:{}'.format(second_block))

    def _apply_single_impl(self, main_program, startup_program, pass_ctx):
        attrs = pass_ctx._attrs
        self.role_maker = attrs['role_maker']
        self.ps_mode = attrs['ps_mode']
        self.is_part_b = attrs['is_heter_worker']  # TODO
        self.ori_main_program = main_program
        self.ori_main_block = main_program.block(0)

        party_program_map = self._split_fl_program()

        prog_a = party_program_map['a']
        _main_file = ps_log_root_dir + '6_fl_A_main_program.prototxt'
        debug_program(_main_file, prog_a)
        self._get_partB_to_partA_grad(
            prog_a.global_block(), self.PART_A_JOINT_OP_DEVICE_FlAG
        )

        prog_b = party_program_map['b']
        _main_file = ps_log_root_dir + '6_fl_B_main_program.prototxt'
        debug_program(_main_file, prog_b)

        if not self.is_part_b:
            self.partA_program = paddle.framework.Program()
            self._get_partA_program(prog_a.global_block())
            pass_ctx._attrs['part_a_main_program'] = self.partA_program
            self._clear_op_device_flag(self.partA_program)
            check_program(self.partA_program)
        else:
            self.partB_program = paddle.framework.Program()
            self._get_partB_program(prog_b.global_block())
            pass_ctx._attrs['part_b_main_program'] = self.partB_program
            self._clear_op_device_flag(self.partB_program)
            check_program(self.partB_program)
