#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Parameter Server utils"""

import numpy as np


class Distributed:
    @staticmethod
    def auc(stat_pos, stat_neg, scope, util):
        stat_pos = np.array(scope.find_var(stat_pos.name).get_tensor())
        stat_neg = np.array(scope.find_var(stat_neg.name).get_tensor())

        # auc pos bucket shape
        old_pos_shape = np.array(stat_pos.shape)
        # reshape to one dim
        stat_pos = stat_pos.reshape(-1)
        global_pos = np.copy(stat_pos) * 0
        # reshape to its original shape

        global_pos = util.all_reduce(stat_pos, "sum")
        global_pos = global_pos.reshape(old_pos_shape)

        # auc neg bucket
        old_neg_shape = np.array(stat_neg.shape)
        stat_neg = stat_neg.reshape(-1)
        global_neg = np.copy(stat_neg) * 0

        global_neg = util.all_reduce(stat_neg, "sum")
        global_neg = global_neg.reshape(old_neg_shape)

        # calculate auc
        num_bucket = len(global_pos[0])
        area = 0.0
        pos = 0.0
        neg = 0.0
        new_pos = 0.0
        new_neg = 0.0
        total_ins_num = 0
        for i in range(num_bucket):
            index = num_bucket - 1 - i
            new_pos = pos + global_pos[0][index]
            total_ins_num += global_pos[0][index]
            new_neg = neg + global_neg[0][index]
            total_ins_num += global_neg[0][index]
            area += (new_neg - neg) * (pos + new_pos) / 2
            pos = new_pos
            neg = new_neg

        auc_value = None
        if pos * neg == 0 or total_ins_num == 0:
            auc_value = 0.5
        else:
            auc_value = area / (pos * neg)

        return auc_value

    @staticmethod
    def estimate(main_program, varname2tables):
        def distributed_ops_pass(program):
            SPARSE_OP_TYPE_DICT = {"lookup_table": "W", "lookup_table_v2": "W"}

            def _get_pull_sparse_ops(_program):
                pull_sparse_ops = {}
                for op in _program.global_block().ops:
                    if op.type in SPARSE_OP_TYPE_DICT.keys() \
                            and op.attr('remote_prefetch') is True:
                        param_name = op.input(SPARSE_OP_TYPE_DICT[op.type])[0]
                        ops = pull_sparse_ops.get(param_name, [])
                        ops.append(op)
                        pull_sparse_ops[param_name] = ops
                return pull_sparse_ops

            def _pull_sparse_fuse(_program, pull_sparse_ops):
                for param, ops in pull_sparse_ops.items():
                    all_ops = program.global_block().ops
                    op_idxs = [all_ops.index(op) for op in ops]

                    inputs = [
                        program.global_block().vars[op.input("Ids")[0]]
                        for op in ops
                    ]

                    w = program.global_block().vars[ops[0].input("W")[0]]

                    if w.name not in varname2tables.keys():
                        raise ValueError(
                            "can not find variable {}, please check your configuration".
                            format(w.name))

                    table_id = varname2tables[w.name]

                    padding_idx = ops[0].attr("padding_idx")
                    is_distributed = ops[0].attr("is_distributed")
                    op_type = ops[0].type

                    outputs = [
                        program.global_block().vars[op.output("Out")[0]]
                        for op in ops
                    ]

                    for idx in op_idxs[::-1]:
                        program.global_block()._remove_op(idx)

                    inputs_idxs = [-1] * len(inputs)
                    outputs_idxs = [-1] * len(outputs)

                    for idx, op in enumerate(program.global_block().ops):
                        for i in range(0, len(op.output_names)):
                            outs = op.output(op.output_names[i])
                            for in_id, in_var in enumerate(inputs):
                                if in_var.name in outs:
                                    inputs_idxs[in_id] = idx
                        for i in range(0, len(op.input_names)):
                            ins = op.input(op.input_names[i])
                            for out_id, out_var in enumerate(outputs):
                                if out_var.name in ins:
                                    outputs_idxs[out_id] = idx

                    if min(outputs_idxs) - max(inputs_idxs) >= 1:
                        distributed_idx = max(inputs_idxs) + 1

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
                                "lookup_table_version": op_type
                            })
                    else:
                        raise ValueError(
                            "something wrong with Fleet, submit a issue is recommended"
                        )

            pull_sparse_ops = _get_pull_sparse_ops(program)
            _pull_sparse_fuse(program, pull_sparse_ops)
            return program

        covert_program = distributed_ops_pass(main_program)
        return covert_program
