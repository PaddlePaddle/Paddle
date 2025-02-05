# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import logging

import paddle
from paddle.base import core

OpRole = core.op_proto_and_checker_maker.OpRole

from paddle.autograd import backward_utils

from ..auto_parallel.static.utils import (
    get_logger,
)
from .pass_base import PassBase, register_pass

logger = get_logger(logging.INFO)


@register_pass("auto_parallel_recompute_pir")
class AutoParallelRecomputePIRPass(PassBase):
    def __init__(self):
        super().__init__()
        self.program_ops = []

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def get_fwd_bwd_ops(self):
        fwd_ops = []
        bwd_ops = []
        for op in self.program_ops:
            if op.op_role == int(OpRole.Forward):
                fwd_ops.append(op)
            elif op.op_role == int(OpRole.Backward):
                bwd_ops.append(op)
        assert len(fwd_ops) and len(bwd_ops)
        return fwd_ops, bwd_ops

    def get_first_bwd_used_op(self, fwd_op, bwd_ops):
        # Find the first user op of the op result in backward op list.
        first_op = bwd_ops[-1]
        for res in fwd_op.results():
            for user_op in res.all_used_ops():
                if user_op in bwd_ops and first_op.id() >= user_op.id():
                    first_op = user_op
        return first_op

    def is_seed_used_by_dropout(self, seed_op):
        # Ensure that the random operator has the same output in backward recompute.
        if seed_op.name() != "seed":
            return False
        seed_value = seed_op.results()[0]
        dropout_ops = ["pd_op.dropout", "pd_op.fused_dropout_add"]
        return any(
            True
            for used_op in seed_value.all_used_ops()
            if used_op.name() in dropout_ops
        )

    def remove_outgoing_op(self, segment):
        # An OP is considered an outgoing OP if all of results' user OPs are not in segment.
        # These OPs do not participate in the backward gradient computation and therefore
        # do not need to have a recomputation during backward.
        segment_ops = [self.program_ops[idx] for idx in segment]
        segment_len = len(segment)
        for idx in range(segment_len - 1, 0, -1):
            op = segment_ops[idx]
            user_ops = set()
            for res in op.results():
                user_ops = user_ops | set(res.all_used_ops())

            if user_ops & set(segment_ops):
                continue
            segment.pop(idx)
            logger.info(
                f"Remove outgoing OP '{op.name()}' from the segment for recomputation, as it does not participate in the backward."
            )
        return segment

    def get_segments(self):
        # `fwd_recompute_id` indicates the ID assigned to the segment for
        # which the OP requires recompute.
        # A segment comprises all OPs within a program, ranging from the OP
        # with the minimum index to the OP with the maximum index, and all
        # these operations share the same `fwd_recompute_id`.
        segment_beg = {}
        segment_end = {}
        max_op_id = len(self.program_ops)
        for idx, op in enumerate(self.program_ops):
            # 1. Find the OPs marked with `fwd_recompute_id`.
            if not op.has_attr("fwd_recompute_id"):
                continue
            # 2. Delineate the segment range marked by `fwd_recompute_id`.
            # Note: there may be some unmarked OPs in between.
            rc_id = op.attrs()["fwd_recompute_id"]
            if rc_id not in segment_beg:
                segment_beg[rc_id] = max_op_id
                segment_end[rc_id] = 0
            segment_beg[rc_id] = min(segment_beg[rc_id], idx)
            segment_end[rc_id] = max(segment_end[rc_id], idx)

        # 3. Aggregate all segment information into a dictionary.
        # The key is the id of the segment, which is used to uniquely identify each segment.
        # The value is a list of indices of the segment OPs in `self.program_ops`.
        segments = {}
        assert len(segment_beg.keys()) == len(segment_end.keys())
        for segment_id, beg_id in segment_beg.items():
            assert segment_id in segment_end.keys()
            end_id = segment_end[segment_id]
            assert beg_id <= end_id
            segment = list(range(beg_id, end_id + 1))
            # 4. Remove the outgoing OPs from the segment, as these OPs
            # do not participate in the backward gradient computation.
            segments[segment_id] = self.remove_outgoing_op(segment)
            logger.info(
                f"Segment ID {segment_id} contains {len(segment)} OPs, all of which will be recomputed."
            )
        return segments

    def get_op_name(self, op):
        return op.name().split('.')[1]

    def match_pattern(
        self,
        op,
        visit,
        fetch_id,
        fetch_pattern,
        target_pattern,
        pre_len,
        main_len,
        count,
        max_count,
    ):
        if count >= max_count:
            return max_count
        if len(fetch_pattern) > len(target_pattern):
            return count
        if self.get_op_name(op) != target_pattern[fetch_id]:
            return count
        if fetch_id == len(target_pattern) - 1:
            for idx in range(pre_len, pre_len + main_len):
                fetch_op = fetch_pattern[idx]
                visit[fetch_op] = -1
            refined_segment = list(set(visit.values()))
            refined_segment.sort()
            refined_segment = [idx for idx in refined_segment if idx != -1]
            return count + 1
        for res_val in op.results():
            for user_op in res_val.all_used_ops():
                fetch_pattern[fetch_id + 1] = user_op
                count = self.match_pattern(
                    op=user_op,
                    visit=visit,
                    fetch_id=fetch_id + 1,
                    fetch_pattern=fetch_pattern,
                    target_pattern=target_pattern,
                    pre_len=pre_len,
                    main_len=main_len,
                    count=count,
                    max_count=max_count,
                )
        return count

    def _apply_single_impl(self, main_program, startup_program, context=None):
        self.program_ops = list(main_program.global_block().ops)
        # 1. Get the recompute segments information form program.
        segments = self.get_segments()
        assert (
            len(segments) > 0
        ), "No segment found in the PIR recompute pass.\n \
            Please disable 'recompute.enable' or check 'recompute()' usage in model code."

        # 2. Get the forward and backward OPs from program.
        fwd_ops, bwd_ops = self.get_fwd_bwd_ops()

        # 3. Refine the segments based on the patterns.
        refined_ops_patterns = self.get_attr("refined_ops_patterns")
        for refined_ops_pattern in refined_ops_patterns:
            # 3.1 get the refined pattern information.
            # refined_ops_patterns = pre_ops + main_ops + suf_ops
            # `main_ops` pattern: it does not participate in backward recomputation
            #                     and needs to be removed from the segment.
            # `pre_ops` pattern: it serve only as markers and do require recomputation.
            # `suf_ops` pattern: it serve only as markers and do require recomputation.
            # `num` : it limits the maximum number of `main_ops` patterns identified
            #         within each segment. A value of -1 represents all patterns.
            num = int(refined_ops_pattern['num'])
            num = num if num >= 0 else len(fwd_ops)
            main_ops = refined_ops_pattern['main_ops']
            pre_ops = refined_ops_pattern['pre_ops']
            suf_ops = refined_ops_pattern['suf_ops']
            pattern_ops = pre_ops + main_ops + suf_ops

            for rc_id in segments.keys():
                # 3.2 Identify and mark the first 'num' patterns in each segment.
                # The dictionary 'op_idx_map' has keys as OP information.
                # If an OP belongs to a pattern, its value in the dictionary is marked as -1.
                op_idx_map = {
                    self.program_ops[idx]: idx for idx in segments[rc_id]
                }
                pattern_count = 0
                fetch_pattern = [None] * len(pattern_ops)
                for idx in segments[rc_id]:
                    op = self.program_ops[idx]
                    fetch_pattern[0] = op
                    pattern_count = self.match_pattern(
                        op=self.program_ops[idx],
                        visit=op_idx_map,
                        fetch_id=0,
                        fetch_pattern=fetch_pattern,
                        target_pattern=pattern_ops,
                        pre_len=len(pre_ops),
                        main_len=len(main_ops),
                        count=pattern_count,
                        max_count=num,
                    )
                # 3.3 Refined segment to exclude the specified pattern.
                refined_segment = list(set(op_idx_map.values()))
                refined_segment.sort()
                refined_segment = [idx for idx in refined_segment if idx != -1]
                segments[rc_id] = refined_segment

        # 4. Construct the segment for backward recomputation.
        # 4.1 Build IrMapping to eplace forward value with backward recompute value.
        input_value = main_program.list_vars()
        value_map = paddle.pir.IrMapping()
        for val in input_value:
            value_map.add(val, val)

        for rc_id, segment in segments.items():
            # 4.2 Find the insertion position for the backward segment,
            # which should be before backward gradient computation.
            first_bwd_used_op = bwd_ops[-1]
            for idx in segment:
                op = self.program_ops[idx]
                bwd_used_op = self.get_first_bwd_used_op(op, bwd_ops)
                if first_bwd_used_op.id() > bwd_used_op.id():
                    first_bwd_used_op = bwd_used_op

            ori_segment_outputs = backward_utils.ValueSet()
            paddle.pir.set_insertion_point(first_bwd_used_op)

            # 4.3 Clone the segment OPs and replace the forward
            # value with backward recompute value.
            for idx in segment:
                op = self.program_ops[idx]
                ori_segment_outputs.update(op.results())

                # Random OPs should produce the same output before and after recomputation.
                if self.is_seed_used_by_dropout(op):
                    continue

                rc_op = op.clone(
                    value_map, paddle.pir.CloneOptions(False, True, True)
                )
                # The forward segment and the backward segment have the same segment ID.
                if rc_op.has_attr("fwd_recompute_id"):
                    rc_op.erase_attr("fwd_recompute_id")

                rc_op.set_int_attr("bwd_recompute_id", rc_id)

                # Updtate attributes.
                if first_bwd_used_op.has_attr('op_role'):
                    rc_op.set_int_attr("op_role", first_bwd_used_op.op_role)

                if first_bwd_used_op.has_attr('chunk_id'):
                    rc_op.set_int_attr("chunk_id", first_bwd_used_op.chunk_id)

            # 4.4 Replace the forward value with backward recompute value.
            for ori_value in ori_segment_outputs:
                rc_value = value_map.look_up(ori_value)
                ori_value.replace_grad_users_with(rc_value, set(bwd_ops))
