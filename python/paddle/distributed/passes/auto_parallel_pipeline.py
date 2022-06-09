# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from .pass_base import PassBase, register_pass
from paddle.distributed.fleet.meta_optimizers.common import OpRole, OP_ROLE_KEY
from paddle.distributed.fleet.fleet_executor_utils import TaskNode, is_forward_op, is_backward_op, is_optimizer_op, is_lr_sched_op


@register_pass("auto_parallel_pipeline")
class PipelinePass(PassBase):

    def __init__(self):
        super(PipelinePass, self).__init__()
        self.set_attr("dist_context", None)

    def _check_self(self):
        if self.get_attr("dist_context") is None:
            return False
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        self._dist_context = self.get_attr("dist_context")
        self._acc_steps = self.get_attr("accumulate_steps")
        self._mode = self.get_attr("schedule_mode")
        self._program = main_program

        self._insert_sync_op()

        if self._mode == "1F1B":
            self._task_1f1b()
        elif self._mode == "F-Then-B":
            raise NotImplementedError("F-Then-B has not been implemented")
        else:
            raise ValueError("Now only 'F-then-B' and '1F1B' are supported."
                             "The given value is {}.".format(self._mode))

    def _insert_sync_op(self):
        """
        This implementation refers to lots of Paddle/python/paddle/fluid/optimizer.py. 
        The difference between this function with 'PipelineOptimizer' is that 
        'send_v2' op and 'recv_v2' op have been inserted in program by 'reshard'.
        """

        for block in self._program.blocks:
            offset = 0
            first_optimize_index = None
            for index, op in enumerate(list(block.ops)):
                op_role = int(op.all_attrs()[OP_ROLE_KEY])
                if is_optimizer_op(op_role):
                    first_optimize_index = index
                    break

            # insert sync ops
            for index, op in enumerate(list(block.ops)):
                if op.type == 'send_v2':
                    # step1: set 'use_calc_stream' False
                    op._set_attr("use_calc_stream", False)
                    op_role = op.attr('op_role')
                    ring_id = op.attr('ring_id')
                    # step2: insert 'c_sync_calc_stream' op before 'send_v2' op
                    var_name = op.input_arg_names[0]
                    var = block.var(var_name)
                    block._insert_op_without_sync(index=index + offset,
                                                  type="c_sync_calc_stream",
                                                  inputs={'X': [var]},
                                                  outputs={'Out': [var]},
                                                  attrs={'op_role': op_role})
                    offset += 1
                    # step3: insert 'c_sync_comm_stream' op after 'send_v2' op or
                    # before first optimize op
                    if int(op_role) == int(OpRole.Backward):
                        index = first_optimize_index + offset
                        new_op_role = OpRole.Optimize
                    else:
                        index = index + offset + 1
                        new_op_role = OpRole.Backward
                    sync_comm_op = block._insert_op_without_sync(
                        index=index,
                        type="c_sync_comm_stream",
                        inputs={'X': [var]},
                        outputs={'Out': [var]},
                        attrs={
                            'op_role': new_op_role,
                            'ring_id': ring_id,
                        })
                    # step4: set 'pipeline_flag' if 'send_v2' op in forward parses
                    if int(op_role) == int(OpRole.Forward):
                        sync_comm_op._set_attr('pipeline_flag', '')
                        offset += 1
            block._sync_with_cpp()

            offset = 0
            backward_recv_index = None
            for index, op in enumerate(block.ops):
                op_role = int(op.all_attrs()[OP_ROLE_KEY])
                if op.type == "recv_v2" and is_backward_op(op_role):
                    backward_recv_index = index
                    break
            if backward_recv_index is None:
                continue

            # replace 'c_sync_comm_stream' op with 'nop' op
            for index, op in enumerate(list(block.ops)):
                if index >= backward_recv_index: break
                if op.type == 'c_sync_comm_stream' and op.has_attr(
                        'pipeline_flag'):
                    var_name = op.output_arg_names[0]
                    var = block.var(var_name)
                    block._remove_op(index + offset, sync=False)
                    offset -= 1
                    block._insert_op_without_sync(
                        index=backward_recv_index,
                        type="nop",
                        inputs={'X': [var]},
                        outputs={'Out': [var]},
                        attrs={'op_role': OpRole.Backward})
            block._sync_with_cpp()

    def _task_1f1b(self):
        cur_rank = int(os.getenv("PADDLE_TRAINER_ID", 0))
        trainer_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS", "").split(',')
        nrank = len(trainer_endpoints)
        pp_stages = len(self._dist_context.process_meshes)
        num_of_functionality = 4

        for idx, process_mesh in enumerate(self._dist_context.process_meshes):
            if cur_rank in process_mesh.processes:
                pp_idx = idx
                break
        max_slot_times = int(pp_stages - pp_idx)

        lr_ops, fwd_ops, bwd_ops, opt_ops = [], [], [], []
        for op in self._program.block(0).ops:
            # split the program based on the op_role
            op_role = int(op.all_attrs()[OP_ROLE_KEY])
            if is_lr_sched_op(op_role):
                lr_ops.append(op.desc)
            elif is_optimizer_op(op_role):
                opt_ops.append(op.desc)
            elif is_forward_op(op_role):
                fwd_ops.append(op.desc)
            elif is_backward_op(op_role):
                bwd_ops.append(op.desc)
            else:
                raise ValueError(
                    "The op role: " + str(op_role) +
                    " isn't one of LRSched, Forward, Backward or Optimizer.")

        # Create task nodes.
        lr_task_node = TaskNode(cur_rank=cur_rank,
                                max_run_times=self._acc_steps,
                                max_slot_times=max_slot_times,
                                role=int(OpRole.Optimize.LRSched),
                                ops=lr_ops,
                                task_id=int(cur_rank * num_of_functionality +
                                            0),
                                node_type="Amplifier")
        lr_task_node.set_run_pre_steps(self._acc_steps)
        fwd_task_node = TaskNode(cur_rank=cur_rank,
                                 max_run_times=self._acc_steps,
                                 max_slot_times=max_slot_times,
                                 role=int(OpRole.Forward),
                                 ops=fwd_ops,
                                 task_id=int(cur_rank * num_of_functionality +
                                             1),
                                 node_type="Compute")
        bwd_task_node = TaskNode(cur_rank=cur_rank,
                                 max_run_times=self._acc_steps,
                                 max_slot_times=max_slot_times,
                                 role=int(OpRole.Backward),
                                 ops=bwd_ops,
                                 task_id=int(cur_rank * num_of_functionality +
                                             2),
                                 node_type="Compute")
        opt_task_node = TaskNode(cur_rank=cur_rank,
                                 max_run_times=self._acc_steps,
                                 max_slot_times=max_slot_times,
                                 role=int(OpRole.Optimize),
                                 ops=opt_ops,
                                 task_id=int(cur_rank * num_of_functionality +
                                             3),
                                 node_type="Amplifier")
        opt_task_node.set_run_pre_steps(self._acc_steps)
        opt_task_node.set_run_at_offset(self._acc_steps - 1)
        task_nodes = [lr_task_node, fwd_task_node, bwd_task_node, opt_task_node]

        up_down_streams = self._dist_context.up_down_streams
        pp_upstream = up_down_streams.ups(cur_rank)
        pp_downstream = up_down_streams.downs(cur_rank)
        for i in range(num_of_functionality):
            task_node = task_nodes[i]
            task_role = task_node.role()
            cur_id = int(cur_rank * num_of_functionality + i)
            prev_id = cur_id - 1
            next_id = cur_id + 1
            ups = []
            downs = []

            pp_buff_size = int(pp_stages - pp_idx)
            if not is_lr_sched_op(task_role):
                buf_size = pp_buff_size if is_backward_op(task_role) else 1
                ups.append((prev_id, buf_size))
            if not is_optimizer_op(task_role):
                buf_size = pp_buff_size if is_forward_op(task_role) else 1
                downs.append((next_id, buf_size))

            for upstream in pp_upstream:
                upstream_id = int(upstream * num_of_functionality + i)
                if is_forward_op(task_role):
                    if upstream != -1:
                        ups.append((upstream_id, 1))
                elif is_backward_op(task_role):
                    if upstream != -1:
                        downs.append((upstream_id, 1))

            for downstream in pp_downstream:
                downstream_id = int(downstream * num_of_functionality + i)
                if is_forward_op(task_role):
                    if downstream != -1:
                        downs.append((downstream_id, 1))
                elif is_backward_op(task_role):
                    if downstream != -1:
                        ups.append((downstream_id, 1))

            for up in ups:
                print("Task:", cur_id, "'s upstream includes:", up[0],
                      ", buffer size is:", up[1])
                task_node.add_upstream_task(up[0], up[1])
            for down in downs:
                print("Task:", cur_id, "'s downstream includes:", down[0],
                      ", buffer size is:", down[1])
                task_node.add_downstream_task(down[0], down[1])

        task_id_to_rank = {}
        for i in range(nrank):
            for j in range(num_of_functionality):
                task_id_to_rank[int(i * num_of_functionality + j)] = i

        self._program._pipeline_opt = {}
        self._program._pipeline_opt['fleet_opt'] = {
            "tasks": task_nodes,
            "task_id_to_rank": task_id_to_rank,
            "num_micro_batches": self._acc_steps
        }
