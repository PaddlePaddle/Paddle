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
import paddle

from paddle.framework import core
from .pass_base import PassBase, register_pass
from paddle.fluid.framework import Program, Parameter
from paddle.distributed.fleet.fleet_executor_utils import TaskNode
from paddle.distributed.fleet.meta_optimizers.common import OpRole, OP_ROLE_KEY, is_loss_grad_op
from paddle.distributed.auto_parallel.partitioner import __not_shape_var_type__
from paddle.distributed.auto_parallel.utils import is_forward_op, is_backward_op, is_optimize_op, is_lr_sched_op


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
        print("**********pipeline pass")
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
                if is_optimize_op(op):
                    first_optimize_index = index
                    break

            # insert sync ops
            for index, op in enumerate(list(block.ops)):
                # if is_loss_grad_op(op):
                #     assert op.type == 'fill_constant', \
                #         "loss_grad_op must be fill_constant op, " \
                #         "but this op is {}".format(op.type)
                #     assert op.has_attr('value')
                #     loss_scale = float(op.attr('value'))
                #     loss_scale = loss_scale / self._acc_steps
                #     op._set_attr('value', loss_scale)
                #     break

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
                    # before the first optimize op
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
                    # step4: If 'send_v2' op in forward parse, set 'pipeline_flag' to distinguish
                    # whether the 'c_sync_comm_stream' op is inserted for pipeline.
                    if int(op_role) == int(OpRole.Forward):
                        sync_comm_op._set_attr('pipeline_flag', '')
                        offset += 1
            block._sync_with_cpp()

            offset = 0
            backward_recv_index = None
            for index, op in enumerate(block.ops):
                if op.type == "recv_v2" and is_backward_op(op):
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

    def _create_param(self, dst_block, src_var):
        copied_kwargs = {}
        copied_kwargs['trainable'] = src_var.trainable
        copied_kwargs['optimize_attr'] = src_var.optimize_attr
        copied_kwargs['regularizer'] = src_var.regularizer
        copied_kwargs['do_model_average'] = src_var.do_model_average
        copied_kwargs['need_clip'] = src_var.need_clip

        Parameter(block=dst_block,
                  type=src_var.type,
                  name=src_var.name,
                  shape=src_var.shape,
                  dtype=src_var.dtype,
                  lod_level=src_var.lod_level,
                  error_clip=src_var.error_clip,
                  stop_gradient=src_var.stop_gradient,
                  is_data=src_var.is_data,
                  belong_to_optimizer=src_var.belong_to_optimizer,
                  **copied_kwargs)

    def _create_inter(self, dst_block, src_var):
        dst_block.create_var(type=src_var.type,
                             name=src_var.name,
                             shape=src_var.shape,
                             dtype=src_var.dtype,
                             lod_level=src_var.lod_level,
                             persistable=src_var.persistable,
                             error_clip=src_var.error_clip,
                             stop_gradient=src_var.stop_gradient,
                             is_data=src_var.is_data,
                             belong_to_optimizer=src_var.belong_to_optimizer)

    def _create_var(self, src_block, dst_block, src_varname):

        src_var = src_block.var(src_varname)
        if src_var.type in __not_shape_var_type__:
            persist = getattr(src_var, 'persistable', False)
            dst_block.create_var(type=src_var.type,
                                 name=src_varname,
                                 persistable=persist,
                                 stop_gradient=True)
        else:
            if isinstance(src_var, Parameter):
                self._create_param(dst_block, src_var)
            else:
                self._create_inter(dst_block, src_var)

    def _create_program(self, src_block, dst_block, src_op):
        dst_op_desc = dst_block.desc.append_op()
        dst_op_desc.copy_from(src_op.desc)
        for input_varname in src_op.input_arg_names:
            if src_block.has_var(input_varname):
                self._create_var(src_block, dst_block, input_varname)
        for output_varname in src_op.output_arg_names:
            if src_block.has_var(output_varname):
                self._create_var(src_block, dst_block, output_varname)

    def _task_1f1b(self):
        cur_rank = int(os.getenv("PADDLE_TRAINER_ID", 0))
        trainer_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS", "").split(',')
        nrank = len(trainer_endpoints)
        pp_stages = len(self._dist_context.process_meshes)
        num_of_functionality = 4

        # all_processes = list(lambda a: a.processes, self._dist_context.process_meshes)
        # pp_processes =

        # compute current pp stage
        for idx, process_mesh in enumerate(self._dist_context.process_meshes):
            if cur_rank in process_mesh.processes:
                pp_idx = idx
                break
        max_slot_times = int(pp_stages - pp_idx)

        print("cur_rank:", cur_rank)
        print("pp stage:", pp_idx)
        print("process_meshes:", self._dist_context.process_meshes)
        for process_mesh in self._dist_context.process_meshes:
            print("    processes:", process_mesh.processes)

        # create program with op_role
        lr_prog, fwd_prog, bwd_prog, opt_prog = Program(), Program(), Program(
        ), Program()
        for idx, src_block in enumerate(self._program.blocks):
            if idx == 0:
                lr_block = lr_prog.block(0)
                fwd_block = fwd_prog.block(0)
                bwd_block = bwd_prog.block(0)
                opt_block = opt_prog.block(0)
            else:
                lr_block = lr_prog._create_block(
                    parent_idx=src_block.parent_idx)
                fwd_block = fwd_prog._create_block(
                    parent_idx=src_block.parent_idx)
                bwd_block = bwd_prog._create_block(
                    parent_idx=src_block.parent_idx)
                opt_block = opt_prog._create_block(
                    parent_idx=src_block.parent_idx)
                lr_block._set_forward_block_idx(src_block.forward_block_idx)
                fwd_block._set_forward_block_idx(src_block.forward_block_idx)
                bwd_block._set_forward_block_idx(src_block.forward_block_idx)
                opt_block._set_forward_block_idx(src_block.forward_block_idx)

            # split the program based on the op_role
            for op in src_block.ops:
                if is_lr_sched_op(op):
                    self._create_program(src_block, lr_block, op)
                elif is_forward_op(op):
                    self._create_program(src_block, fwd_block, op)
                elif is_backward_op(op):
                    self._create_program(src_block, bwd_block, op)
                elif is_optimize_op(op):
                    self._create_program(src_block, opt_block, op)
                else:
                    raise ValueError(
                        "The op role: " + str(op.attr('op_role')) +
                        " isn't one of LRSched, Forward, Backward or Optimizer."
                    )

        lr_prog._sync_with_cpp()
        fwd_prog._sync_with_cpp()
        bwd_prog._sync_with_cpp()
        opt_prog._sync_with_cpp()

        lr_prog._rollback()
        fwd_prog._rollback()
        bwd_prog._rollback()
        opt_prog._rollback()
        # print("lr*******************")
        # print(lr_prog)
        # print("fwd*******************")
        # print(fwd_prog)
        # print("bwd*******************")
        # print(bwd_prog)
        # print("opt*******************")
        # print(opt_prog)

        # Create task nodes.
        lr_task_node = TaskNode(rank=cur_rank,
                                max_run_times=self._acc_steps,
                                max_slot_times=max_slot_times,
                                program=lr_prog,
                                task_id=int(cur_rank * num_of_functionality +
                                            0),
                                node_type="Amplifier",
                                lazy_initialize=True)
        lr_task_node.set_run_pre_steps(self._acc_steps)
        fwd_task_node = TaskNode(rank=cur_rank,
                                 max_run_times=self._acc_steps,
                                 max_slot_times=max_slot_times,
                                 program=fwd_prog,
                                 task_id=int(cur_rank * num_of_functionality +
                                             1),
                                 node_type="Compute",
                                 lazy_initialize=True)
        bwd_task_node = TaskNode(rank=cur_rank,
                                 max_run_times=self._acc_steps,
                                 max_slot_times=max_slot_times,
                                 program=bwd_prog,
                                 task_id=int(cur_rank * num_of_functionality +
                                             2),
                                 node_type="Compute",
                                 lazy_initialize=True)
        opt_task_node = TaskNode(rank=cur_rank,
                                 max_run_times=self._acc_steps,
                                 max_slot_times=max_slot_times,
                                 program=opt_prog,
                                 task_id=int(cur_rank * num_of_functionality +
                                             3),
                                 node_type="Amplifier",
                                 lazy_initialize=True)
        opt_task_node.set_run_pre_steps(self._acc_steps)
        opt_task_node.set_run_at_offset(self._acc_steps - 1)
        task_nodes = {
            "lr": lr_task_node,
            "fwd": fwd_task_node,
            "bwd": bwd_task_node,
            "opt": opt_task_node
        }

        # get upstream ranks and downstream ranks of cur_rank
        up_down_streams = self._dist_context.up_down_streams
        pp_upstream = up_down_streams.ups(cur_rank)
        pp_downstream = up_down_streams.downs(cur_rank)

        # set upstream/downstream for task_nodes of cur_rank
        for i, (task_role, task_node) in enumerate(task_nodes.items()):

            cur_id = int(cur_rank * num_of_functionality + i)
            ups = []
            downs = []

            # set upstream/downstream and buffersize in pipeline stage
            pp_buff_size = int(pp_stages - pp_idx)
            prev_id = cur_id - 1
            next_id = cur_id + 1
            if task_role != "lr":
                buf_size = pp_buff_size if task_role == "bwd" else 2
                ups.append((prev_id, buf_size))
            if task_role != "opt":
                buf_size = pp_buff_size if task_role == "fwd" else 2
                downs.append((next_id, buf_size))

            # set upstream/downstream and buffersize cross pipeline stage
            for upstream in pp_upstream:
                upstream_id = int(upstream * num_of_functionality + i)
                if task_role == "fwd":
                    if upstream != -1:
                        ups.append((upstream_id, 2))
                elif task_role == "bwd":
                    if upstream != -1:
                        downs.append((upstream_id, 2))
            for downstream in pp_downstream:
                downstream_id = int(downstream * num_of_functionality + i)
                if task_role == "fwd":
                    if downstream != -1:
                        downs.append((downstream_id, 2))
                elif task_role == "bwd":
                    if downstream != -1:
                        ups.append((downstream_id, 2))

            for up in ups:
                print("Task:", cur_id, "'s upstream includes:", up[0],
                      ", buffer size is:", up[1])
                task_node.add_upstream_task(up[0], up[1])
            for down in downs:
                print("Task:", cur_id, "'s downstream includes:", down[0],
                      ", buffer size is:", down[1])
                task_node.add_downstream_task(down[0], down[1])

        # record global message: task_id_to_rank
        task_id_to_rank = {}
        for i in range(nrank):
            for j in range(num_of_functionality):
                task_id_to_rank[int(i * num_of_functionality + j)] = i

        self._program._pipeline_opt = {}
        self._program._pipeline_opt['fleet_opt'] = {
            "tasks": list(task_nodes.values()),
            "task_id_to_rank": task_id_to_rank,
            "num_micro_batches": self._acc_steps
        }
