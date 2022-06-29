#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import copy
from collections import defaultdict

from paddle.fluid import program_guard
from paddle.fluid.backward import append_backward
from paddle.distributed.passes import new_pass

from .reshard import Resharder
from .partitioner import Partitioner
from .dist_op import DistributedOperator
from .dist_saver import DistributedSaver
from .dist_loader import NonIterableGeneratorLoader
from .utils import make_data_unshard, set_grad_var_shape
from .utils import print_program_with_dist_attr, to_list
from .process_group import get_all_process_groups, get_world_process_group
from .dist_context import DistributedContext, get_default_distributed_context


class Parallelizer:

    def __init__(self, mode, completer, dist_context):
        self._mode = mode
        self._completer = completer
        self._dist_context = dist_context
        assert self._dist_context._is_initialized
        self._pass_context = self._dist_context.pass_context
        self._strategy = self._dist_context.strategy

    def parallel_all(self):
        # def is_union_process_mesh(process_mesh, dist_context):
        #     sub_set_count = 0
        #     for item in dist_context.process_meshes:
        #         for process in item.processes:
        #             if process in process_mesh.processes:
        #                 sub_set_count += 1
        #                 break
        #     if sub_set_count > 1:
        #         return True
        #     return False

        # # This is a trick to avoid output process mesh different from tensor process mesh (non-union process mesh)
        # for dist_op in self._dist_context._dist_ops_for_program.values():
        #     serial_op = dist_op.serial_op
        #     if serial_op.type == "while":
        #         continue
        #     else:
        #         for var_name in serial_op.output_arg_names:
        #             var = serial_op.block._var_recursive(var_name)
        #             dist_tensor = self._dist_context.get_dist_tensor_for_program(var)
        #             if dist_tensor.dist_attr.process_mesh != dist_op.dist_attr.process_mesh:
        #                 print("parallel_tuner.py process_mesh", dist_tensor.dist_attr.process_mesh, dist_op.dist_attr.process_mesh, is_union_process_mesh(dist_tensor.dist_attr.process_mesh, self._dist_context), is_union_process_mesh(dist_op.dist_attr.process_mesh, self._dist_context))

        #                 if not is_union_process_mesh(dist_tensor.dist_attr.process_mesh, self._dist_context) and not is_union_process_mesh(dist_op.dist_attr.process_mesh, self._dist_context):
        #                     dist_tensor.dist_attr.process_mesh = dist_op.dist_attr.process_mesh
        #                     print("parallel_tuner.py after change process_mesh", dist_tensor, dist_op)

        world_process_group = get_world_process_group()
        all_ranks = world_process_group.ranks
        for rank in all_ranks:
            self._dist_context._backup(serial=True, dist=True)
            self.parallel(rank)
            self._dist_context._restore(serial=True, dist=True)

    def parallel(self, rank):
        serial_main_program = self._dist_context.serial_main_program
        serial_startup_program = self._dist_context.serial_startup_program
        serial_optimizer = self._dist_context.serial_optimizer
        if self._mode == "train" and serial_optimizer:
            # Generate backward
            serial_loss = self._dist_context.serial_loss
            params_grads = self._generate_backward(serial_main_program,
                                                   serial_startup_program,
                                                   serial_loss)
            # Apply pre optimization passes
            self._apply_pre_optimization(serial_main_program,
                                         serial_startup_program, serial_loss,
                                         serial_optimizer, params_grads)

            # Do logical partition
            partitioner = Partitioner(self._dist_context, rank)
            dist_main_prog, dist_startup_prog, dist_params_grads = partitioner.partition(
                serial_main_program, serial_startup_program, params_grads)
            # Generate optimizer
            self._generate_optimizer(dist_main_prog, dist_startup_prog,
                                     serial_optimizer, dist_params_grads)
            # Do reshard process
            set_grad_var_shape(dist_main_prog, self._dist_context)
            resharder = Resharder(dist_main_prog, dist_startup_prog, rank,
                                  self._dist_context, dist_params_grads)
            resharder.reshard()

            # Apply post optimization passes
            self._apply_post_optimization(dist_main_prog, dist_startup_prog,
                                          rank, dist_params_grads)
        else:
            # Apply pre optimization passes
            self._apply_pre_optimization(serial_main_program,
                                         serial_startup_program, None, None,
                                         None)
            # Do logical partition
            partitioner = Partitioner(self._dist_context, rank)
            dist_main_prog, dist_startup_prog, dist_params_grads = partitioner.partition(
                serial_main_program, serial_startup_program, [])
            # Do reshard process
            resharder = Resharder(dist_main_prog, dist_startup_prog, rank,
                                  self._dist_context, [], 1)
            resharder.reshard()

            # # insert print op to debug lod_set op error
            # for i in range(len(dist_main_prog.blocks)):
            #     block = dist_main_prog.blocks[i]
            #     index = 0
            #     while index < len(block.ops):
            #         op = block.ops[index]
            #         # print("parallelizer_v2 op_type****", op.type)
            #         if op.type == "lod_reset":
            #             print("parallelizer_v2 Enter lod_reset****")
            #             if "lod_reset_0.tmp_0" in op.output_arg_names:
            #                 for var_name in op.input_arg_names:
            #                     input_var = block._var_recursive(var_name)
            #                     print("parallelizer_v2.py input", input_var)
            #                     out = block.create_var(name=input_var.name+"@PRINT", dtype=input_var.dtype, type=input_var.type)
            #                     block._insert_op(
            #                         index,
            #                         type='print',
            #                         inputs={'In': input_var},
            #                         outputs={'Out': out},
            #                         attrs={
            #                             'first_n': -1,
            #                             'summarize': 20,
            #                             'message': "",
            #                             'print_tensor_name': True,
            #                             'print_tensor_type': True,
            #                             'print_tensor_shape': True,
            #                             'print_tensor_layout': True,
            #                             'print_tensor_lod': True,
            #                             'print_phase': 'both'.upper()
            #                         })
            #                     index += 1
            #                 break
            #         else:
            #             index += 1
        # Clone program for test
        if self._mode != 'train':
            dist_main_prog = dist_main_prog.clone(for_test=True)
            dist_startup_prog = dist_startup_prog.clone(for_test=True)

        # Store the distributed programs for further usages
        self._dist_context.dist_main_programs[rank] = dist_main_prog
        self._dist_context.dist_startup_programs[rank] = dist_startup_prog

    def _generate_backward(self, main_program, startup_program, loss):
        with program_guard(main_program, startup_program):
            params_grads = append_backward(
                loss, distop_context=self._dist_context.dist_op_context)
        self._completer.complete_backward_annotation(main_program)
        self._dist_context.block_state.parse_backward_blocks(main_program)
        return params_grads

    def _generate_optimizer(self, main_program, startup_program, optimizer,
                            params_grads):
        with program_guard(main_program, startup_program):
            optimizer_ops = copy.deepcopy(optimizer).apply_gradients(
                params_grads)
        self._completer.complete_update_annotation(main_program)
        return optimizer_ops

    def _apply_pre_optimization(self, main_program, startup_program, loss,
                                optimizer, params_grads):
        if self._strategy is None:
            return
        # apply amp pass
        if self._strategy.amp:
            config = copy.deepcopy(self._strategy.amp_configs)
            config["dist_context"] = self._dist_context
            config["params_grads"] = params_grads
            config["loss"] = loss
            config["input_data"] = self._dist_context.serial_feed_vars["inputs"] \
                + self._dist_context.serial_feed_vars["labels"]
            if config["use_pure_fp16"]:
                config["base_opt"] = optimizer
                auto_parallel_fp16_pass = new_pass("auto_parallel_fp16", config)
                auto_parallel_fp16_pass.apply([main_program], [startup_program],
                                              self._pass_context)
            else:
                auto_parallel_amp_pass = new_pass("auto_parallel_amp", config)
                auto_parallel_amp_pass.apply([main_program], [startup_program],
                                             self._pass_context)

        # apply recompute pass
        if self._strategy.recompute:
            config = copy.deepcopy(self._strategy.recompute_configs)
            config["dist_context"] = self._dist_context
            config["no_grad_set"] = None
            config["loss"] = loss
            auto_parallel_recompute_pass = new_pass("auto_parallel_recompute",
                                                    config)
            auto_parallel_recompute_pass.apply([main_program],
                                               [startup_program],
                                               self._pass_context)

    def _apply_post_optimization(self, main_program, startup_program, rank,
                                 params_grads):
        if self._strategy is None:
            return
        if self._strategy.sharding:
            config = copy.deepcopy(self._strategy.sharding_configs)
            config["dist_context"] = self._dist_context
            config["params_grads"] = params_grads
            config["global_rank"] = rank
            auto_parallel_sharding_pass = new_pass("auto_parallel_sharding",
                                                   config)
            auto_parallel_sharding_pass.apply([main_program], [startup_program],
                                              self._pass_context)

        if self._strategy.gradient_merge:
            config = copy.deepcopy(self._strategy.gradient_merge_configs)
            config["dist_context"] = self._dist_context
            config["params_grads"] = params_grads
            auto_parallel_gradient_merge_pass = new_pass(
                "auto_parallel_gradient_merge_pass", config)
            auto_parallel_gradient_merge_pass.apply([main_program],
                                                    [startup_program],
                                                    self._pass_context)
