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

import time
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
        world_process_group = get_world_process_group()
        all_ranks = world_process_group.ranks
        for rank in all_ranks:
            # self._dist_context._backup(serial=True, dist=True)
            self.parallel(rank)
            # self._dist_context._restore(serial=True, dist=True)

    def parallel(self, rank):
        serial_main_program = self._dist_context.serial_main_program
        serial_startup_program = self._dist_context.serial_startup_program
        serial_optimizer = self._dist_context.serial_optimizer
        if self._mode == "train" and serial_optimizer:
            # Generate backward
            time1 = time.time()
            print("**********_generate_backward")
            serial_loss = self._dist_context.serial_loss
            params_grads = self._generate_backward(serial_main_program,
                                                   serial_startup_program,
                                                   serial_loss)
            for process_mesh in self._dist_context.process_meshes:
                print("--> processes:", process_mesh.processes)
            print("--> time:", time.time() - time1)
            # Apply pre optimization passes
            time2 = time.time()
            print("**********_apply_pre_optimization")
            self._apply_pre_optimization(serial_main_program,
                                         serial_startup_program, serial_loss,
                                         serial_optimizer, params_grads)
            for process_mesh in self._dist_context.process_meshes:
                print("--> processes:", process_mesh.processes)
            print("--> time:", time.time() - time2)

            # Do logical partition
            time3 = time.time()
            print("**********partition")
            partitioner = Partitioner(self._dist_context, rank)
            dist_main_prog, dist_startup_prog, dist_params_grads = partitioner.partition(
                serial_main_program, serial_startup_program, params_grads)
            for process_mesh in self._dist_context.process_meshes:
                print("--> processes:", process_mesh.processes)
            print("--> time:", time.time() - time3)

            # Generate optimizer
            time4 = time.time()
            print("**********_generate_optimizer")
            self._generate_optimizer(dist_main_prog, dist_startup_prog,
                                     serial_optimizer, dist_params_grads)
            for process_mesh in self._dist_context.process_meshes:
                print("--> processes:", process_mesh.processes)
            print("--> time:", time.time() - time4)

            # Do reshard process
            set_grad_var_shape(dist_main_prog, self._dist_context)
            time5 = time.time()
            print("**********Resharder")
            resharder = Resharder(dist_main_prog, dist_startup_prog, rank,
                                  self._dist_context, dist_params_grads)
            resharder.reshard()
            for process_mesh in self._dist_context.process_meshes:
                print("--> processes:", process_mesh.processes)
            print("--> time:", time.time() - time5)

            # Apply post optimization passes
            time6 = time.time()
            print("**********_apply_post_optimization")
            self._apply_post_optimization(dist_main_prog, dist_startup_prog,
                                          rank, dist_params_grads)
            for process_mesh in self._dist_context.process_meshes:
                print("--> processes:", process_mesh.processes)
            print("--> time:", time.time() - time6)
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
            print("**********amp")
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
            print("**********recompute")
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

        if self._strategy.pipeline:
            print("**********pipeline")
            acc_steps = self._strategy.pipeline_configs["accumulate_steps"]
            self._strategy.gradient_merge = True
            self._strategy.gradient_merge_configs = {
                "k_steps": acc_steps,
                "avg": True
            }

        if self._strategy.gradient_merge:
            print("**********gradient merge")
            config = copy.deepcopy(self._strategy.gradient_merge_configs)
            config["dist_context"] = self._dist_context
            config["params_grads"] = params_grads
            auto_parallel_gradient_merge_pass = new_pass(
                "auto_parallel_gradient_merge_pass", config)
            auto_parallel_gradient_merge_pass.apply([main_program],
                                                    [startup_program],
                                                    self._pass_context)

        if self._strategy.pipeline:
            print("**********pipeline")
            config = copy.deepcopy(self._strategy.pipeline_configs)
            config["dist_context"] = self._dist_context
            # config["params_grads"] = params_grads
            auto_parallel_pipeline_pass = new_pass("auto_parallel_pipeline",
                                                   config)
            auto_parallel_pipeline_pass.apply([main_program], [startup_program],
                                              self._pass_context)
