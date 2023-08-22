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
import logging
import time

from paddle.distributed.passes import PassManager, new_pass
from paddle.static import append_backward, program_guard
from paddle.utils import unique_name

from ...utils.log_utils import get_logger
from ..random import init_auto_parallel_rng
from .partitioner import Partitioner
from .process_group import get_world_process_group
from .reshard import Resharder
from .utils import set_grad_var_shape


class Parallelizer:
    def __init__(self, mode, completer, dist_context):
        self._mode = mode
        self._completer = completer
        self._dist_context = dist_context
        assert self._dist_context._is_initialized
        self._pass_context = self._dist_context.pass_context
        self._strategy = self._dist_context.strategy
        self._logger = get_logger(logging.INFO)

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
            serial_loss = self._dist_context.serial_loss
            params_grads = self._generate_backward(
                serial_main_program, serial_startup_program, serial_loss
            )
            # Apply pre optimization passes
            time0 = time.time()
            (
                serial_main_program,
                serial_startup_program,
                params_grads,
            ) = self._apply_pre_optimization(
                serial_main_program,
                serial_startup_program,
                serial_loss,
                serial_optimizer,
                params_grads,
            )
            self._logger.debug(
                "within parallel apply_pre_optimization time: {}, mode {}".format(
                    time.time() - time0, self._mode
                )
            )
            # Do logical partition
            time0 = time.time()
            partitioner = Partitioner(self._dist_context, rank)
            (
                dist_main_prog,
                dist_startup_prog,
                dist_params_grads,
            ) = partitioner.partition(
                serial_main_program, serial_startup_program, params_grads
            )

            init_auto_parallel_rng()

            self._logger.debug(
                "within parallel partitioner time: {}, mode {}".format(
                    time.time() - time0, self._mode
                )
            )
            # Generate optimizer
            time0 = time.time()
            self._generate_optimizer(
                dist_main_prog,
                dist_startup_prog,
                serial_optimizer,
                dist_params_grads,
            )
            self._logger.debug(
                "within parallel optimizer time: {}, mode {}".format(
                    time.time() - time0, self._mode
                )
            )
            set_grad_var_shape(dist_main_prog, self._dist_context)
            resharder = Resharder(
                dist_main_prog,
                dist_startup_prog,
                rank,
                self._dist_context,
                dist_params_grads,
            )
            resharder.reshard()
            self._logger.debug(
                "within parallel reshard time: {}, mode {}".format(
                    time.time() - time0, self._mode
                )
            )
            # Apply post optimization passes
            time0 = time.time()
            self._apply_post_optimization(
                dist_main_prog, dist_startup_prog, rank, dist_params_grads
            )
            self._logger.debug(
                "within parallel apply_post_optimization time: {}, mode {}".format(
                    time.time() - time0, self._mode
                )
            )
        else:
            # Apply pre optimization passes
            time0 = time.time()
            (
                serial_main_program,
                serial_startup_program,
                params_grads,
            ) = self._apply_pre_optimization(
                serial_main_program, serial_startup_program, None, None, []
            )
            self._logger.debug(
                "within parallel apply_pre_optimization time: {}, mode {}".format(
                    time.time() - time0, self._mode
                )
            )
            # Do logical partition
            time0 = time.time()
            partitioner = Partitioner(self._dist_context, rank)
            (
                dist_main_prog,
                dist_startup_prog,
                dist_params_grads,
            ) = partitioner.partition(
                serial_main_program, serial_startup_program, []
            )
            # Do reshard process
            self._logger.debug(
                "within parallel partitioner time: {}, mode {}".format(
                    time.time() - time0, self._mode
                )
            )
            time0 = time.time()
            # Do reshard process
            micro_bsz = (
                1
                if not self._strategy.pipeline.enable
                else self._strategy.pipeline.micro_batch_size
            )
            resharder = Resharder(
                dist_main_prog,
                dist_startup_prog,
                rank,
                self._dist_context,
                [],
                micro_bsz,
            )
            resharder.reshard()
            self._logger.debug(
                "within parallel reshard time: {}, mode {}".format(
                    time.time() - time0, self._mode
                )
            )
            # Apply post optimization passes
            time0 = time.time()
            self._apply_post_optimization(
                dist_main_prog, dist_startup_prog, rank, dist_params_grads
            )
            self._logger.debug(
                "within parallel apply_post_optimization time: {}, mode {}".format(
                    time.time() - time0, self._mode
                )
            )
        # Clone program for test
        if self._mode != 'train':
            pipeline_opt = dist_main_prog._pipeline_opt
            dist_main_prog = dist_main_prog.clone(for_test=True)
            dist_startup_prog = dist_startup_prog.clone(for_test=True)
            dist_main_prog._pipeline_opt = pipeline_opt

        # Store the distributed programs for further usages
        self._dist_context.dist_main_programs[rank] = dist_main_prog
        self._dist_context.dist_startup_programs[rank] = dist_startup_prog

    def _generate_backward(self, main_program, startup_program, loss):
        with program_guard(main_program, startup_program):
            params_grads = append_backward(
                loss, distop_context=self._dist_context.dist_op_context
            )
        self._completer.complete_backward_annotation(main_program)
        self._dist_context.block_state.parse_backward_blocks(main_program)
        return params_grads

    def _generate_optimizer(
        self, main_program, startup_program, optimizer, params_grads
    ):
        # NOTE: `apply_gradients` will add an Accumulator for a parameter only once,
        # but optimizer will be called repeatedly in re-launch, so optimizer need to be copied.
        optimizer = copy.deepcopy(optimizer)
        self._dist_context._serial_optimizer = optimizer
        with program_guard(main_program, startup_program):
            with unique_name.guard("opt_"):
                optimizer_ops = optimizer.apply_gradients(params_grads)
        self._completer.complete_update_annotation(main_program)
        return optimizer_ops

    def _apply_pre_optimization(
        self, main_program, startup_program, loss, optimizer, params_grads
    ):
        if self._strategy is None:
            return

        # apply amp pass on train/eval/predict
        if self._strategy.amp.enable:
            config = copy.deepcopy(self._strategy.amp.to_dict())
            config["dist_context"] = self._dist_context
            config["params_grads"] = params_grads
            config["loss"] = loss
            config["input_data"] = (
                self._dist_context.serial_feed_vars["inputs"]
                + self._dist_context.serial_feed_vars["labels"]
            )
            self._logger.info(
                "Applying AMP-{}-{} ...".format(
                    config["dtype"], config['level']
                ),
            )
            if config['level'] == "o1":
                auto_parallel_amp_pass = new_pass("auto_parallel_amp", config)
                auto_parallel_amp_pass.apply(
                    [main_program], [startup_program], self._pass_context
                )
                loss = auto_parallel_amp_pass.get_loss()
            elif config['level'] in ['o2', 'o3']:
                config["base_opt"] = optimizer
                auto_parallel_fp16_pass = new_pass("auto_parallel_fp16", config)
                auto_parallel_fp16_pass.apply(
                    [main_program], [startup_program], self._pass_context
                )
                loss = auto_parallel_fp16_pass.get_loss()
            else:
                raise ValueError("AMP level should be one of o1, o2, o3")

        # apply quantization pass
        # The pass can be applied when mode must be 'train'
        if self._mode == 'train' and self._strategy.qat.enable:
            config = copy.deepcopy(self._strategy.qat.to_dict())
            config["dist_context"] = self._dist_context
            config["params_grads"] = params_grads
            config["mode"] = self._mode
            config["loss"] = loss
            auto_parallel_quantization_pass = new_pass(
                "auto_parallel_quantization", config
            )
            auto_parallel_quantization_pass.apply(
                [main_program], [startup_program], self._pass_context
            )
            main_program = self._pass_context.get_attr("main_program")
            startup_program = self._pass_context.get_attr("startup_program")
            params_grads = self._pass_context.get_attr("params_grads")
            loss = self._pass_context.get_attr("loss")

        # apply recompute pass
        # recompute is then train-only optimization
        if self._mode == "train" and self._strategy.recompute.enable:
            config = copy.deepcopy(self._strategy.recompute.to_dict())
            config["dist_context"] = self._dist_context
            config["no_grad_set"] = None
            config["loss"] = loss
            auto_parallel_recompute_pass = new_pass(
                "auto_parallel_recompute", config
            )
            auto_parallel_recompute_pass.apply(
                [main_program], [startup_program], self._pass_context
            )

        return main_program, startup_program, params_grads

    def _apply_post_optimization(
        self, main_program, startup_program, rank, params_grads
    ):
        if self._strategy is None:
            return

        # data parallel optimization
        if self._strategy.dp_optimization.enable:
            config = copy.deepcopy(self._strategy.dp_optimization.to_dict())
            config["dist_context"] = self._dist_context
            config["global_rank"] = rank
            config["use_sharding"] = self._strategy.sharding.enable
            dp_pass = new_pass(
                "auto_parallel_data_parallel_optimization", config
            )
            dp_pass.apply([main_program], [startup_program], self._pass_context)

        if self._strategy.sharding.enable:
            config = copy.deepcopy(self._strategy.sharding.to_dict())
            config["dist_context"] = self._dist_context
            config["params_grads"] = params_grads
            config["global_rank"] = rank
            auto_parallel_sharding_pass = new_pass(
                "auto_parallel_sharding", config
            )
            auto_parallel_sharding_pass.apply(
                [main_program], [startup_program], self._pass_context
            )
            params_grads = self._pass_context.get_attr("params_grads")

        if self._mode == "train":
            # GradClip is train-only optimization
            config = copy.deepcopy(self._strategy.sharding.to_dict())
            config["dist_context"] = self._dist_context
            config["params_grads"] = params_grads
            config["rank_id"] = rank
            auto_parallel_clip_pass = new_pass(
                "auto_parallel_grad_clip", config
            )
            auto_parallel_clip_pass.apply(
                [main_program], [startup_program], self._pass_context
            )

            # deps for newexe
            config = {}
            config["dist_context"] = self._dist_context
            APSED_pass = new_pass(
                "auto_parallel_supplement_explicit_dependencies", config
            )
            APSED_pass.apply(
                [main_program], [startup_program], self._pass_context
            )

        if self._strategy.pipeline.enable:
            self._strategy.gradient_merge.enable = True
            self._strategy.gradient_merge.k_steps = (
                self._strategy.pipeline.accumulate_steps
            )
            self._strategy.gradient_merge.avg = True

        # gradient_merge is then train-only optimization
        if self._mode == "train" and self._strategy.gradient_merge.enable:
            config = copy.deepcopy(self._strategy.gradient_merge.to_dict())
            config["dist_context"] = self._dist_context
            config["params_grads"] = params_grads
            auto_parallel_gradient_merge_pass = new_pass(
                "auto_parallel_gradient_merge_pass", config
            )
            auto_parallel_gradient_merge_pass.apply(
                [main_program], [startup_program], self._pass_context
            )

        if self._strategy.pipeline.enable:
            config = copy.deepcopy(self._strategy.pipeline.to_dict())
            config["dist_context"] = self._dist_context
            auto_parallel_pipeline_pass = new_pass(
                "auto_parallel_pipeline", config
            )
            auto_parallel_pipeline_pass.apply(
                [main_program], [startup_program], self._pass_context
            )

        if self._mode == "train" and self._strategy.fused_passes.enable:
            if len(self._strategy.fused_passes.fused_passes_list) > 0:
                new_pass_list = []
                for op in self._strategy.fused_passes.fused_passes_list:
                    new_pass_list.append(new_pass(op))
                pass_manager = PassManager(new_pass_list)
                pass_manager.apply([main_program], [startup_program])
