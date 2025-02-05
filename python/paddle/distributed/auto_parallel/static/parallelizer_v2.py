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
import os
import time

from paddle.distributed.passes.pass_base import PassManager, new_pass
from paddle.framework import get_flags
from paddle.static import append_backward, program_guard

from ...utils.log_utils import get_logger
from ..random import init_auto_parallel_rng
from .partitioner import Partitioner
from .process_group import get_world_process_group
from .reshard import Resharder
from .utils import (
    get_pp_stage,
    is_sequential_run,
)

PIR_PASS = [
    'fused_gemm_epilogue_pass',
    'fused_linear_param_grad_add_pass',
    'fuse_allreduce_split_to_reducescatter_pass',
    'fused_dropout_add_pass',
]

PIR_PYTHON_PASS = [
    'eliminate_transpose',
]


class Parallelizer:
    def __init__(self, mode, completer, dist_context):
        self._mode = mode
        self._completer = completer
        self._dist_context = dist_context
        assert self._dist_context._is_initialized
        self._pass_context = self._dist_context.pass_context
        self._strategy = self._dist_context.strategy
        self._logger = get_logger(logging.INFO)

    @property
    def is_train(self):
        return self._mode == "train"

    @property
    def is_test(self):
        return self._mode in ["eval", "predict"]

    def parallel_all(self, parameter_list=None):
        world_process_group = get_world_process_group()
        all_ranks = world_process_group.ranks
        for rank in all_ranks:
            # self._dist_context._backup(serial=True, dist=True)
            self.parallel(rank, parameter_list)
            # self._dist_context._restore(serial=True, dist=True)

    def parallel(self, rank, parameter_list=None):
        serial_main_program = self._dist_context.serial_main_program
        serial_startup_program = self._dist_context.serial_startup_program
        serial_optimizer = self._dist_context.serial_optimizer
        if self.is_train and serial_optimizer:
            # Generate backward
            serial_loss = self._dist_context.serial_loss
            params_grads = self._generate_backward(
                serial_main_program,
                serial_startup_program,
                serial_loss,
                parameter_list,
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
                f"within parallel apply_pre_optimization time: {time.time() - time0}, mode {self._mode}"
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
                f"within parallel partitioner time: {time.time() - time0}, mode {self._mode}"
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
                f"within parallel optimizer time: {time.time() - time0}, mode {self._mode}"
            )

            resharder = Resharder(
                dist_main_prog,
                dist_startup_prog,
                rank,
                self._dist_context,
                dist_params_grads,
            )
            resharder.reshard()
            self._logger.debug(
                f"within parallel reshard time: {time.time() - time0}, mode {self._mode}"
            )
            # Apply post optimization passes
            time0 = time.time()
            self._apply_post_optimization(
                dist_main_prog, dist_startup_prog, rank, dist_params_grads
            )
            self._logger.debug(
                f"within parallel apply_post_optimization time: {time.time() - time0}, mode {self._mode}"
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
                f"within parallel apply_pre_optimization time: {time.time() - time0}, mode {self._mode}"
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
                f"within parallel partitioner time: {time.time() - time0}, mode {self._mode}"
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
                f"within parallel reshard time: {time.time() - time0}, mode {self._mode}"
            )
            # Apply post optimization passes
            time0 = time.time()
            self._apply_post_optimization(
                dist_main_prog, dist_startup_prog, rank, dist_params_grads
            )
            self._logger.debug(
                f"within parallel apply_post_optimization time: {time.time() - time0}, mode {self._mode}"
            )

        # Clone program for test
        if self.is_test:
            pipeline_opt = dist_main_prog._pipeline_opt
            dist_main_prog = dist_main_prog.clone(for_test=True)
            dist_startup_prog = dist_startup_prog.clone(for_test=True)
            dist_main_prog._pipeline_opt = pipeline_opt

        # Store the distributed programs for further usages
        self._dist_context.dist_main_programs[rank] = dist_main_prog
        self._dist_context.dist_startup_programs[rank] = dist_startup_prog

    def _generate_backward(
        self, main_program, startup_program, loss, parameter_list=None
    ):
        # NOTE(zhaoyinglia):
        # Guarantee the order of params_grads is same between dynamic mode and static mode
        # by making parameter_list equal to model.parameters(),
        # because the order affect the result of ClipGradByGLobalNorm.
        # If parameter_list is not None, the order of params_grads is same with parameter_list.
        # If parameter_list is None, params_grads will be as prog.global_block().all_parameters().
        with program_guard(main_program, startup_program):
            params_grads = append_backward(
                loss,
                parameter_list=parameter_list,
                distop_context=self._dist_context.dist_op_context,
            )
        self._completer.complete_backward_annotation(main_program)
        self._dist_context.block_state.parse_backward_blocks(main_program)
        return params_grads

    def _generate_optimizer(
        self, main_program, startup_program, optimizer, params_grads
    ):
        # NOTE:
        # 1. `apply_gradients` will add an Accumulator for a parameter only once,
        #    but optimizer will be called repeatedly in re-launch, so optimizer need to be copied.
        # 2. lr_scheduler cannot be deepcopy, cause 'deepcopy' will lead to difference of learning_rate between executor and engine.
        learning_rate = optimizer._learning_rate
        new_optimizer = copy.deepcopy(optimizer)
        new_optimizer._learning_rate = learning_rate
        new_optimizer._sorted = False
        self._dist_context._serial_optimizer = optimizer
        self._dist_context._serial_optimizer._learning_rate = learning_rate

        with program_guard(main_program, startup_program):
            with main_program.switch_name_generator_guard("opt_"):
                optimizer_ops = new_optimizer.apply_gradients(params_grads)
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
        if self.is_train and self._strategy.qat.enable:
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
        if self.is_train and self._strategy.recompute.enable:
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

    def _check_dist_attr(self, program, num_model_chunks, dist_context):
        for _, block in enumerate(program.blocks):
            for _, op in enumerate(block.ops):
                op_dist_attr = dist_context.get_op_dist_attr_for_program(op)
                if op_dist_attr is None:
                    raise ValueError(
                        f"There is not dist_attr for op[{op.type}]."
                    )

    def _apply_post_optimization(
        self, main_program, startup_program, rank, params_grads
    ):
        if self._strategy is None:
            return

        # sequence parallel optimization
        if self._strategy.sp_optimization.enable:
            config = copy.deepcopy(self._strategy.sp_optimization.to_dict())
            config["dist_context"] = self._dist_context
            config["global_rank"] = rank
            sp_pass = new_pass(
                "auto_parallel_sequence_parallel_optimization", config
            )
            sp_pass.apply([main_program], [startup_program], self._pass_context)

        # apply fused linear promotion pass
        if (
            self.is_train
            and self._strategy.fused_linear_promotion.enable
            and self._strategy.fused_passes.enable
        ):
            if (
                len(self._strategy.fused_passes.fused_passes_list) > 0
                and "fuse_gemm_epilogue"
                in self._strategy.fused_passes.fused_passes_list
            ):
                amp_config = None
                if self._strategy.amp.enable:
                    amp_config = copy.deepcopy(self._strategy.amp.to_dict())
                config = {}
                config["dist_context"] = self._dist_context
                config["global_rank"] = rank
                config["enable_sp"] = self._strategy.sp_optimization.enable
                config["params_grads"] = params_grads
                config["amp_level"] = (
                    amp_config['level'] if amp_config is not None else "o0"
                )
                fused_linear_promotion_pass = new_pass(
                    "auto_parallel_fused_linear_promotion", config
                )
                fused_linear_promotion_pass.apply(
                    [main_program], [startup_program], self._pass_context
                )

        # apply master grad pass
        if self._strategy.amp.enable:
            amp_config = copy.deepcopy(self._strategy.amp.to_dict())
            config = {}
            config["dist_context"] = self._dist_context
            config["params_grads"] = params_grads
            config["completer"] = self._completer
            if amp_config['level'] == "o2" and amp_config["use_master_grad"]:
                master_grad_pass = new_pass(
                    "auto_parallel_master_grad_pass", config
                )
                master_grad_pass.apply(
                    [main_program], [startup_program], self._pass_context
                )

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

        gradient_sync_after_accumulate = (
            self._strategy.dp_optimization.gradient_sync_after_accumulate
        )
        if gradient_sync_after_accumulate:
            global_params_grads = params_grads

        if self._strategy.sharding.enable:
            config = copy.deepcopy(self._strategy.sharding.to_dict())
            config["dist_context"] = self._dist_context
            config["params_grads"] = params_grads
            config["global_rank"] = rank
            config["gradient_sync_after_accumulate"] = (
                gradient_sync_after_accumulate
            )
            if self._strategy.amp.enable:
                amp_config = copy.deepcopy(self._strategy.amp.to_dict())
                config["amp_dtype"] = amp_config['dtype']
            auto_parallel_sharding_pass = new_pass(
                "auto_parallel_sharding", config
            )
            auto_parallel_sharding_pass.apply(
                [main_program], [startup_program], self._pass_context
            )
            params_grads = self._pass_context.get_attr("params_grads")

        if self._strategy.mp_optimization.allreduce_matmul_grad_overlapping:
            if int(os.getenv("CUDA_DEVICE_MAX_CONNECTIONS", "0")) != 1:
                self._logger.warning(
                    "You set mp_optimization.allreduce_matmul_grad_overlapping=True, but you did not set environment "
                    "variable CUDA_DEVICE_MAX_CONNECTIONS=1, which may leads to performance "
                    "loss. Try to export CUDA_DEVICE_MAX_CONNECTIONS=1 for better performance."
                )

            config = {
                "dist_context": self._dist_context,
            }
            allreduce_matmul_grad_overlapping_pass = new_pass(
                "allreduce_matmul_grad_overlapping", config
            )
            allreduce_matmul_grad_overlapping_pass.apply(
                [main_program], [startup_program], self._pass_context
            )

        if self.is_train:
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

        if not is_sequential_run():
            # deps for newexe
            config = {}
            config["dist_context"] = self._dist_context
            APSED_pass = new_pass(
                "auto_parallel_supplement_explicit_dependencies", config
            )
            APSED_pass.apply(
                [main_program], [startup_program], self._pass_context
            )

        if self.is_train and self._strategy.pipeline.enable:
            self._strategy.gradient_merge.enable = True
            self._strategy.gradient_merge.k_steps = (
                self._strategy.pipeline.accumulate_steps
            )
            self._strategy.gradient_merge.avg = True

        # gradient_merge is then train-only optimization
        grad_to_global_grad = {}
        if self.is_train and self._strategy.gradient_merge.enable:
            config = copy.deepcopy(self._strategy.gradient_merge.to_dict())
            config["dist_context"] = self._dist_context
            config["grad_to_global_grad"] = grad_to_global_grad
            config["pipeline_mode"] = self._strategy.pipeline.schedule_mode
            if gradient_sync_after_accumulate:
                config["params_grads"] = global_params_grads
                config["gradient_sync_after_accumulate"] = (
                    gradient_sync_after_accumulate
                )
            else:
                config["params_grads"] = params_grads
            auto_parallel_gradient_merge_pass = new_pass(
                "auto_parallel_gradient_merge_pass", config
            )
            auto_parallel_gradient_merge_pass.apply(
                [main_program], [startup_program], self._pass_context
            )

        self._check_dist_attr(
            main_program,
            self._strategy.pipeline.vpp_degree,
            self._dist_context,
        )

        enable_ir = get_flags("FLAGS_enable_pir_in_executor")[
            'FLAGS_enable_pir_in_executor'
        ]
        ir_pass_list = []
        if self.is_train and self._strategy.fused_passes.enable:
            if len(self._strategy.fused_passes.fused_passes_list) > 0:
                program_pass_list = []
                for p in self._strategy.fused_passes.fused_passes_list:
                    if enable_ir and p in (PIR_PASS + PIR_PYTHON_PASS):
                        ir_pass_list.append(p)
                    else:
                        program_pass_list.append(new_pass(p))
                pass_manager = PassManager(program_pass_list)
                pass_manager.apply([main_program], [startup_program])

        main_program._pass_opt = {}
        main_program._pass_opt['pass_list'] = ir_pass_list

        if self.is_train and self._strategy.pipeline.enable:
            enable_send_recv_overlap = (
                self._strategy.pipeline.enable_send_recv_overlap
            )
            if (
                enable_send_recv_overlap
                and int(os.getenv("CUDA_DEVICE_MAX_CONNECTIONS", "0")) != 1
            ):
                self._logger.warning(
                    "You set pipeline.enable_send_recv_overlap=True, but you did not set environment "
                    "variable CUDA_DEVICE_MAX_CONNECTIONS=1, which may leads to performance "
                    "loss. Try to export CUDA_DEVICE_MAX_CONNECTIONS=1 for better performance."
                )

            main_program._pipeline_opt = {}
            main_program._pipeline_opt["standalone_opt"] = {
                "enable_send_recv_overlap": enable_send_recv_overlap,
                "schedule_mode": self._strategy.pipeline.schedule_mode,
                "num_micro_batches": self._strategy.pipeline.accumulate_steps,
                "pp_degree": len(self._dist_context.process_meshes),
                "pp_stage": get_pp_stage(self._dist_context, rank),
                "vpp_degree": self._strategy.pipeline.vpp_degree,
                "dist_context": self._dist_context,
                "program_runtimes": self._strategy.pipeline.program_runtimes,
                "memory_limit_times": self._strategy.pipeline.memory_limit_times,
                "split_backward": self._strategy.pipeline.split_backward,
                "grad_to_global_grad": grad_to_global_grad,
            }
