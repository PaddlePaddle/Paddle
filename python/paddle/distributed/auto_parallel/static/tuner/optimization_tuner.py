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
import json
import logging

# import yaml
import os
import pathlib
import pickle
import shlex
import shutil
import subprocess
import sys
import time

import paddle
from paddle.distributed.auto_parallel.static.completion import Completer
from paddle.distributed.auto_parallel.static.dist_context import (
    DistributedContext,
)
from paddle.distributed.auto_parallel.static.partitioner import Partitioner
from paddle.distributed.auto_parallel.static.process_group import (
    clear_all_process_groups,
    get_all_process_groups,
    new_process_group,
)
from paddle.distributed.auto_parallel.static.reshard import Resharder
from paddle.distributed.auto_parallel.static.utils import debug_program
from paddle.distributed.passes import PassContext, new_pass
from paddle.static import append_backward, program_guard

from ..utils import get_logger
from .algorithms import new_algorithm
from .config import TuningConfig
from .trial import TrialStatus


def _get_new_params_grads(target_program, ref_program, ref_params_grads):
    ref_block = ref_program.global_block()
    target_block = target_program.global_block()
    target_params_grads = []

    for p, g in ref_params_grads:
        # NOTE grad var might not be generated
        assert ref_block.has_var(p.name)
        assert target_block.has_var(p.name)
        new_p = target_block.var(p.name)
        if g:
            new_g = target_block.var(g.name)
        else:
            new_g = None

        target_params_grads.append((new_p, new_g))

    return target_params_grads


def _get_new_loss(target_program, ref_program, loss):
    ref_block = ref_program.global_block()
    target_block = target_program.global_block()
    assert ref_block.has_var(loss.name)

    return target_block.var(loss.name)


def parse_process_groups():
    group_map = {}
    all_process_groups = get_all_process_groups()
    for process_group in all_process_groups:
        group_map[process_group.id] = process_group.ranks
    return group_map


def get_metric(results):
    assert isinstance(
        results, dict
    ), f"results should be type of dictionary, but got {type(results)}."
    if 'Throughput' in results and isinstance(results['Throughput'], float):
        return float(results['Throughput'])
    else:
        return -1.0


def parse_results(results):
    if results['Throughput'] > 0:
        return "Throughput: {} step / s.".format(results['Throughput'])
    et = results.get("ErrorType", None)
    if et == "ResourceExhaustedError":
        return "Fail with OOM"
    else:
        return "Fail with UNKNOWN ERROR"


# TODO only dependent on dist context
# all env need to be start a new pass are member of dist context
def _copy_context(ref_dist_context):
    # clear all process groups and recover the world process group
    clear_all_process_groups()
    ranks = []
    for process_mesh in ref_dist_context._process_meshes:
        ranks.extend(process_mesh.process_ids)
    new_process_group(list(set(ranks)))

    new_dist_context = DistributedContext()
    new_dist_context._serial_main_program = (
        ref_dist_context.serial_main_program.clone(for_test=False)
    )
    new_dist_context._serial_startup_program = (
        ref_dist_context.serial_startup_program.clone(for_test=False)
    )

    # mapping variable into new dist context
    if getattr(ref_dist_context, '_params_grads', None):
        new_dist_context._params_grads = _get_new_params_grads(
            new_dist_context.serial_main_program,
            ref_dist_context.serial_main_program,
            ref_dist_context._params_grads,
        )
    new_dist_context._serial_loss = _get_new_loss(
        new_dist_context.serial_main_program,
        ref_dist_context.serial_main_program,
        ref_dist_context.serial_loss,
    )

    for key, var_list in ref_dist_context._serial_feed_vars.items():
        new_var_list = []
        for var in var_list:
            block_idx = var.block.idx
            var_name = var.name
            var = new_dist_context._serial_main_program.blocks[
                block_idx
            ]._var_recursive(var_name)
            new_var_list.append(var)
        new_dist_context._serial_feed_vars[key] = new_var_list

    for key, var_list in ref_dist_context._serial_fetch_vars.items():
        new_var_list = []
        # metrics is a list of list
        if key == "metrics":
            for inner_var_list in var_list:
                new_inner_var_list = []
                for var in inner_var_list:
                    block_idx = var.block.idx
                    var_name = var.name
                    var = new_dist_context._serial_main_program.blocks[
                        block_idx
                    ]._var_recursive(var_name)
                    new_inner_var_list.append(var)
                new_var_list.append(new_inner_var_list)
        else:
            for var in var_list:
                block_idx = var.block.idx
                var_name = var.name
                var = new_dist_context._serial_main_program.blocks[
                    block_idx
                ]._var_recursive(var_name)
                new_var_list.append(var)
        new_dist_context._serial_fetch_vars[key] = new_var_list

    # copy information in forward and backward
    new_dist_context._serial_optimizer = copy.deepcopy(
        ref_dist_context.serial_optimizer
    )
    new_dist_context._dist_tensors_for_program = copy.deepcopy(
        ref_dist_context._dist_tensors_for_program
    )
    new_dist_context._dist_ops_for_program = copy.deepcopy(
        ref_dist_context._dist_ops_for_program
    )
    for pm in ref_dist_context.process_meshes:
        new_dist_context.add_process_mesh(pm)
    new_dist_context._dist_op_context = copy.deepcopy(
        ref_dist_context._dist_op_context
    )
    new_dist_context._block_state = copy.deepcopy(ref_dist_context.block_state)

    return new_dist_context


class OptimizationTuner:
    """
    OptimizationTuner is used to manage the tuning procedure of hyper-parameters (configs)
    of Optimization Pass in AutoParallel.
    """

    def __init__(
        self,
        dist_context,
        dataset,
        inputs_spec,
        labels_spec,
        batch_size,
        rank,
    ):
        self._config = TuningConfig(dist_context.strategy)
        # should not modify dist context from calling function
        self._baseline_dist_context = _copy_context(dist_context)
        self._baseline_completer = Completer(self._baseline_dist_context)

        self._rank = rank
        self._inputs_spec = inputs_spec
        self._labels_spec = labels_spec
        self._dataset = dataset
        self._batch_size = batch_size

        self._finished_trials = []
        self._best_metric = None
        self._best_iter = float("-inf")

        self._logger = get_logger(logging.INFO)

        self._build_programs_without_optimization()
        self._select_tuning_algorithm()

    @property
    def project_dir(self):
        dirname = self._config.project_dir
        if not os.path.exists(dirname):
            if self.rank == 0:
                pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
        return dirname

    @property
    def rank(self):
        return self._rank

    @property
    def device_id(self):
        return paddle.distributed.ParallelEnv().device_id

    # TODO Generate complete program with all parts like forward, backward, update
    # as well as parallelism transformation.
    def _build_programs_without_optimization(self):
        serial_main_program = self._baseline_dist_context.serial_main_program
        serial_startup_program = (
            self._baseline_dist_context.serial_startup_program
        )
        serial_loss = self._baseline_dist_context.serial_loss

        with program_guard(serial_main_program, serial_startup_program):
            params_grads = append_backward(
                serial_loss,
                distop_context=self._baseline_dist_context.dist_op_context,
            )

        self._baseline_completer.complete_backward_annotation(
            serial_main_program
        )
        self._baseline_dist_context.block_state.parse_backward_blocks(
            serial_main_program
        )
        self._baseline_dist_context._params_grads = params_grads

        if self._config.debug:
            baseline_dir = os.path.join(self.project_dir, "baseline")
            if not os.path.exists(baseline_dir):
                pathlib.Path(baseline_dir).mkdir(parents=True, exist_ok=True)
            debug_program(
                self._baseline_dist_context._serial_main_program,
                baseline_dir,
                "main",
            )
            debug_program(
                self._baseline_dist_context._serial_startup_program,
                baseline_dir,
                "startup",
            )

    def _select_tuning_algorithm(self):
        selected_passes_set = self._config.tuning_passes_name
        algorithm_name = "_".join(sorted(selected_passes_set))
        self._algorithm = new_algorithm(algorithm_name, self._config)

    def _apply_optimization(self, trial):
        new_strategy = trial.space
        dist_context = _copy_context(self._baseline_dist_context)
        pass_context = PassContext()
        completer = Completer(dist_context)

        main_program = dist_context.serial_main_program
        startup_program = dist_context.serial_startup_program

        # applying optimization pass
        if new_strategy.amp.enable:
            config = copy.deepcopy(new_strategy.amp.to_dict())
            config["dist_context"] = dist_context
            config["params_grads"] = dist_context._params_grads
            # TODO AMP Pass should not use loss var
            config["loss"] = dist_context.serial_loss
            config["input_data"] = (
                self._baseline_dist_context.serial_feed_vars["inputs"]
                + self._baseline_dist_context.serial_feed_vars["labels"]
            )
            if config["dtype"] == "float16" and config["level"] == "o2":
                config["base_opt"] = dist_context.serial_optimizer
                auto_parallel_fp16_pass = new_pass("auto_parallel_fp16", config)
                auto_parallel_fp16_pass.apply(
                    [main_program], [startup_program], pass_context
                )
                dist_context._serial_loss = auto_parallel_fp16_pass.get_loss()
            else:
                auto_parallel_amp_pass = new_pass("auto_parallel_amp", config)
                auto_parallel_amp_pass.apply(
                    [main_program], [startup_program], pass_context
                )
                dist_context._serial_loss = auto_parallel_amp_pass.get_loss()

        if new_strategy.recompute.enable:
            config = copy.deepcopy(new_strategy.recompute.to_dict())
            config["dist_context"] = dist_context
            config["no_grad_set"] = None
            config["loss"] = dist_context.serial_loss
            auto_parallel_recompute_pass = new_pass(
                "auto_parallel_recompute", config
            )
            auto_parallel_recompute_pass.apply(
                [main_program], [startup_program], pass_context
            )

        # Do logical partition
        partitioner = Partitioner(dist_context, self.rank)
        (
            dist_main_prog,
            dist_startup_prog,
            dist_params_grads,
        ) = partitioner.partition(
            main_program, startup_program, dist_context._params_grads
        )

        # Generate optimizer
        # FIXME should be remove from apply pass after pass support optimizers
        with program_guard(dist_main_prog, dist_startup_prog):
            with dist_main_prog.switch_name_generator_guard("opt_"):
                optimizer_ops = dist_context.serial_optimizer.apply_gradients(
                    dist_params_grads
                )
        completer.complete_update_annotation(dist_main_prog)

        resharder = Resharder(
            dist_main_prog,
            dist_startup_prog,
            self.rank,
            dist_context,
            dist_params_grads,
        )
        resharder.reshard()

        config = {}
        config["dist_context"] = dist_context
        config["global_rank"] = self.rank
        config["use_sharding"] = new_strategy.sharding.enable
        dp_pass = new_pass("auto_parallel_data_parallel_optimization", config)
        dp_pass.apply([dist_main_prog], [dist_startup_prog], pass_context)

        if new_strategy.sharding.enable:
            config = copy.deepcopy(new_strategy.sharding.to_dict())
            config["dist_context"] = dist_context
            config["params_grads"] = dist_params_grads
            config["global_rank"] = self.rank
            auto_parallel_sharding_pass = new_pass(
                "auto_parallel_sharding", config
            )
            auto_parallel_sharding_pass.apply(
                [dist_main_prog], [dist_startup_prog], pass_context
            )
            dist_params_grads = pass_context.get_attr("params_grads")

        # gradient clip
        config = copy.deepcopy(new_strategy.sharding.to_dict())
        config["dist_context"] = dist_context
        config["params_grads"] = dist_params_grads
        config["rank_id"] = self.rank
        auto_parallel_clip_pass = new_pass("auto_parallel_grad_clip", config)
        auto_parallel_clip_pass.apply(
            [dist_main_prog], [dist_startup_prog], pass_context
        )

        if new_strategy.gradient_merge.enable:
            config = copy.deepcopy(new_strategy.gradient_merge.to_dict())
            config["dist_context"] = dist_context
            config["params_grads"] = dist_params_grads
            auto_parallel_gradient_merge_pass = new_pass(
                "auto_parallel_gradient_merge_pass", config
            )
            auto_parallel_gradient_merge_pass.apply(
                [dist_main_prog], [dist_startup_prog], pass_context
            )
        trial.main_program, trial.startup_program = (
            dist_main_prog,
            dist_startup_prog,
        )
        return trial

    def _get_profile_context(self, trial, result_path):
        profile_ctx = {}

        profile_ctx['distributed_env'] = copy.deepcopy(
            paddle.distributed.ParallelEnv()
        )
        profile_ctx['group_map'] = parse_process_groups()
        profile_ctx[
            "loss_var_name"
        ] = self._baseline_dist_context.serial_loss.name
        profile_ctx[
            "main_program_decs"
        ] = trial.main_program.desc.serialize_to_string()
        profile_ctx[
            "startup_program_decs"
        ] = trial.startup_program.desc.serialize_to_string()
        self._dataset.batch_size = self._batch_size
        self._dataset.input_names = self._get_input_names()

        profile_ctx["dataset"] = self._dataset
        profile_ctx["result_filename"] = result_path

        return profile_ctx

    def _get_input_names(self):
        input_names = []
        for input_spec in self._inputs_spec[:] + self._labels_spec[:]:
            input_names.append(input_spec.name)
        return input_names

    def _launch_profile(self, ctx_path, trial_dir):
        if os.environ.get("WITH_COVERAGE", "OFF") == "ON":
            coverage_args = ["-m", "coverage", "run", "--branch", "-p"]
        else:
            coverage_args = []

        profile_args = " ".join(
            [
                "--rank",
                str(self.rank),
                "--device_id",
                str(self.device_id),
                "--ctx_filename",
                ctx_path,
                "--profile_start_step",
                str(self._config.profile_start_step),
                "--profile_end_step",
                str(self._config.profile_end_step),
            ]
        )
        cmd_args = (
            "-m paddle.distributed.auto_parallel.static.tuner.profiler"
            + " "
            + profile_args
        )
        cmd = [sys.executable, "-u"] + coverage_args + shlex.split(cmd_args)

        parent_env = copy.copy(os.environ.copy())
        # env flags need for profile
        new_env = {}
        new_env.update(parent_env)

        # TODO if any rank hang or fail, kill all processes
        self._logger.debug("Executing cmd:\n{} .".format(" ".join(cmd)))
        # new_process = subprocess.Popen(cmd, env=new_env)
        with open(
            os.path.join(trial_dir, "stdout.log" + str(self.rank)), "wb"
        ) as out, open(
            os.path.join(trial_dir, "stderr.log" + str(self.rank)), "wb"
        ) as err:
            result = subprocess.Popen(cmd, stdout=out, stderr=err, env=new_env)
            result.wait()
            out.flush()
            err.flush()
            os.fsync(out)
            os.fsync(err)

    def _profile_trial(self, trial):
        # Making working directory
        trial_dir = self._get_trial_dir(trial)
        if not os.path.exists(trial_dir):
            if self.rank == 0:
                pathlib.Path(trial_dir).mkdir(parents=True, exist_ok=True)
            else:
                while not os.path.exists(trial_dir):
                    pass
        ctx_filename = "profile_ctx." + str(self.rank)
        ctx_path = os.path.join(trial_dir, ctx_filename)
        result_path = os.path.join(trial_dir, "result.json")

        # Prepare Profile Context
        profile_ctx = self._get_profile_context(trial, result_path)
        with open(ctx_path, 'wb') as f:
            pickle.dump(profile_ctx, f, protocol=4)

        if self._config.debug:
            debug_program(trial.main_program, trial_dir, "main_program")
            debug_program(trial.startup_program, trial_dir, "startup_program")

        # Run
        self._launch_profile(ctx_path, trial_dir)

        # Load results
        try:
            with open(result_path, 'r') as fp:
                results = json.load(fp)
            return results
        except FileNotFoundError:
            Error_results = {"Throughput": -1, "ErrorType": 'FatalError'}
            return Error_results

    def _evaluate_trial(self, trial):
        self._logger.info(f"Trial {trial.name} evaluation start.")
        self._apply_optimization(trial)

        if self._config.mode == "PROFILE":
            results = self._profile_trial(trial)

        elif self._config.mode == "COSTMODEL":
            raise NotImplementedError(
                "COSTMODEL mode for optimization tuning is not supported yet!"
            )
        else:
            raise NotImplementedError(
                f"invalid evaluation mode: {self._config.mode}"
            )

        self._logger.info(
            f"Trial {trial.name} evaluation finish with {parse_results(results)}."
        )
        return results

    def _update(self, i, trial, results):
        self._finished_trials.append(trial)

        cur_metric = get_metric(results)
        if self._best_metric is None or cur_metric > self._best_metric:
            self._best_metric = cur_metric
            self._best_iter = i

    def _get_trial_dir(self, trial):
        return os.path.join(self.project_dir, trial.name)

    def get_best_config(self):
        """
        Return the best optimization configuration found in the tuning.

        Returns:
            A object of fleet.DistributedStrategy with best configuration.
        """
        assert self._best_iter >= 0, "The best configuration is not found yet !"
        best_trial = self._finished_trials[self._best_iter]
        return self._algorithm.get_config_from_trial(best_trial)

    def summary(self):
        """
        Display tuning result summary.
        """
        # TODO summary with the trial_name with metric_of_trial
        best_trial = self._finished_trials[self._best_iter]
        summary_ = f"""
Tuning Result Summary
Run total {len(self._finished_trials)} trials with {(time.time() - self._tuning_start_time) / 60} min.
The best trial is: [{best_trial.name}], whose configuration is following:
        """
        summary_ += "\n" + best_trial.summary() + "\n"
        self._logger.info(summary_)
        with open(os.path.join(self.project_dir, "summary.txt"), "w+") as fw:
            for line in summary_.split("\n"):
                fw.write(line + "\n")

        # full_strategy = self.get_best_config()
        # path = os.path.join(self.project_dir, "tuned_dist_strategy.yaml")
        # with open(path, 'w') as outfile:
        #     yaml.dump(full_strategy, outfile, default_flow_style=False)

    def clear(self):
        """
        Clear the temporary file generated in tuning procedure.
        """
        # TODO clear up zombie process created by tuning
        if not self._config.debug:
            for trial in self._finished_trials:
                trial_dir = self._get_trial_dir(trial)
                shutil.rmtree(trial_dir, ignore_errors=True)

    def tune(self):
        """
        Performs the search for best hyperparameter configurations
        for the selected optimization pass(es).
        """

        # step1: collect model info which might be used for
        # pruning the search space of the algorithm
        self._tuning_start_time = time.time()
        self._algorithm.collect_model_info(
            self._baseline_dist_context.serial_main_program,
            self._baseline_dist_context.serial_startup_program,
        )

        # main search loop
        i = 0
        while i < self._config.max_num_trial:
            # step2: create a new trial
            trial = self._algorithm.next_trial()

            if trial.status == TrialStatus.STOPPED:
                break

            # step3: evaluate the trial
            results = self._evaluate_trial(trial)

            # step4: update the algorithm with last result,
            # which could be used by algorithm to pruning the
            # remaining search space.
            self._algorithm.update(results)
            self._update(i, trial, results)

            # early stop
            i += 1
            if (
                self._config.early_stop
                and self._config.early_stop <= i - self._best_iter
            ):
                self._logger.info(
                    f"Early stop the Tuning since there is no better trial found within [{self._config.early_stop}] trials"
                )
                break

        # step5: summary the best config and return
        self.summary()

        self.clear()
