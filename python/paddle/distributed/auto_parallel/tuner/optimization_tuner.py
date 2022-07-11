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

import os
import sys
import copy
import shlex
import pathlib
import pickle
import json
import subprocess
import traceback

import paddle
from paddle.fluid import program_guard
from paddle.fluid.backward import append_backward

from paddle.distributed.auto_parallel.dist_context import DistributedContext, get_default_distributed_context
from paddle.distributed.auto_parallel.completion import Completer
from paddle.distributed.auto_parallel.reshard import Resharder
from paddle.distributed.auto_parallel.partitioner import Partitioner
from paddle.distributed.auto_parallel.process_group import clear_all_process_groups, get_all_process_groups
from paddle.distributed.auto_parallel.utils import debug_program
from paddle.distributed.auto_parallel.utils import make_data_unshard, set_grad_var_shape
from paddle.distributed.passes import new_pass, PassContext

from .config import TuningConfig
from .algorithms import new_algorithm


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


def infer_batch_size(main_program, inputs_spec):
    input_varname = inputs_spec[0].name
    input_var = main_program.global_block().var(input_varname)
    return input_var.shape[0]


def get_metric(results):
    assert isinstance(
        results,
        dict), "results should be type of dictionary, but got {}.".format(
            type(results))
    if 'throughtput' in results and isinstance(results['throughtput'], float):
        return float(results['throughtput'])
    else:
        return -1.0


class OptimizationTuner:
    """
    OptimizationTuner is used to manage the tuning procedure of hyper-parameters (configs) 
    of Optimization Pass in AutoParallel.
    """

    def __init__(
        self,
        user_configs,
        dist_context,
        completer,
        inputs_spec,
        labels_spec,
        batch_size,
        rank,
    ):

        self._dist_context = dist_context
        self._completer = completer
        self._config = TuningConfig(user_configs, self._dist_context._strategy)
        self._rank_id = rank
        self._inputs_spec = inputs_spec
        self._labels_spec = labels_spec

        self._build_programs_without_optimization()
        self._select_tuning_algorithm()

    # TODO this func may be intergated into parallelizer
    def _build_programs_without_optimization(self):
        # NOTE (JZ-LIANG): This function is supposed to generate a compeleted program
        # with all parts like forward, backward, update, parallelism transformation,
        # only without optimization transformation. It will be update in future.

        serial_main_program = self._dist_context.serial_main_program
        serial_startup_program = self._dist_context.serial_startup_program
        serial_loss = self._dist_context.serial_loss

        with program_guard(serial_main_program, serial_startup_program):
            params_grads = append_backward(
                serial_loss, distop_context=self._dist_context.dist_op_context)
        self._completer.complete_backward_annotation(serial_main_program)
        self._dist_context.block_state.parse_backward_blocks(
            serial_main_program)
        self._dist_context._params_grads = params_grads

        if self._config.verbose:
            debug_program(self._dist_context._serial_main_program,
                          os.path.join(self._config.log_dir, "Programs"),
                          "vanilla_main")
            debug_program(self._dist_context._serial_startup_program,
                          os.path.join(self._config.log_dir, "Programs"),
                          "vanilla_startup")

    def _get_new_context(self):
        # TODO only dependent on dist context
        # all env need to be start a new pass are member of dist context
        clear_all_process_groups()
        new_dist_context = DistributedContext()
        new_dist_context._serial_main_program = self._dist_context.serial_main_program.clone(
            for_test=False)
        new_dist_context._serial_startup_program = self._dist_context.serial_startup_program.clone(
            for_test=False)

        new_dist_context._params_grads = _get_new_params_grads(
            new_dist_context.serial_main_program,
            self._dist_context.serial_main_program,
            self._dist_context._params_grads)
        new_dist_context._serial_loss = _get_new_loss(
            new_dist_context.serial_main_program,
            self._dist_context.serial_main_program,
            self._dist_context.serial_loss)

        new_dist_context._serial_optimizer = copy.deepcopy(
            self._dist_context.serial_optimizer)
        new_dist_context._dist_tensors_for_program = copy.deepcopy(
            self._dist_context._dist_tensors_for_program)
        new_dist_context._dist_ops_for_program = copy.deepcopy(
            self._dist_context._dist_ops_for_program)
        for pm in self._dist_context.process_meshes:
            new_dist_context.add_process_mesh(pm)
        new_dist_context._dist_op_context = copy.deepcopy(
            self._dist_context._dist_op_context)
        new_dist_context._block_state = copy.deepcopy(
            self._dist_context.block_state)
        pass_context = PassContext()
        new_completer = Completer(new_dist_context)

        return new_dist_context, pass_context, new_completer

    def _select_tuning_algorithm(self):

        selected_passes_set = self._config.tuning_passes_name
        algorithm_name = "_".join(sorted(selected_passes_set))
        self.algorithm = new_algorithm(algorithm_name, self._config)

    def _apply_optimization(self, new_strategy):

        dist_context, pass_context, completer = self._get_new_context()

        main_program = dist_context.serial_main_program
        startup_program = dist_context.serial_startup_program

        if new_strategy.amp:
            config = copy.deepcopy(new_strategy.amp_configs)
            config["dist_context"] = dist_context
            config["params_grads"] = dist_context._params_grads

            # TODO AMP Pass should not use loss var
            config["loss"] = dist_context.serial_loss
            config["input_data"] = self._dist_context.serial_feed_vars["inputs"] \
                + self._dist_context.serial_feed_vars["labels"]
            if config["use_pure_fp16"]:
                config["base_opt"] = dist_context.optimizer
                auto_parallel_fp16_pass = new_pass("auto_parallel_fp16", config)
                auto_parallel_fp16_pass.apply([main_program], [startup_program],
                                              pass_context)
            else:
                auto_parallel_amp_pass = new_pass("auto_parallel_amp", config)
                auto_parallel_amp_pass.apply([main_program], [startup_program],
                                             pass_context)

        if new_strategy.recompute:
            config = copy.deepcopy(new_strategy.recompute_configs)
            config["dist_context"] = dist_context
            config["no_grad_set"] = None
            config["loss"] = dist_context.serial_loss
            auto_parallel_recompute_pass = new_pass("auto_parallel_recompute",
                                                    config)
            auto_parallel_recompute_pass.apply([main_program],
                                               [startup_program], pass_context)
            print("apply recompute pass !" * 8)

        # Do logical partition
        partitioner = Partitioner(dist_context, self._rank_id)
        dist_main_prog, dist_startup_prog, dist_params_grads = partitioner.partition(
            main_program, startup_program, dist_context._params_grads)

        # Generate optimizer
        # FIXME should be remove from apply pass after pass support optimizers
        with program_guard(dist_main_prog, dist_startup_prog):
            optimizer_ops = dist_context.serial_optimizer.apply_gradients(
                dist_params_grads)
        completer.complete_update_annotation(dist_main_prog)

        # Do reshard process
        set_grad_var_shape(dist_main_prog, dist_context)
        resharder = Resharder(dist_main_prog, dist_startup_prog, self._rank_id,
                              dist_context, dist_params_grads)
        resharder.reshard()

        if new_strategy.sharding:
            config = copy.deepcopy(new_strategy.sharding_configs)
            config["dist_context"] = dist_context
            config["params_grads"] = dist_params_grads
            config["global_rank"] = self._rank_id
            auto_parallel_sharding_pass = new_pass("auto_parallel_sharding",
                                                   config)
            auto_parallel_sharding_pass.apply([dist_main_prog],
                                              [dist_startup_prog], pass_context)

        # recompute is then train-only optimization
        if new_strategy.gradient_merge:
            config = copy.deepcopy(new_strategy.gradient_merge_configs)
            config["dist_context"] = dist_context
            config["params_grads"] = dist_params_grads
            auto_parallel_gradient_merge_pass = new_pass(
                "auto_parallel_gradient_merge_pass", config)
            auto_parallel_gradient_merge_pass.apply([dist_main_prog],
                                                    [dist_startup_prog],
                                                    pass_context)
        return dist_main_prog, dist_startup_prog

    # TODO a profiler class
    def _evaluate_trial(self, main_program, startup_program):

        if self._config.mode == "PROFILE":

            print("strat evaluating: {} .".format(
                self.algorithm.get_trial_name()))

            # Make workspace
            work_dir = os.path.join(self._config.log_dir,
                                    self.algorithm.get_trial_name())
            if not os.path.exists(work_dir):
                if paddle.distributed.get_rank() == 0:
                    pathlib.Path(work_dir).mkdir(parents=True, exist_ok=True)
                else:
                    while not os.path.exists(work_dir):
                        pass

            # Prepare Profile Context
            profile_ctx = {}
            dist_env = copy.deepcopy(paddle.distributed.ParallelEnv())
            rank = dist_env.rank
            ctx_filename = "profile_ctx." + str(rank)
            ctx_path = os.path.join(work_dir, ctx_filename)
            assert isinstance(rank, int)
            device_id = dist_env.device_id
            assert isinstance(device_id, int)
            profile_ctx['distributed_env'] = dist_env
            profile_ctx['group_map'] = parse_process_groups()
            profile_ctx["loss_var_name"] = self._dist_context.serial_loss.name
            profile_ctx["batch_size"] = infer_batch_size(
                main_program, self._inputs_spec)
            profile_ctx[
                "main_program_decs"] = main_program.desc.serialize_to_string()
            profile_ctx[
                "startup_program_decs"] = startup_program.desc.serialize_to_string(
                )
            result_path = os.path.join(work_dir, "result.json")
            profile_ctx["result_filename"] = result_path

            with open(ctx_path, 'wb') as f:
                pickle.dump(profile_ctx, f, protocol=4)

            if self._config.verbose:
                debug_program(main_program, work_dir, "main_program")
                debug_program(startup_program, work_dir, "startup_program")
                #TODO dump cur pass config to file

            # Run profile
            if os.environ.get("WITH_COVERAGE", "OFF") == "ON":
                coverage_args = ["-m", "coverage", "run", "--branch", "-p"]
            else:
                coverage_args = []
            profile_args = " ".join([
                "--rank",
                str(rank),
                "--device_id",
                str(device_id),
                "--ctx_filename",
                ctx_path,
            ])
            cmd_args = "-m paddle.distributed.auto_parallel.tuner.profiler" + " " + profile_args
            cmd = [sys.executable, "-u"] + coverage_args + shlex.split(cmd_args)
            # TODO if any rank hang or fail, kill all, otherwise
            parent_env = copy.copy(os.environ.copy())
            new_env = {
                "FLAGS_USE_STANDALONE_EXECUTOR": "False",
            }
            new_env.update(parent_env)
            print("executing cmd: {} .".format(" ".join(cmd)))
            new_process = subprocess.Popen(cmd, env=new_env)

            with open(os.path.join(work_dir, "stdout.log" + str(rank)),
                      "wb") as out, open(
                          os.path.join(work_dir, "stderr.log" + str(rank)),
                          "wb") as err:
                result = subprocess.Popen(cmd, stdout=out, stderr=err)
                result.wait()
                out.flush()
                err.flush()
                os.fsync(out)
                os.fsync(err)

            # load results
            with open(result_path, 'r') as fp:
                results = json.load(fp)

            print("end evaluating: {} .".format(
                self.algorithm.get_trial_name()))

            return results

        elif self._config.mode == "COSTMODEL":
            raise NotImplementedError(
                "COSTMODEL mode for optimization tuning is not supported yet!")
        else:
            raise NotImplementedError("invalid evaluation mode: {}".format(
                self._config.mode))

    def tune(self):

        # step1: collect model info which might
        self.algorithm.collect_model_info(
            self._dist_context.serial_main_program,
            self._dist_context.serial_startup_program)

        # step2: main while loop
        i = 0
        best_i = 0
        best_metric = None
        try:
            while i < self._config.max_num_trial:

                # TODO TrialStatus.STOPPED
                if self.algorithm.status() == "STOP":
                    break

                new_strategy = self.algorithm.next_trial()
                new_main_program, new_startup_program = self._apply_optimization(
                    new_strategy)
                last_results = self._evaluate_trial(new_main_program,
                                                    new_startup_program)

                if best_metric == None or get_metric(
                        last_results) < best_metric:
                    best_metric = get_metric(last_results)
                    best_i = i

                # TODO trial result class
                self.algorithm.update(last_results)

                i += 1
                if self._config.early_stop and self._config.early_stop <= i - best_i:
                    print(
                        "Early stop the Tuning since there is no better trial found within [{}] trials"
                        .format(self._config.early_stop))
                    break
        except:
            print("Tuner got Error: {}".format(sys.exc_info()[0]))
            print(sys.exc_info()[1])
            traceback.print_tb(sys.exc_info()[2])

        # step3: summary the best config and return
        self.algorithm.summary()
