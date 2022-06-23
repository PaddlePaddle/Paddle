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
import pickle
import subprocess

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

class OptimizationTuner:
    """
    generate vanilla program
    backup & restore vanilla program
    select corresponding tunning algortihm（tuner, initialize it with all information it need.
    manage while loop
        call tuner to generate trial
        apply pass
        evaluation
        check stop metric
    summary & return 
    """

    def __init__(self,
                 dist_context,
                 completer,
                 tuning_config,
                 rank,
                 ):

        self._dist_context = dist_context
        self._completer = completer
        self._tuning_config = tuning_config
        self._rank_id = rank
        self._evaluate_mode = "Profile"

        self._build_program_without_optimization()
        self._backup_context()
        self._select_tuning_algor()

    # TODO this func may be intergated into parallelizer
    def _build_program_without_optimization(self):
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
        self._dist_context.block_state.parse_backward_blocks(serial_main_program)
        self._dist_context._params_grads = params_grads

        debug_program(self._dist_context._serial_main_program, "./", "tuner_vanilla_main")
        debug_program(self._dist_context._serial_startup_program, "./", "tuner_vanilla_startup")

    def _backup_context(self):
        self._backup_main_program = self._dist_context.serial_main_program.clone(for_test=False)
        self._backup_startup_program = self._dist_context.serial_startup_program.clone(for_test=False)
        self._user_optimizer = copy.deepcopy(self._dist_context.serial_optimizer)
        self._backup_dist_tensors_for_program = copy.deepcopy(self._dist_context._dist_tensors_for_program)
        self._backup_dist_ops_for_program = copy.deepcopy(self._dist_context._dist_ops_for_program)
        self._backup_process_meshes = copy.deepcopy(self._dist_context.process_meshes)
        self._backup_distop_ctx = copy.deepcopy(self._dist_context._dist_op_context)
        self._backup_block_state = copy.deepcopy(self._dist_context.block_state)
        # self._backup_completer = copy.deepcopy(self._completer)
    
    def _get_new_env(self):
        # TODO only dependent on dist context
        # all env need to be start a new pass are member of dist context
        clear_all_process_groups()
        new_dist_context = DistributedContext()
        new_main_program = self._backup_main_program.clone(for_test=False)
        new_startup_program = self._backup_startup_program.clone(for_test=False)
        new_optimizer = copy.deepcopy(self._user_optimizer)
        new_dist_context._dist_tensors_for_program = copy.deepcopy(self._backup_dist_tensors_for_program)
        new_dist_context._dist_ops_for_program = copy.deepcopy(self._backup_dist_ops_for_program)
        for pm in self._backup_process_meshes :
            new_dist_context.add_process_mesh(pm)
        new_dist_context._dist_op_context = copy.deepcopy(self._backup_distop_ctx)
        new_dist_context._block_state = copy.deepcopy(self._backup_block_state)
        pass_context = PassContext()
        # new_completer = copy.deepcopy(self._backup_completer)
        new_completer = Completer(new_dist_context)
        return new_dist_context, new_main_program, new_startup_program, new_optimizer, pass_context, new_completer

    def _select_tuning_algor(self):

        # TODO select tuning based on user setting

        self.algorithm = ShardingStageTuner(self._dist_context._strategy, self._tuning_config)

    # FIXME reorder all the auto parallel pass and then reuse code in parallelizer
    # TODO tuner self maintain loss / params_grads
    # def _apply_pass(self, new_strategy , time = 0):
    #     startup_prog = paddle.static.Program()
    #     main_prog = paddle.static.Program()    
    #     with paddle.static.program_guard(main_prog, startup_prog):
    #         x = paddle.static.data(name='X', shape=[1000, 784], dtype='float32')
    #         y = paddle.static.data(name='Y', shape=[784, 100], dtype='float32')
    #         z = paddle.matmul(x=x, y=y)
    #     return main_prog, main_prog

    def _apply_pass(self, new_strategy , time = 0):

        dist_context, main_program, startup_program, optimizer, pass_context, completer = self._get_new_env()

        debug_program(main_program, "./", "new_env_{}_main".format(time))
        debug_program(startup_program, "./", "new_env_{}_startup".format(time))

        if new_strategy.amp:
            config = copy.deepcopy(new_strategy.amp_configs)
            config["dist_context"] = dist_context
            config["params_grads"] = self._dist_context._params_grads

            # TODO AMP Pass should not use 
            config["loss"] = self._dist_context.serial_loss
            config["input_data"] = self._dist_context.serial_feed_vars["inputs"] \
                + self._dist_context.serial_feed_vars["labels"]
            if config["use_pure_fp16"]:
                config["base_opt"] = optimizer
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
            config["loss"] = self._dist_context.serial_loss
            auto_parallel_recompute_pass = new_pass("auto_parallel_recompute",
                                                    config)
            auto_parallel_recompute_pass.apply([main_program],
                                               [startup_program],
                                               pass_context)

        # Do logical partition
        partitioner = Partitioner(dist_context, self._rank_id)
        dist_main_prog, dist_startup_prog, dist_params_grads = partitioner.partition(
            main_program, startup_program, self._dist_context._params_grads)

        # Generate optimizer
        # FIXME should be remove from apply pass after pass support optimizers
        with program_guard(dist_main_prog, dist_startup_prog):
            optimizer_ops = optimizer.apply_gradients(dist_params_grads)
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
            auto_parallel_sharding_pass.apply([dist_main_prog], [dist_startup_prog],
                                              pass_context)

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

        if self._evaluate_mode == "Profile":
            """
            use dict & python pickle
            prepare ctx
                manager and parse path

                context need to rerun the trial in another process
                main_program
                startup_program
                nccl comm 
                    ring_id -- ranks 
                    dist env 
            launch task
            """
            print("=====tuner" * 8)
            def parse_process_groups():
                group_map = {}
                all_process_groups = get_all_process_groups()
                for process_group in all_process_groups:
                    group_map[process_group.id] = process_group.ranks
                    print(process_group)
                return group_map

            # parse path
            profile_ctx = {}
            dist_env = copy.deepcopy(paddle.distributed.ParallelEnv())
            self.rank = dist_env.rank
            ctx_path = "./" + self.algorithm.get_trial_name() + ".pfcontext." + str(self.rank)
            assert isinstance(self.rank, int)
            self.device_id = dist_env.device_id 
            assert isinstance(self.device_id, int)
            self.current_endpoint = dist_env.current_endpoint
            profile_ctx['distributed_env'] = dist_env
            profile_ctx['group_map'] = parse_process_groups()

            with open(ctx_path, 'wb') as f:
                pickle.dump(profile_ctx, f, protocol=4)

            main_program_filename = "./" + self.algorithm.get_trial_name() + ".main_program_decs." + str(self.rank)
            main_binary_str = main_program.desc.serialize_to_string()
            with open(main_program_filename,  "wb") as f:
                f.write(main_binary_str)

            startup_program_filename = "./" + self.algorithm.get_trial_name() + ".startup_program_decs." + str(self.rank)
            startup_binary_str = startup_program.desc.serialize_to_string()
            with open(startup_program_filename, "wb") as f:
                f.write(startup_binary_str)

            # run profile
            if os.environ.get("WITH_COVERAGE", "OFF") == "ON":
                coverage_args = ["-m", "coverage", "run", "--branch", "-p"]
            else:
                coverage_args = []
            profile_args = " ".join(["--rank", str(self.rank), "--device_id", str(self.device_id), "--ctx_filename", ctx_path, "--main_program_filename",  main_program_filename, "--startup_program_filename",  startup_program_filename,])
            new_cmd_args = "-m paddle.distributed.auto_parallel.tuner.profiler" + " " + profile_args
            new_cmd = [sys.executable, "-u"] + coverage_args + shlex.split(new_cmd_args)
            print(new_cmd)
            new_process = subprocess.Popen(new_cmd)
            new_process.wait()
            print("=====tuner" * 8)
            assert new_process.returncode == 0, "Porfile failed !"
            print("Profile Successfully !")


        elif self._evaluate_mode == "CostModel":
            raise NotImplementedError("CostModel mode for optimization tuning is not supported yet!")
        else:
            raise NotImplementedError("invalid evaluation mode: {}".format(self._evaluate_mode))

        return 0

    
    def tune(self):

        # step1: baseline trial
        new_strategy = self.algorithm.get_baseline_trial()
        new_main_program, new_startup_program = self._apply_pass(new_strategy,  time = 0)
        last_trial_result = self._evaluate_trial(new_main_program, new_startup_program)
        debug_program(new_main_program, "./", "baseline_main")
        debug_program(new_startup_program, "./", "baseline_startup")

        # step2: while loop to find out best trial
        time = 0
        while True:
            
            # TODO TrialStatus.STOPPED
            if self.algorithm.get_status() == "STOP":
                break
            
            print("loooooop : ", time)
            new_strategy = self.algorithm.get_next_trial()
            new_main_program, new_startup_program = self._apply_pass(new_strategy)
            last_trial_result = self._evaluate_trial(new_main_program, new_startup_program)
            debug_program(new_main_program, "./", "trial_{}_main".format(time))
            debug_program(new_startup_program, "./", "trial_{}_startup".format(time))

            # TODO trial result class
            self.algorithm.update_statue(last_trial_result)
            time += 1
            
        # step3: summary the best config and return 
        self.algorithm.summary()

class ShardingStageTuner:
    """
    should inherit from a base class
    get full strategy and change partititon of it each 
    """

    def __init__(self, strategy, tuning_config):
        self._strategy = copy.deepcopy(strategy)
        self._tuning_config = tuning_config
        self.stage_range = sorted(self._tuning_config["sharding"]["stage"])
        self._sharding_configs = copy.deepcopy(strategy.sharding_configs)
        print("self._sharding_configs: ", type(self._sharding_configs))
        print("stage_range: ", self.stage_range)
        assert isinstance(self.stage_range, list)
        self.cur_idx = len(self.stage_range) - 1
        # maintain "self.status"
         
    def get_baseline_trial(self):
        return self.get_next_trial()

    def get_next_trial(self):
        print("generate next stage: ", self.stage_range[self.cur_idx])

        if self.cur_idx >= 0:
            print(self._strategy.sharding_configs["stage"])
            print(self._strategy.sharding_configs)
            self._sharding_configs["stage"] = self.stage_range[self.cur_idx]
            self._strategy.sharding_configs = self._sharding_configs
            print(self._strategy.sharding_configs["stage"])
            print(self._strategy.sharding_configs)
            assert self._strategy.sharding_configs["stage"] == self.stage_range[self.cur_idx]
            self.cur_idx -= 1
            print(str(self._strategy))
            return copy.deepcopy(self._strategy)
        else:
            return None
    
    # TODO should return a trial class and trial name should be its member
    def get_trial_name(self):
        return "Sharing_stage_{}_trial".format(self.cur_idx + 1)
        

    def get_status(self):
        if self.cur_idx >= 0:
            return "RUNNING"
        else:
            return "STOP"

    def update_statue(self, result):
        pass
    
    def summary(self):
        print(" algorithm.summary() " * 8)



    