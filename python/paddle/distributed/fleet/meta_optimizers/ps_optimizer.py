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

from paddle import fluid
import paddle.distributed.passes
from .meta_optimizer_base import MetaOptimizerBase
from paddle.fluid import core
import subprocess
import re
import os
import platform
from paddle.distributed.ps.utils.public import *
from ..base.private_helper_function import wait_server_ready


class ParameterServerOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(ParameterServerOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = []
        self.context = {}

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer,
                        user_defined_strategy):
        super(ParameterServerOptimizer, self)._set_basic_info(
            loss, role_maker, user_defined_optimizer, user_defined_strategy)

    def _init_ps_pass_context(self, loss, startup_program):
        # trainer
        self.context["env"] = get_dist_env()

        self.context['min_block_size'] = 81920
        self.context['origin_main_program'] = loss.block.program
        self.context['origin_startup_program'] = startup_program

        self.context['cloned_main'] = loss.block.program.clone()
        self.context['cloned_startup'] = startup_program.clone()

        self.context['user_defined_strategy'] = self.user_defined_strategy
        self.context['trainer'] = TrainerRuntimeConfig(
            self.user_defined_strategy)
        self.context['ps_mode'] = self.context['trainer'].mode

        self.context['role_maker'] = self.role_maker
        self.context[
            'is_heter_ps_mode'] = self.role_maker._is_heter_parameter_server_mode
        self.context['is_worker'] = self.role_maker._is_worker()
        self.context['is_server'] = self.role_maker._is_server()
        self.context['is_heter_worker'] = self.role_maker._is_heter_worker()

        self.context['use_ps_gpu'] = self.user_defined_strategy.a_sync_configs[
            "use_ps_gpu"]
        self.context[
            'lr_decay_steps'] = self.user_defined_strategy.a_sync_configs[
                "lr_decay_steps"]
        self.context['k_steps'] = self.user_defined_strategy.a_sync_configs[
            "k_steps"]
        self.context[
            'launch_barrier'] = self.user_defined_strategy.a_sync_configs[
                "launch_barrier"]

        self.context['launch_barrier_flag'] = int(
            os.getenv("FLAGS_LAUNCH_BARRIER", "1"))

        build_var_distributed(self.context)

        # server 
        self.context['_main_server'] = fluid.Program()
        self.context['_startup_server'] = fluid.Program()
        self.context['tensor_table'] = {}

    def _is_graph_out(self):
        return False

    def _can_apply(self):
        if self.context['role_maker']._is_collective or self.context[
                'k_steps'] < 0:
            return False
        return True

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        self.inner_opt.minimize(loss, startup_program, parameter_list,
                                no_grad_set)
        self.init_ps_pass_context(loss, startup_program)
        ps_builder = PsProgramBuilderFactory()._create_ps_program_builder(
            self.context)
        ps_builder.build_programs()
        return None, None

    def _can_apply_geo(self, program):
        def get_sys_free_mem():
            plat = platform.system()
            if platform.system() == "Darwin":
                vm = subprocess.Popen(
                    ['vm_stat'], stdout=subprocess.PIPE).communicate()[0]
                # Process vm_stat
                vmLines = vm.split('\n')
                sep = re.compile(r':[\s]+')
                vmStats = {}
                for row in range(1, len(vmLines) - 2):
                    rowText = vmLines[row].strip()
                    rowElements = sep.split(rowText)
                    vmStats[(rowElements[0]
                             )] = int(rowElements[1].strip(r'\.')) * 4096
                return vmStats["Pages free"]
            elif platform.system() == "Linux":
                mems = {}
                with open('/proc/meminfo', 'rb') as f:
                    for line in f:
                        fields = line.split()
                        mems[fields[0]] = int(fields[1]) * 1024
                free = mems[b'MemFree:']
                return free
            else:
                raise ValueError(
                    "%s platform is unsupported is parameter server optimizer" %
                    (platform.system()))

        if not isinstance(self.inner_opt, fluid.optimizer.SGDOptimizer):
            return False

        free = get_sys_free_mem()
        processed_var_names = set(["@EMPTY@"])
        param_memory_size = 0
        for varname in program.global_block().vars:
            var = program.global_block().vars[varname]
            if not var.persistable or var.desc.type(
            ) != core.VarDesc.VarType.LOD_TENSOR:
                continue
            set_var_lod_type(var)
            param_memory_size += get_var_mem_size(var)
            processed_var_names.add(varname)

        upper_mem_use = param_memory_size * 5.0

        program_tmp_vars = dict()
        eval_batch_size = 1024
        for op in program.global_block().ops:
            for var_name in op.output_arg_names:
                if var_name in processed_var_names:
                    continue
                processed_var_names.add(var_name)
                var = program.global_block().vars[var_name]

                if var.desc.type() != core.VarDesc.VarType.LOD_TENSOR:
                    continue

                data_count = 1
                neg_dim_count = 0
                for x in var.shape:
                    if x < 0:
                        if neg_dim_count >= 1:
                            raise ValueError(
                                "Var %s has more than one negative dim." %
                                (var_name))
                        neg_dim_count += 1
                        data_count *= (-x)
                    else:
                        data_count *= x
                program_tmp_vars[var_name] = (
                    data_count, neg_dim_count,
                    vars_metatools.dtype_to_size[var.dtype])

        for varname in program_tmp_vars:
            data_count, neg_dim_count, type_size = program_tmp_vars[varname]
            if neg_dim_count == 1:
                data_count *= eval_batch_size
            var_memory = data_count * type_size
            upper_mem_use += var_memory

        if upper_mem_use < free:
            return True
        else:
            return False

    def _enable_strategy(self, dist_strategy, context):
        if dist_strategy.a_sync_configs["k_steps"] >= 0:
            return
        dist_strategy.a_sync = True
        is_geo = self._can_apply_geo(context["origin_main_program"])
        dist_strategy.a_sync_configs["k_steps"] = 800 if is_geo else 0

    def _disable_strategy(self, dist_strategy):
        dist_strategy.a_sync = False
        dist_strategy.a_sync_configs["k_steps"] = -1
