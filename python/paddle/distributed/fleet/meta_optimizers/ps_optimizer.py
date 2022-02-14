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
from paddle.distributed.passes import PassContext
from ..base.private_helper_function import wait_server_ready
from paddle.distributed.ps.utils.ps_factory import PsProgramBuilderFactory


class ParameterServerOptimizer(MetaOptimizerBase):
    def __init__(self, optimizer):
        super(ParameterServerOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        # we do not allow meta optimizer to be inner optimizer currently
        self.meta_optimizers_white_list = []
        self.pass_ctx = PassContext()

    def _set_basic_info(self, loss, role_maker, user_defined_optimizer,
                        user_defined_strategy):
        super(ParameterServerOptimizer, self)._set_basic_info(
            loss, role_maker, user_defined_optimizer, user_defined_strategy)

    def _init_ps_pass_context(self, loss, startup_program):
        attrs = {}
        # trainer
        attrs["env"] = get_dist_env()

        attrs['loss'] = loss
        attrs['min_block_size'] = 81920
        attrs['origin_main_program'] = loss.block.program
        attrs['origin_startup_program'] = startup_program

        attrs['cloned_main'] = attrs['origin_main_program'].clone()
        attrs['cloned_startup'] = attrs['origin_startup_program'].clone()

        attrs['user_defined_strategy'] = self.user_defined_strategy
        attrs['trainer'] = TrainerRuntimeConfig(self.user_defined_strategy)
        attrs['ps_mode'] = attrs['trainer'].mode

        attrs['role_maker'] = self.role_maker
        attrs[
            'is_heter_ps_mode'] = self.role_maker._is_heter_parameter_server_mode
        attrs['is_worker'] = self.role_maker._is_worker()
        attrs['is_server'] = self.role_maker._is_server()
        attrs['is_heter_worker'] = self.role_maker._is_heter_worker()

        attrs['use_ps_gpu'] = self.user_defined_strategy.a_sync_configs[
            "use_ps_gpu"]
        attrs['lr_decay_steps'] = self.user_defined_strategy.a_sync_configs[
            "lr_decay_steps"]
        attrs['k_steps'] = self.user_defined_strategy.a_sync_configs["k_steps"]
        attrs['launch_barrier'] = self.user_defined_strategy.a_sync_configs[
            "launch_barrier"]

        attrs['launch_barrier_flag'] = int(
            os.getenv("FLAGS_LAUNCH_BARRIER", "1"))

        build_var_distributed(attrs)

        # server 
        attrs['_main_server'] = fluid.Program()
        attrs['_startup_server'] = fluid.Program()
        attrs['tensor_table'] = {}

        self.pass_ctx._attrs = attrs

    def _is_graph_out(self):
        return False

    def _can_apply(self):
        if self._attrs['role_maker']._is_collective or self._attrs[
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
        if startup_program == None:
            startup_program = paddle.static.default_startup_program()
        self._init_ps_pass_context(loss, startup_program)
        ps_builder = PsProgramBuilderFactory()._create_ps_program_builder(
            self.pass_ctx)
        ps_builder._build_programs()
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
