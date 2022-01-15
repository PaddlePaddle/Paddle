# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


@register_pass("ps_gpu_pass")
class PsGpuPass(PassBase):
    def __init__(self):
        super(ShardingPass, self).__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        pass


@register_pass("ps_transpile_pass")
class PsTranspilePass(PassBase):
    def __init__(self):
        super(ShardingPass, self).__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        pass


@register_pass("ps_add_lr_decay_table_pass")
class PsAddLrDecayTablePass(PassBase):
    def __init__(self):
        super(ShardingPass, self).__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        pass


@register_pass("ps_append_send_ops_pass")
class PsAppendSendOpsPass(PassBase):
    def __init__(self):
        super(ShardingPass, self).__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        pass


@register_pass("ps_delete_optimizer_pass")
class PsDeleteOptimizesPass(PassBase):
    def __init__(self):
        super(ShardingPass, self).__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        pass


@register_pass("ps_distributed_ops_pass")
class PsDistributedOpsPass(PassBase):
    def __init__(self):
        super(ShardingPass, self).__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        pass


@register_pass("ps_split_heter_worker_ops_pass")
class PsSplitHeterWorkerOpsPass(PassBase):
    def __init__(self):
        super(ShardingPass, self).__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        pass


@register_pass("ps_set_heter_pipeline_opt_pass")
class PsSetHeterPipelineOptPass(PassBase):
    def __init__(self):
        super(ShardingPass, self).__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        pass


@register_pass("ps_delete_extra_optimizer_pass")
class PsDeleteExtraOptimizerPass(PassBase):
    def __init__(self):
        super(ShardingPass, self).__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        pass
