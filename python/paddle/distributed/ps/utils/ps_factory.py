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
from .ps_program_builder import *
from .public import *

__all__ = [
    'PsProgramBuilder', 'GeoPsProgramBuilder', 'CpuSyncPsProgramBuilder',
    'CpuAsyncPsProgramBuilder', 'GpuPsProgramBuilder',
    'HeterAsyncPsProgramBuilder', 'FlPsProgramBuilder'
]


class PsProgramBuilderFactory(object):
    def __init__(self):
        pass

    def _create_ps_program_builder(self, pass_ctx):
        attrs = pass_ctx._attrs
        if attrs['ps_mode'] == DistributedMode.GEO:
            return globals()['GeoPsProgramBuilder'](pass_ctx)
        elif attrs['use_ps_gpu']:
            return globals()['GpuPsProgramBuilder'](pass_ctx)
        elif attrs['is_heter_ps_mode']:
            return globals()['HeterAsyncPsProgramBuilder'](pass_ctx)
        elif 'is_fl_ps_mode' in attrs and attrs[
                'is_fl_ps_mode'] == DistributedMode.FL:
            return globals()['FlPsProgramBuilder'](pass_ctx)
        else:
            return globals()['CpuSyncPsProgramBuilder'](pass_ctx)
