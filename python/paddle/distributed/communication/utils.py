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

import os
import paddle.fluid.core as core
import paddle.fluid.framework as framework
import paddle.distributed.parallel as parallel


def _set_place(backend):
    if parallel._is_cpuonly(backend):
        place = core.CPUPlace()
    elif core.is_compiled_with_cuda():
        selected_gpus = os.getenv("FLAGS_selected_gpus", "0").split(",")
        device_id = int(selected_gpus[0])
        place = core.CUDAPlace(device_id)
    else:
        raise NotImplementedError(
            "Valid backends are {}. But input {} as parameter.".format(
                _ProcessGroupManager.valid_backend_list, backend))

    framework._set_expected_place(place)


class _ProcessGroupManager():
    global_group_id = 0
    valid_backend_list = ['nccl', 'gloo']
    default_group_name = "_default_pg"
