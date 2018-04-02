#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import core
import multiprocessing
import framework
import executor

__all__ = ['ParallelExecutor']


class ParallelExecutor(object):
    def __init__(self, loss_name, use_cuda, num_threads=None):
        places = []
        if use_cuda:
            for i in xrange(core.get_cuda_device_count()):
                p = core.Place()
                p.set_place(core.CUDAPlace(i))
                places.append(p)
        else:
            for i in xrange(multiprocessing.cpu_count()):
                p = core.Place()
                p.set_place(core.CPUPlace())
                places.append(p)

        if num_threads is None:
            if use_cuda:
                # Experiments on se-resnext shows that too many threads hurt
                # performance. Worth tunning for other models in the future.
                num_threads = len(places)
            else:
                min(len(places) * 2, multiprocessing.cpu_count())

        startup = framework.default_startup_program()
        main = framework.default_main_program()
        scope = executor.global_scope()

        self.executor = core.ParallelExecutor(
            num_threads,
            True if use_cuda else False,  # use_event
            places,
            set([
                p.name for p in main.global_block().iter_parameters()
                if not p.stop_gradient
            ]),
            startup.desc,
            main.desc,
            loss_name,
            scope)
        self.scope = scope

    def run(self, fetch_list):
        fetch_var_name = '@FETCHED_VAR_NAME@'
        self.executor.run(fetch_list, fetch_var_name)
        arr = self.scope.find_var(fetch_var_name).get_lod_tensor_array()
        return [arr[i] for i in range(len(arr))]
