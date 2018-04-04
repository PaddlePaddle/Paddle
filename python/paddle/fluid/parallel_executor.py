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
    def __init__(self,
                 loss_name=None,
                 use_cuda=None,
                 num_threads=None,
                 allow_op_delay=False,
                 main_program=None,
                 startup_program=None,
                 local_scopes=None,
                 run_startup=True):
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

        startup = startup_program if startup_program else framework.default_startup_program(
        )
        main = main_program if main_program else framework.default_main_program(
        )
        scope = executor.global_scope()

        if run_startup:
            place = core.CUDAPlace(0) if use_cuda else core.CPUPlace()
            exe = executor.Executor(place)
            exe.run(startup)

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
            loss_name if loss_name else '',
            scope,
            local_scopes if local_scopes else [],
            allow_op_delay)
        self.scope = scope

    def local_scopes(self):
        return self.executor.local_scopes()

    def run(self, fetch_list):
        fetch_var_name = '@FETCHED_VAR_NAME@'
        self.executor.run(fetch_list, fetch_var_name)
        arr = self.scope.find_var(fetch_var_name).get_lod_tensor_array()
        return [arr[i] for i in range(len(arr))]
