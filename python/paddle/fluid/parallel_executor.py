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
                 use_cuda,
                 loss_name=None,
                 main_program=None,
                 num_threads=None,
                 allow_op_delay=False,
                 share_vars_from=None):
        self._places = []
        self._act_places = []
        if use_cuda:
            for i in xrange(core.get_cuda_device_count()):
                p = core.Place()
                self._act_places.append(core.CUDAPlace(i))
                p.set_place(self._act_places[-1])
                self._places.append(p)
        else:
            for i in xrange(multiprocessing.cpu_count()):
                p = core.Place()
                self._act_places.append(core.CPUPlace(i))
                p.set_place(self._act_places[-1])
                self._places.append(p)
        assert self._places, "no place for execution"

        if num_threads is None:
            if use_cuda:
                # Experiments on se-resnext shows that too many threads hurt
                # performance. Worth tunning for other models in the future.
                num_threads = len(self._places)
            else:
                min(len(self._places) * 2, multiprocessing.cpu_count())

        main = main_program
        main = main if main else framework.default_main_program()
        scope = executor.global_scope()

        local_scopes = share_vars_from.executor.local_scopes(
        ) if share_vars_from else []

        vars = [
            v.name
            for v in filter(lambda var: var.persistable, main.list_vars())
        ]

        self.executor = core.ParallelExecutor(
            num_threads,
            True if use_cuda else False,  # use_event
            self._places,
            set([
                p.name for p in main.global_block().iter_parameters()
                if not p.stop_gradient
            ]),
            set(vars),
            main.desc,
            loss_name if loss_name else '',
            scope,
            local_scopes,
            allow_op_delay)
        self.scope = scope

    def run(self, fetch_list, feed_dict={}):
        """
        :param fetch_list: A list of variable names that will be fetched.
        :param feed_dict: A dict mapping for feed variable name to LoDTensor
          or numpy array.
        :return: fetched value list.
        """
        if not isinstance(feed_dict, dict):
            raise TypeError("feed_dict should be a dict")

        feed_tensor_dict = {}
        for i, feed_name in enumerate(feed_dict):
            feed_tensor = feed_dict[feed_name]
            if not isinstance(feed_tensor, core.LoDTensor):
                feed_tensor = core.LoDTensor()
                feed_tensor.set(feed_dict[feed_name], self._act_places[0])
            feed_tensor_dict[feed_name] = feed_tensor

        fetch_var_name = '@FETCHED_VAR_NAME@'
        self.executor.run(fetch_list, fetch_var_name, feed_tensor_dict)
        arr = self.scope.find_var(fetch_var_name).get_lod_tensor_array()
        return [arr[i] for i in range(len(arr))]
