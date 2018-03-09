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

from __future__ import print_function
from threading import Lock

import executor
import time
from . import core
from threading import Thread
from Queue import Queue

__all__ = ['ParallelExecutor']

print_lock = Lock()


def save_print(*args, **kwargs):
    with print_lock:
        print(time.time(), *args, **kwargs)


def pretty_id_indent(idx):
    return '\t' * idx * 4 + str(idx) + ":"


def run_exe(q, idx, exe, program, feed, fetch_list, feed_var_name,
            fetch_var_name, cur_scope, return_numpy):
    q.put((idx, exe.run(program, feed, fetch_list, feed_var_name,
                        fetch_var_name, cur_scope, return_numpy)))


class ParallelExecutor(object):
    def __init__(self, places):
        self.scopes = {}
        self.executors = []
        for place in places:
            self.executors.append(executor.Executor(place))

    def run(self,
            program=None,
            feed=None,
            fetch_list=None,
            feed_var_name='feed',
            fetch_var_name='fetch',
            scope=None,
            return_numpy=True):
        # TODO(helin): split input
        q = Queue(maxsize=len(self.executors))
        for idx, exe in enumerate(self.executors):
            if scope is None:
                if idx in self.scopes:
                    cur_scope = self.scopes[idx]
                else:
                    cur_scope = core.Scope()
                    self.scopes[idx] = cur_scope
            t = Thread(
                target=run_exe,
                args=(q, idx, exe, program, feed, fetch_list, feed_var_name,
                      fetch_var_name, cur_scope, return_numpy))
            t.daemon = True
            t.start()

        results = []
        for _ in self.executors:
            results.append(q.get())

        results.sort(key=lambda x: x[0])
        # TODO(helin): concat output
        return results[0][1]
