# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import multiprocessing
import signal
import sys

import paddle.fluid as fluid


def _py_version_check():
    if not sys.version_info >= (3, 4):
        raise RuntimeError(
            "Use `paddle.distributed.run` to start parallel training "
            "requires python version greater than 3.4, if your python "
            "is lower than this version, please use "
            "`paddle.distributed.launch` instead.")


def _func_wrapper(func, i, args, error_queue):
    try:
        func(i, *args)
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback
        error_queue.put(traceback.format_exc())
        sys.exit(1)


class MultiprocessContext(object):
    def __init__(self, processes, error_queues):
        _py_version_check()
        self.error_queues = error_queues
        self.processes = processes
        self.sentinels = {
            process.sentinel: index
            for index, process in enumerate(processes)
        }

    def join(self, timeout=None):
        if len(self.sentinels) == 0:
            return True

        ready = multiprocessing.connection.wait(
            self.sentinels.keys(), timeout=timeout)

        error_index = None
        for sentinel in ready:
            index = self.sentinels.pop(sentinel)
            process = self.processes[index]
            process.join()
            if process.exitcode != 0:
                error_index = index
                break

        if error_index is None:
            return len(self.sentinels) == 0

        for process in self.processes:
            if process.is_alive():
                process.terminate()
            process.join()

        if self.error_queues[error_index].empty():
            exitcode = self.processes[error_index].exitcode
            if exitcode < 0:
                name = signal.Signals(-exitcode).name
                raise Exception("Process %d terminated with signal %s." %
                                (error_index, name))
            else:
                raise Exception("Process %d terminated with exit code %s." & (
                    error_index, exitcode))

        original_trace = self.error_queues[error_index].get()
        msg = "\n\n-- Procces %d terminated with the following error:\n" % error_index
        msg += original_trace
        raise Exception(msg)


def launch_processes(func,
                     args=(),
                     nprocs=1,
                     join=True,
                     daemon=False,
                     start_method='spawn'):
    mp = multiprocessing.get_context(start_method)
    error_queues = []
    processes = []
    for i in range(nprocs):
        error_queue = mp.SimpleQueue()
        process = mp.Process(
            target=_func_wrapper,
            args=(func, i, args, error_queue),
            daemon=daemon)
        process.start()
        error_queues.append(error_queue)
        processes.append(process)

    context = MultiprocessContext(processes, error_queues)
    if not join:
        return context

    # loop until all process end
    while not context.join():
        pass


def run(func, args=(), nprocs=1, join=True, daemon=False, start_method='spawn'):
    return launch_processes(func, args, nprocs, join, daemon, start_method)
