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

from __future__ import print_function, division

import multiprocessing
import os
import signal
import six
import sys
import warnings

import paddle.fluid as fluid
from paddle.distributed.utils import find_free_ports

# SimpleQueue is different in py2 and py3
if six.PY2:
    import multiprocessing.queues as queues_py2


def _support_set_start_method():
    if not sys.version_info >= (3, 4):
        warnings.warn(
            "`paddle.distributed.run` only supports setting the process"
            " start when python version greater than 3.4, if your python"
            " is lower than this version, only can start processes by"
            " default method of current platform.")
        return False
    return True


def _support_connection_wait():
    if not sys.version_info >= (3, 3):
        return False
    return True


def _set_default_master_env():
    # set default master trainer ip addr
    os.environ['PADDLE_MASTER_IPADDR'] = '127.0.0.1'
    # set default master trainer port
    port_set = find_free_ports(1)
    if port_set is None:
        raise RuntimeError("no free port can be used to parallel training now.")
    os.environ['PADDLE_MASTER_PORT'] = str(list(port_set)[0])


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
        self.error_queues = error_queues
        self.processes = processes
        # NOTE(chenweihang): multiprocessing.connection.wait is a new feature
        # supported from python3.3, which can provide more fine-grained support 
        # for multi-subprocess exit monitoring.
        self.use_connection_wait = _support_connection_wait()
        if self.use_connection_wait:
            self.sentinels = {
                process.sentinel: index
                for index, process in enumerate(processes)
            }

    def join(self, timeout=None):
        if self.use_connection_wait:
            return self._join_with_conn_wait(timeout)
        else:
            return self._join_without_conn_wait(timeout)

    def _join_with_conn_wait(self, timeout=None):
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

        self._join_and_throw_exception(error_index)

    # NOTE(chenweihng): This method is not as efficient as connection.wait.
    # Beccause if process has already stopped, p.join() will return immediately.
    # If process hasn't stopped, it will block until the process end, and 
    # as the same time, the other process may have been end, which makes it 
    # impossible for us to accurately capture the first failed process.
    # Here we avoid process block by setting timeout, but if the other process
    # exit before the timeout end, we will also encounter the previois problem,
    # but maybe there is no better way here. When we fully migrated to python3, 
    # this problem disappeared
    def _join_without_conn_wait(self, timeout=None):
        finished_processes = []
        error_index = None
        timeout = timeout // len(self.processes) if timeout else 1
        for index, process in enumerate(self.processes):
            # try to join selected process
            process.join(timeout=timeout)
            # This will be None if the process has not yet terminated
            if process.exitcode is not None:
                # exit with exception
                if process.exitcode == 0:
                    finished_processes.append(index)
                else:
                    error_index = index
                    finished_processes.append(index)
                    break

        if error_index is None:
            return len(finished_processes) == len(self.processes)

        self._join_and_throw_exception(error_index)

    def _join_and_throw_exception(self, error_index):
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
                raise Exception("Process %d terminated with exit code %d." & (
                    error_index, exitcode))

        original_trace = self.error_queues[error_index].get()
        msg = "\n\n----------------------------------------------\n" \
              "Procces %d terminated with the following error:\n" \
              "----------------------------------------------\n\n" % error_index
        msg += original_trace
        raise Exception(msg)


def launch_processes(func,
                     args=(),
                     nprocs=1,
                     join=True,
                     daemon=False,
                     start_method='spawn'):
    # NOTE(chenweihang): [ why need set default master info before run? ]
    # when using `paddle.distributed.run` start parallel training,
    # users need use `init_parallel_env` to config some cluster info
    # inner subprocess, if each process find free port for itself,
    # the started port may be different, it will cause endpoints is
    # different in different subprocesses
    _set_default_master_env()

    # start processes
    if _support_set_start_method():
        mp = multiprocessing.get_context(start_method)
    else:
        mp = multiprocessing

    error_queues = []
    processes = []
    for i in range(nprocs):
        if six.PY2:
            error_queue = queues_py2.SimpleQueue()
        else:
            error_queue = mp.SimpleQueue()
        process = mp.Process(
            target=_func_wrapper, args=(func, i, args, error_queue))
        process.daemon = daemon
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
