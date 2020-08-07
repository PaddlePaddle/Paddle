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

import copy
import multiprocessing
import os
import signal
import six
import sys

import paddle.fluid as fluid
from paddle.distributed.launch import get_cluster_and_pod


def _py_version_check():
    if not sys.version_info >= (3, 4):
        raise RuntimeError(
            "Use `paddle.distributed.run` to start parallel training "
            "requires python version greater than 3.4, if your python "
            "is lower than this version, please use "
            "`paddle.distributed.launch` instead.")


class ParallelEnvArgs(object):
    def __init__(self):
        self.cluster_node_ips = None
        self.node_ip = None
        self.use_paddlecloud = None
        self.started_port = None
        self.selected_gpus = None


def init_parallel_env(trainer_id=-1, trainer_num=-1, backend='nccl', **kwargs):
    """

    Args:
        backend(str, optional): The backend to communication between multiple devices.
            Now only support `nccl`. Default value is `nccl`.
    """
    # 1. input check
    if not isinstance(trainer_id, six.integer_types):
        raise TypeError(
            "input `trainer_id` type error, expected type is integer, but received type is %s."
            % type(trainer_id))
    if not isinstance(trainer_num, six.integer_types):
        raise TypeError(
            "input `trainer_num` type error, expected type is integer, but received type is %s."
            % type(trainer_id))
    if not isinstance(backend, six.string_types):
        raise TypeError(
            "input `backend` type error, expected type is str, but received type is %s."
            % type(trainer_id))

    if trainer_id > 0:
        raise ValueError(
            "input `trainer_id` should be greater than 0, but received %d." %
            trainer_id)
    if trainer_num > 0:
        raise ValueError(
            "input `trainer_num` should be greater than 0, but received %d." %
            trainer_num)
    if trainer_id < trainer_num:
        raise ValueError(
            "input `trainer_id` should be less than or equal to `trainer_num`, but `trainer_id` is %d, `trainer_num` is %d."
            % (trainer_id, trainer_num))
    if six.ensure_str(backend) != 'nccl':
        raise ValueError(
            "backend `%s` is not supported, now only supports `nccl` backend." %
            backend)

    # 2. check and prepare environment variables
    # The necessary environment variables include:
    # - PADDLE_TRAINER_ID
    # - PADDLE_TRAINERS_NUM
    # - PADDLE_CURRENT_ENDPOINT
    # - PADDLE_TRAINER_ENDPOINTS

    # get args from kwargs
    args = ParallelEnvArgs()
    args.cluster_node_ips = kwargs.get('cluster_node_ips', "127.0.0.1")
    args.node_ip = kwargs.get('node_ip', "127.0.0.1")
    args.use_paddlecloud = kwargs.get('use_paddlecloud', "False")
    args.started_port = kwargs.get('started_port', None)
    args.selected_gpus = kwargs.get('selected_gpus', None)

    # reuse code of launch.py
    cluster, pod = get_cluster_and_pod(args)

    # copy env & remove useless env vars
    current_env = copy.copy(os.environ.copy())
    current_env.pop("http_proxy", None)
    current_env.pop("https_proxy", None)

    # prepare env var

    assert trainer_num == cluster.trainers_nranks(
    ), "trainer number parse error."
    for trainer in pod.trainers:
        if trainer.id == trainer_id:
            proc_env = {
                "FLAGS_selected_gpus":
                "%s" % ",".join([str(g) for g in selected_gpus]),
                "PADDLE_TRAINER_ID": "%d" % trainer.id,
                "PADDLE_CURRENT_ENDPOINT": "%s" % trainer.endpoint,
                "PADDLE_TRAINERS_NUM": "%d" % cluster.trainers_nranks(),
                "PADDLE_TRAINER_ENDPOINTS":
                ",".join(cluster.trainers_endpoints())
            }
            current_env.update(proc_env)
            break


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
