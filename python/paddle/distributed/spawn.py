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
import sys
import warnings

from paddle.distributed.utils import find_free_ports


def _py_supported_check():
    if not sys.version_info >= (3, 4):
        raise RuntimeError(
            "Use `paddle.distributed.spawn` or `paddle.distributed.start_processes` "
            "to start parallel training requires python version greater than 3.4, "
            "if your python is lower than this version, please use "
            "`paddle.distributed.launch` instead.")


def _set_default_assist_env(nprocs):
    # set default master trainer ip addr
    os.environ['PADDLE_MASTER_IPADDR'] = '127.0.0.1'
    # set default master trainer port
    port_set = find_free_ports(1)
    if port_set is None:
        raise RuntimeError("no free port can be used to parallel training now.")
    os.environ['PADDLE_MASTER_PORT'] = str(list(port_set)[0])
    # set default selected gpus
    # e.g. if the nprocs is 4, the selected gpus is "0,1,2,3"
    # NOTE(chenweihang): [ why not use FLAGS_selected_gpus directly? ]
    # because the FLAGS_selected_gpus may be used in other place,
    # if we set FLAGS_selected_gpus to be `0,1,2,3`, it may cause error
    # when using `ParallelEnv`
    os.environ['PADDLE_CUDA_VISIBLE_DEVICES'] = ",".join(
        [str(x) for x in range(0, nprocs)])


def _func_wrapper(func, i, args, error_queue, return_queue):
    try:
        result = func(i, *args)
        return_queue.put(result)
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback
        error_queue.put(traceback.format_exc())
        sys.exit(1)


class MultiprocessContext(object):
    def __init__(self, processes, error_queues, return_queues):
        _py_supported_check()
        self.error_queues = error_queues
        # NOTE(chenweihang): The `start_processes` method is mainly used 
        # to wrap the outermost execution function of the program for 
        # parallel execution. Generally, the return value is not concerned, 
        # but if the user needs to obtain the return value, users can get  
        # the return result of each process from context.return_queues
        self.return_queues = return_queues
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

        self._throw_exception(error_index)

    def _throw_exception(self, error_index):
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
              "Process %d terminated with the following error:\n" \
              "----------------------------------------------\n\n" % error_index
        msg += original_trace
        raise Exception(msg)


# NOTE(chenweihang): [ why default start method is spawn? ]
# The CUDA runtime does not support the fork start method, 
# either the spawn or forkserver start method are required 
# to use CUDA in subprocesses.
def start_processes(func,
                    args=(),
                    nprocs=1,
                    join=True,
                    daemon=False,
                    start_method='spawn'):
    """
    Start multiple processes for parallel training.

    Args:
        func (function): The target function is called by started process.
            This function need to be able to pickled, so it must be defined
            at the top level of a module.
            This function should be called as ``func(i, *args)`` , ``i`` is
            the process index and ``args`` contains other arguments as tuple.
        args (tuple): Arguments passed to ``func`` .
        nprocs (int): Number of processed to start. 
        join (bool): Perform a blocking join on all started processes.
            Default: True.
        daemon (bool): The started processes' daemon flag. Default: False.
        start_method (string): the way to start a process. The start method
            can be ``spawn`` , ``fork`` , ``forkserver`` . Because the CUDA 
            runtime does not support the ``fork`` start method, when use 
            CUDA in subprocesses, we should start process by ``spawn`` or
            ``forkserver`` method. Default: 'spawn'.

    Returns:
        ``MultiprocessContext`` object, it hold the started processes.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn
            import paddle.optimizer as opt
            import paddle.distributed as dist

            class LinearNet(nn.Layer):
                def __init__(self):
                    super(LinearNet, self).__init__()
                    self._linear1 = nn.Linear(10, 10)
                    self._linear2 = nn.Linear(10, 1)
                    
                def forward(self, x):
                    return self._linear2(self._linear1(x))

            def train(rank):
                # 1. enable dynamic mode
                paddle.disable_static()
                
                # 2. initialize parallel environment
                dist.init_parallel_env(rank)

                # 3. create data parallel layer & optimizer
                layer = LinearNet()
                dp_layer = paddle.DataParallel(layer)

                loss_fn = nn.MSELoss()
                sgd = opt.SGD(
                    learning_rate=0.001, parameter_list=dp_layer.parameters())

                # 4. run layer
                inputs = paddle.randn([10, 10], 'float32')
                outputs = dp_layer(inputs)
                labels = paddle.randn([10, 1], 'float32')
                loss = loss_fn(outputs, labels)
                
                loss = dp_layer.scale_loss(loss)
                loss.backward()
                dp_layer.apply_collective_grads()

                sgd.minimize(loss)
                dp_layer.clear_gradients()

            if __name__ == '__main__':
                dist.start_processes(train, args=(), nprocs=2)
    """
    # NOTE(chenweihang): [ why only supports python3.4+ ? ]
    # Python supported setting the child process startup method
    # since 3.4. The previous version can only use the default startup 
    # method, while the default startup method of Unix is fork, which 
    # cannot support CUDA runtime multi-process
    _py_supported_check()

    # NOTE(chenweihang): [ why need set default master info before run? ]
    # when using `paddle.distributed.spawn/start_processes` start 
    # parallel training, users need use `init_parallel_env` to config 
    # cluster info inner subprocess, if each process find free port for 
    # itself, the started port may be different, it will cause endpoints is
    # different in different subprocesses
    _set_default_assist_env(nprocs)

    # start processes
    mp = multiprocessing.get_context(start_method)

    error_queues = []
    return_queues = []
    processes = []
    for i in range(nprocs):
        error_queue = mp.SimpleQueue()
        return_queue = mp.SimpleQueue()
        process = mp.Process(
            target=_func_wrapper,
            args=(func, i, args, error_queue, return_queue))
        process.daemon = daemon
        process.start()
        error_queues.append(error_queue)
        return_queues.append(return_queue)
        processes.append(process)

    context = MultiprocessContext(processes, error_queues, return_queues)
    if not join:
        return context

    # loop until all process end
    while not context.join():
        pass

    # finally return context
    return context


# NOTE(chenweihang): this method only supports start processes
# by `spwan` method, if users want to start processes by other
# method, they can use start_processes
def spawn(func, args=(), nprocs=1, join=True, daemon=False):
    """
    Start multiple processes with ``spawn`` method for parallel training.
    
    This is a specialized method of ``paddle.distributed.start_processes`` .

    Args:
        func (function): The target function is called by spawned process.
            This function need to be able to pickled, so it must be defined
            at the top level of a module.
            This function should be called as ``func(i, *args)``, ``i`` is
            the process index and ``args`` contains other arguments as tuple.
        args (tuple): Arguments passed to ``func``.
        nprocs (int): Number of processed to spawn. 
        join (bool): Perform a blocking join on all spawned processes.
            Default: True.
        daemon (bool): The spawned processes' daemon flag. Default: False.

    Returns:
        ``MultiprocessContext`` object, it hold the spawned processes.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn
            import paddle.optimizer as opt
            import paddle.distributed as dist

            class LinearNet(nn.Layer):
                def __init__(self):
                    super(LinearNet, self).__init__()
                    self._linear1 = nn.Linear(10, 10)
                    self._linear2 = nn.Linear(10, 1)
                    
                def forward(self, x):
                    return self._linear2(self._linear1(x))

            def train(rank):
                # 1. enable dynamic mode
                paddle.disable_static()
                
                # 2. initialize parallel environment
                dist.init_parallel_env(rank)

                # 3. create data parallel layer & optimizer
                layer = LinearNet()
                dp_layer = paddle.DataParallel(layer)

                loss_fn = nn.MSELoss()
                sgd = opt.SGD(
                    learning_rate=0.001, parameter_list=dp_layer.parameters())

                # 4. run layer
                inputs = paddle.randn([10, 10], 'float32')
                outputs = dp_layer(inputs)
                labels = paddle.randn([10, 1], 'float32')
                loss = loss_fn(outputs, labels)
                
                loss = dp_layer.scale_loss(loss)
                loss.backward()
                dp_layer.apply_collective_grads()

                sgd.minimize(loss)
                dp_layer.clear_gradients()

            if __name__ == '__main__':
                dist.spawn(train, args=(), nprocs=2)
    """
    return start_processes(func, args, nprocs, join, daemon, 'spawn')
