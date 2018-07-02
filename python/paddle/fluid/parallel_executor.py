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
import warnings
import sys
import os

__all__ = ['ParallelExecutor', 'ExecutionStrategy', 'BuildStrategy']


class BuildStrategy(object):
    """
    BuildStrategy allows user to more preciously control how to build
    the SSA Graph in ParallelExecutor.

    Args:
        reduce_strategy (str): There are two reduce strategies, 'AllReduce'
            and 'Reduce'. If you want that all parameters will be optimized
            on all devices, you can choose 'AllReduce'; if you choose
            'Reduce', all parameters will be evenly allocated to different
            devices for optimization, and then broadcast the optimized
            parameter to other devices. Default 'AllReduce'.
        gradient_scale_strategy (str): There are two ways of defining loss@grad,
            'CoeffNumDevice' and 'Customized'. By default, ParallelExecutor
            sets the loss@grad according to the number of devices. If you want
            to customize loss@grad, you can choose 'Customized'.
            Default 'CoeffNumDevice'.
        debug_graphviz_path (str): Whether to write the SSA Graph to file in the
            form of graphviz. It is useful for debugging. Default "".

    Returns:
        BuildStrategy: The initialized BuildStrategy object.

    Raises:
        ValueError: If reduce_strategy is not 'AllReduce' or 'Reduce',
            or gradient_scale is not 'CoeffNumDevice' or 'Customized'.

    """

    def __init__(self,
                 reduce_strategy="AllReduce",
                 gradient_scale_strategy="CoeffNumDevice",
                 debug_graphviz_path=""):
        if reduce_strategy not in ["AllReduce", "Reduce"]:
            raise ValueError("Unknown reduce_strategy: '%s'. It can only be "
                             "'AllReduce' or 'Reduce'.", str(reduce_strategy))
        if gradient_scale_strategy not in ["CoeffNumDevice", "Customized"]:
            raise ValueError(
                "Unknown gradient_scale_strategy: '%s'. It can only be "
                "'CoeffNumDevice' or 'Customized'.",
                str(gradient_scale_strategy))

        self.reduce_strategy_map = {
            "AllReduce":
            core.ParallelExecutor.BuildStrategy.ReduceStrategy.AllReduce,
            "Reduce": core.ParallelExecutor.BuildStrategy.ReduceStrategy.Reduce
        }
        self.gradient_scale_strategy_map = {
            "CoeffNumDevice": core.ParallelExecutor.BuildStrategy.
            GradientScaleStrategy.CoeffNumDevice,
            "Customized":
            core.ParallelExecutor.BuildStrategy.GradientScaleStrategy.Customized
        }

        self._reduce_strategy = self.reduce_strategy_map[reduce_strategy]
        self._gradient_scale_strategy = \
            self.gradient_scale_strategy_map[gradient_scale_strategy]
        self._debug_graphviz_path = debug_graphviz_path

    @property
    def reduce_strategy(self):
        return self._reduce_strategy

    @reduce_strategy.setter
    def reduce_strategy(self, reduce_strategy):
        assert isinstance(reduce_strategy, str)
        assert reduce_strategy in ["AllReduce", "Reduce"]
        self._reduce_strategy = self.reduce_strategy_map[reduce_strategy]

    @property
    def gradient_scale_strategy(self):
        return self._gradient_scale_strategy

    @gradient_scale_strategy.setter
    def gradient_scale_strategy(self, gradient_scale_strategy):
        assert isinstance(gradient_scale_strategy, str)
        assert gradient_scale_strategy in ["CoeffNumDevice", "Customized"]
        self._gradient_scale_strategy =\
            self.gradient_scale_strategy_map[gradient_scale_strategy]

    @property
    def debug_graphviz_path(self):
        return self._debug_graphviz_path

    @debug_graphviz_path.setter
    def debug_graphviz_path(self, debug_graphviz_path):
        assert isinstance(debug_graphviz_path, str)
        self._debug_graphviz_path = debug_graphviz_path


class ExecutionStrategy(object):
    """
    ExecutionStrategy allows user to more preciously control how to run
    the program in ParallelExecutor.

    Args:
        use_cuda (bool): Whether to use CUDA or not. Default True.
        num_threads (int): The number of threads that used to run the
            operators in ParallelExecutor. If it is not set, it will be
            set in ParallelExecutor according to the device count.
            Default 0.
        allow_op_delay (bool): Whether to delay the communication operators
            to run. Default False.
        num_iteration_per_drop_scope (int): how many iterations between
            the two dropping local scopes. Default 100.

    Returns:
        ExecutionStrategy: The initialized ExecutionStrategy object.

    """

    def __init__(self,
                 use_cuda=True,
                 num_threads=0,
                 allow_op_delay=False,
                 num_iteration_per_drop_scope=100):
        assert isinstance(use_cuda, bool)
        assert num_threads >= 0
        assert isinstance(allow_op_delay, bool)
        assert num_iteration_per_drop_scope > 0
        self._use_cuda = use_cuda
        self._num_threads = num_threads
        self._allow_op_delay = allow_op_delay
        self._num_iteration_per_drop_scope = num_iteration_per_drop_scope

    @property
    def use_cuda(self):
        return self._use_cuda

    @use_cuda.setter
    def use_cuda(self, use_cuda):
        assert isinstance(use_cuda, bool)
        self._use_cuda = use_cuda

    @property
    def num_threads(self):
        return self._num_threads

    @num_threads.setter
    def num_threads(self, num_threads):
        assert num_threads >= 0
        self._num_threads = num_threads

    @property
    def allow_op_delay(self):
        return self._num_threads

    @allow_op_delay.setter
    def allow_op_delay(self, allow_op_delay):
        assert isinstance(allow_op_delay, bool)
        self._allow_op_delay = allow_op_delay

    @property
    def num_iteration_per_drop_scope(self):
        return self._num_threads

    @num_iteration_per_drop_scope.setter
    def num_iteration_per_drop_scope(self, num_iteration_per_drop_scope):
        assert num_iteration_per_drop_scope > 0
        self._num_iteration_per_drop_scope = num_iteration_per_drop_scope


class ParallelExecutor(object):
    """
    ParallelExecutor can run program in parallel.

    Args:
        use_cuda (bool): Whether to use CUDA or not.
        loss_name (str): The loss name must set in training. Default None.
        main_program (Program): The program that need to run, if not provided,
            then default_main_program will be used. Default None.
        share_vars_from(ParallelExecutor): If provied, it will share variables
            from the specified ParallelExecutor. Default None.
        exec_strategy(ExecutionStrategy): exec_strategy allows a user to more
            preciously control how to run the program, such as how many threads
            are used to run the operators (default: dev_cnt * 2 for CPU,
            dev_cnt * 4 for GPU), whether to delay the communication operators,
            how many iterations between the two dropping local scopes.
            For detail information, please refer to ExecutionStrategy's Doc.
            Default None.
        build_strategy(BuildStrategy): build_strategy allows a user to more
            preciously control how to build the SSA Graph, such as whether to
            make balancing parameter optimization between GPUs, whether to
            write the SSA Graph to file in the form of graphviz. For detail
            information, please refer to BuildStrategy's Doc. Default None.
        num_trainers(int): num_trainers is a parameter for distributed training.
            If it is greater than 1, NCCL will be initialized with multiple rank
            of nodes, and each node should have same number of GPUs. Distributed
            training will be enabled then. Default 1.
        trainer_id(int): trainer_id is a parameter for distributed training. The
            trainer_id is used together with num_trainers. It is the "rank" of
            current node starts from 0. Default 0.

    Returns:
        ParallelExecutor: The initialized ParallelExecutor object.

    Raises:
        TypeError: If share_vars_from is provided, but not ParallelExecutor object.

    Examples:
        .. code-block:: python

          train_exe = fluid.ParallelExecutor(use_cuda=True, loss_name=loss.name)
          test_exe = fluid.ParallelExecutor(use_cuda=True,
                                            main_program=test_program,
                                            share_vars_from=train_exe)

          train_loss, = train_exe.run([loss.name], feed=feed_dict)
          test_loss, = test_exe.run([loss.name], feed=feed_dict)
    """

    def __init__(self,
                 use_cuda,
                 loss_name=None,
                 main_program=None,
                 share_vars_from=None,
                 exec_strategy=None,
                 build_strategy=None,
                 num_trainers=1,
                 trainer_id=0,
                 **kwargs):
        if len(kwargs) != 0:
            err_msg = ""
            for key in kwargs:
                if key in dir(ExecutionStrategy):
                    err_msg += \
                        "Setting {0} by constructor is deprecated. Use " \
                        "strategy=ExecutionStrategy(); strategy.{0}=xxx; " \
                        "pe=ParallelExecutor(exec_strategy=strategy) " \
                        "instead.\n ".format(key)
                elif key in dir(BuildStrategy):
                    err_msg += \
                        "Setting {0} by constructor is deprecated. Use " \
                        "strategy=BuildStrategy(); See help(" \
                        "paddle.fluid.ParallelExecutor.BuildStrategy) \n".format(
                            key)
                else:
                    err_msg += "Setting {0} by constructor is deprecated. Use strategy.\n".format(
                        key)
            raise ValueError(err_msg)

        self._places = []
        self._act_places = []
        if use_cuda:
            for i in xrange(core.get_cuda_device_count()):
                p = core.Place()
                self._act_places.append(core.CUDAPlace(i))
                p.set_place(self._act_places[-1])
                self._places.append(p)
        else:
            cpu_num = int(
                os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
            for i in xrange(cpu_num):
                p = core.Place()
                self._act_places.append(core.CPUPlace())
                p.set_place(self._act_places[-1])
                self._places.append(p)
        assert self._places, "no place for execution"

        core_exec_strategy = core.ParallelExecutor.ExecutionStrategy()
        core_exec_strategy.use_cuda = use_cuda
        if exec_strategy is not None:
            core_exec_strategy.num_threads = exec_strategy.num_threads
            core_exec_strategy.allow_op_delay = True if exec_strategy.allow_op_delay else False
            core_exec_strategy.num_iteration_per_drop_scope = exec_strategy.num_iteration_per_drop_scope

        if core_exec_strategy.num_threads == 0:
            if use_cuda:
                # Experiments on se-resnext shows that too many threads hurt
                # performance. Worth tunning for other models in the future.
                core_exec_strategy.num_threads = len(self._places) * 4
            else:
                cpu_num = int(
                    os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
                core_exec_strategy.num_threads = cpu_num * 2

        core_build_strategy = core.ParallelExecutor.BuildStrategy()
        if build_strategy is not None:
            core_build_strategy.reduce_strategy = build_strategy.reduce_strategy
            core_build_strategy.gradient_scale_strategy = build_strategy.gradient_scale_strategy
            core_build_strategy.debug_graphviz_path = build_strategy.debug_graphviz_path

        main = main_program
        main = main if main else framework.default_main_program()
        scope = executor.global_scope()

        # FIXME(Yancey1989): it's a temporary approach to determinate the distribute
        # train program, call self.bcast_param() at the end of each mini-batch.
        self.is_dist = True if "recv" in [
            op.type for op in main.global_block().ops
        ] else False

        if share_vars_from and not isinstance(share_vars_from,
                                              ParallelExecutor):
            raise TypeError("share_vars_from must be ParallelExecutor.")

        local_scopes = share_vars_from.executor.local_scopes(
        ) if share_vars_from else []

        self.persistable_vars = [
            v.name
            for v in filter(
                lambda var: var.persistable and var.type != core.VarDesc.VarType.RAW,
                main.list_vars())
        ]

        self.executor = core.ParallelExecutor(
            self._places,
            set([
                p.name for p in main.global_block().iter_parameters()
                if not p.stop_gradient
            ]),
            set(self.persistable_vars), main.desc, loss_name
            if loss_name else '', scope, local_scopes, core_exec_strategy,
            core_build_strategy, num_trainers, trainer_id)
        self.scope = scope

    def run(self, fetch_list, feed=None, feed_dict=None, return_numpy=True):
        """
        Run a parallel executor with fetch_list.

        The feed parameter can be a dict or a list. If feed is a dict, the
        feed data will be split into multiple devices. If feed is a list, we
        assume the data has been splitted into multiple devices, the each
        element in the list will be copied to each device directly.

        For example, if the feed is a dict:

        >>> exe = ParallelExecutor()
        >>> # the image will be splitted into devices. If there is two devices
        >>> # each device will process an image with shape (24, 1, 28, 28)
        >>> exe.run(feed={'image': numpy.random.random(size=(48, 1, 28, 28))})

        For example, if the feed is a list:

        >>> exe = ParallelExecutor()
        >>> # each device will process each element in the list.
        >>> # the 1st device will process an image with shape (48, 1, 28, 28)
        >>> # the 2nd device will process an image with shape (32, 1, 28, 28)
        >>> #
        >>> # you can use exe.device_count to get the device number.
        >>> exe.run(feed=[{"image": numpy.random.random(size=(48, 1, 28, 28))},
        >>>               {"image": numpy.random.random(size=(32, 1, 28, 28))},
        >>>              ])

        Args:
            fetch_list(list): The fetched variable names
            feed(list|dict|None): The feed variables. If the feed is a dict,
                tensors in that dict will be splitted into each devices. If
                the feed is a list, each element of the list will be copied
                to each device. Default None.
            feed_dict: Alias for feed parameter, for backward compatibility.
                This parameter has been deprecated. Default None.
            return_numpy(bool): Whether converts the fetched tensor to numpy.
                Default: True.

        Returns:
            List: The fetched result list.

        Raises:
            ValueError: If the feed is a list, but its length is not equal the
                length of active places, or its element's is not dict.

        NOTES:
            1. If the feed's type is dict, the number of data that feeds to
               ParallelExecutor must be bigger than active places. Otherwise,
               it will throw exception from C++ side. Special attention should be
               paid to check whether the last batch of the dataset is bigger
               than active places.
            2. If active places are more than one, the fetch results for each
               variable is a list, and each element of this list is the variable of
               respective active place.

        Examples:
            .. code-block:: python

                pe = fluid.ParallelExecutor(use_cuda=use_cuda,
                                            loss_name=avg_cost.name,
                                            main_program=fluid.default_main_program())
                loss = pe.run(feed=feeder.feed(cur_batch),
                              fetch_list=[avg_cost.name]))
        """
        if feed is None and feed_dict is not None:
            feed = feed_dict
            print >> sys.stderr, "`feed_dict` is deprecated. Please use `feed=`"

        if isinstance(feed, dict):
            feed_tensor_dict = dict()
            for feed_name in feed:
                feed_tensor = feed[feed_name]
                if not isinstance(feed_tensor, core.LoDTensor):
                    feed_tensor = core.LoDTensor()
                    # always set to CPU place, since the tensor need to be splitted
                    # it is fast in CPU
                    feed_tensor.set(feed[feed_name], core.CPUPlace())
                feed_tensor_dict[feed_name] = feed_tensor

            self.executor.feed_and_split_tensor_into_local_scopes(
                feed_tensor_dict)
        elif isinstance(feed, list) or isinstance(feed, tuple):
            if len(feed) != len(self._act_places):
                raise ValueError(
                    "Feed a list of tensor, the list should be the same size as places"
                )

            res = list()

            for i, each in enumerate(feed):
                if not isinstance(each, dict):
                    raise TypeError(
                        "Each element of feed list should be a dict")
                res_dict = dict()
                for feed_name in each:
                    tensor = each[feed_name]
                    if not isinstance(tensor, core.LoDTensor):
                        tmp = core.LoDTensor()
                        tmp.set(tensor, self._act_places[i])
                        tensor = tmp
                    res_dict[feed_name] = tensor
                res.append(res_dict)
            self.executor.feed_tensors_into_local_scopes(res)

        fetch_var_name = '@FETCHED_VAR_NAME@'
        self.executor.run(fetch_list, fetch_var_name)
        arr = self.scope.find_var(fetch_var_name).get_lod_tensor_array()

        if self.is_dist:
            self.bcast_params()

        if return_numpy:
            return executor.as_numpy(arr)

        return [arr[i] for i in range(len(arr))]

    def bcast_params(self):
        """
        Broadcast the parameters to other devices. It is used during
        distributed training.
        """
        self.executor.bcast_params(set(self.persistable_vars))

    @property
    def device_count(self):
        return len(self._act_places)
