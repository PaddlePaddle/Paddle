# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
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

from paddle.fluid.wrapped_decorator import wrap_decorator
from .strategy_base import DistributedStrategyBase

non_auto_func_called = True


def __non_auto_func_called__(func):

    def __impl__(*args, **kwargs):
        global non_auto_func_called
        non_auto_func_called = False
        return func(*args, **kwargs)

    return __impl__


is_strict_auto = wrap_decorator(__non_auto_func_called__)


def get_msg_dict(msg):
    res_dict = {}
    fields = msg.DESCRIPTOR.fields
    for f in fields:
        res_dict[f.name] = getattr(msg, f.name)
    return res_dict


def assign_configs_value(msg, config):
    fields = msg.DESCRIPTOR.fields
    for key in config:
        for f in fields:
            if key == f.name:
                # LABEL_OPTIONAL = 1
                # LABEL_REPEATED = 3
                # LABEL_REQUIRED = 2
                if f.label == 3:
                    getattr(msg, f.name).extend(config[f.name])
                elif f.label == 1 or f.label == 2:
                    setattr(msg, f.name, config[f.name])


def check_configs_key(msg, config, field_name):
    key_list = msg.DESCRIPTOR.fields_by_name.keys()
    for key in config:
        assert key in key_list, "key:{} not in {}".format(key, field_name)


class EagerStrategy(DistributedStrategyBase):
    __lock_attr = False

    def __init__(self):
        super(EagerStrategy, self).__init__()
        self.__lock_attr = True

    def __setattr__(self, key, value):
        if self.__lock_attr and not hasattr(self, key):
            raise TypeError("%s is not a attribute of %s" %
                            (key, self.__class__.__name__))
        object.__setattr__(self, key, value)

    @property
    def dgc(self):
        """
        Indicating whether we are using Deep Gradient Compression training. For more details, please refer to
        [Deep Gradient Compression](https://arxiv.org/abs/1712.01887).

        Default Value: False

        Examples:

          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.dgc = True # by default this is false

        """
        return self.strategy.dgc

    @dgc.setter
    @is_strict_auto
    def dgc(self, flag):
        if isinstance(flag, bool):
            self.strategy.dgc = flag
        else:
            print("WARNING: dgc should have value of bool type")

    @property
    def dgc_configs(self):
        r"""
        Set Deep Gradient Compression training configurations. In general, dgc has serveral configurable
        settings that can be configured through a dict.

        **Notes**:
            rampup_begin_step(int): The beginning step from which gradient compression is implemented. Default 0.

            rampup_step(int): Time steps used in sparsity warm-up periods. Default is 1. \
                    For example, if the sparsity is [0.75, 0.9375, 0.984375, 0.996, 0.999], and the rampup_step is 100, \
                    it will use 0.75 at 0~19 steps, and 0.9375 at 20~39 steps, and so on. And when reach sparsity array \
                    ends, it will use 0.999 then and after.

            sparsity(list[float]): Get top important element from gradient tensor, the ratio is (1 - sparsity). \
                    Default is [0.999]. For example, if the sparsity is [0.99, 0.999], the top [1%, 0.1%] important \
                    element will be transmitted.

        Examples:

          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.dgc = True
            strategy.dgc_configs = {"rampup_begin_step": 1252}
        """
        return get_msg_dict(self.strategy.dgc_configs)

    @dgc_configs.setter
    @is_strict_auto
    def dgc_configs(self, configs):
        check_configs_key(self.strategy.dgc_configs, configs, "dgc_configs")
        assign_configs_value(self.strategy.dgc_configs, configs)

    @property
    def nccl_comm_num(self):
        """
        Specifying the number of NCCL communicator

        Default value: 1

        Examples:

          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.nccl_comm_num = 2
        """

        return self.strategy.nccl_comm_num

    @nccl_comm_num.setter
    @is_strict_auto
    def nccl_comm_num(self, value):
        if isinstance(value, int):
            self.strategy.nccl_comm_num = value
        else:
            print("WARNING: nccl_comm_num should have value of int type")

    @property
    def fuse_grad_size_in_MB(self):
        """
        Specifying the size of gradient to fuse in Mega-Bytes

        Default value: 32

        Examples:

          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.fuse_grad_size_in_MB = 50
        """
        return self.strategy.fuse_grad_size_in_MB

    @fuse_grad_size_in_MB.setter
    @is_strict_auto
    def fuse_grad_size_in_MB(self, value):
        if isinstance(value, int):
            self.strategy.fuse_grad_size_in_MB = value
        else:
            print("WARNING: fuse_grad_size_in_MB should have value of int type")

    @property
    def last_comm_group_size_MB(self):
        """
        Specifying the size of gradient to fuse in Mega-Bytes when 
        the last group of each batch communicates. Making the last group 
        small is useful to improve performance. 

        Default value: 1

        Examples:
          .. code-block:: python
        
            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.last_comm_group_size_MB = 2
        """
        return self.strategy.last_comm_group_size_MB

    @last_comm_group_size_MB.setter
    @is_strict_auto
    def last_comm_group_size_MB(self, value):
        if value > 0:
            self.strategy.last_comm_group_size_MB = value
        else:
            raise ValueError("last_comm_group_size_MB should be greater than 0")

    @property
    def find_unused_parameters(self):
        """
        Indicating whether we are using find_unused_parameters to 
        find unused parameters in DataParallel.

        Default value: False

        Examples:

          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.find_unused_parameters = True
        """

        return self.strategy.find_unused_parameters

    @find_unused_parameters.setter
    @is_strict_auto
    def find_unused_parameters(self, flag):
        if isinstance(flag, bool):
            self.strategy.find_unused_parameters = flag
        else:
            print(
                "WARNING: find_unused_parameters should have value of bool type"
            )

    @property
    def hybrid_configs(self):
        """
        Dynamic graph hybrid parallel strategy configuration. Three-way hybrid parallelism 
        needs to meet the following relationships

        total_number_GPUs = dp_degree * mp_degree * pp_degree

        **Note**:
            dp_degree(int): set number of GPUs in a data parallel group. Default -1.
                                    This value should be an integer greater than 0.
                                    If it is not set, or set to -1, its value will be inferred 
                                    based on the total number of cards.
            mp_degree(int): set number of GPUs in a model parallel group. Default 1
            pp_degree(int): set number of GPUs in a pipeline parallel group. Default 1


        Examples:
          .. code-block:: python
            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.hybrid_configs = {
                "dp_degree": 1,
                "mp_degree": 2,
                "pp_degree": 1}
        """
        return get_msg_dict(self.strategy.hybrid_configs)

    @hybrid_configs.setter
    def hybrid_configs(self, configs):
        check_configs_key(self.strategy.hybrid_configs, configs,
                          "hybrid_configs")
        assign_configs_value(self.strategy.hybrid_configs, configs)

    @property
    def heter_ccl_mode(self):
        """
        Indicating whether we are using heter_ccl_mode for model training.
        This feature is currently an experimental feature. Currently,
        heter_ccl_mode can be used only for dataparallel with dygraph mode.
        Default Value: False

        Examples:

          .. code-block:: python

            import paddle
            import paddle.distributed.fleet as fleet

            strategy = fleet.DistributedStrategy()
            strategy.heter_ccl_mode = True

            # for initialize parallel env, only need to call
            paddle.distributed.init_parallel_env()
            # then the heterogenous context will be created.
        """
        return self.strategy.heter_ccl_mode

    @heter_ccl_mode.setter
    def heter_ccl_mode(self, flag):
        if isinstance(flag, bool):
            self.strategy.heter_ccl_mode = flag
        else:
            print("WARNING: heter_ccl_mode should have value of bool type")

    @property
    def amp(self):
        """
        Indicating whether we are using automatic mixed precision training
        Default Value: False

        Examples:

          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.amp = True # by default this is false

        """
        return self.strategy.amp

    @amp.setter
    @is_strict_auto
    def amp(self, flag):
        if isinstance(flag, bool):
            self.strategy.amp = flag
        else:
            print("WARNING: amp should have value of bool type")

    @property
    def recompute(self):
        """
        Indicating whether we are using forward recomputation for memory optimization
        Default value: False

        Examples:

          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.recompute = True
            # suppose x and y are names of checkpoint tensors for recomputation
            strategy.recompute_configs = {"checkpoints": ["x", "y"]}
        """
        return self.strategy.recompute

    @recompute.setter
    @is_strict_auto
    def recompute(self, flag):
        if isinstance(flag, bool):
            self.strategy.recompute = flag
        else:
            print("WARNING: recompute should have value of bool type")
