# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.distributed.fleet.proto import distributed_strategy_pb2
from paddle.fluid.framework import Variable, set_flags, core, _global_flags
from paddle.fluid.wrapped_decorator import wrap_decorator
import google.protobuf.text_format
import google.protobuf

__all__ = []

non_auto_func_called = True


def __non_auto_func_called__(func):

    def __impl__(*args, **kwargs):
        global non_auto_func_called
        non_auto_func_called = False
        return func(*args, **kwargs)

    return __impl__


is_strict_auto = wrap_decorator(__non_auto_func_called__)


class DistributedJobInfo(object):
    """
    DistributedJobInfo will serialize all distributed training information
    Just for inner use: 1) debug 2) replicate experiments
    """

    def __init__(self):
        self.job_info = distributed_strategy_pb2.DistributedJobInfo()

    def _set_worker_num(self, worker_num):
        self.job_info.worker_num = worker_num

    def _set_server_num(self, server_num):
        self.job_info.server_num = server_num

    def _set_worker_ips(self, worker_ips):
        self.job_info.worker_ips.extend(worker_ips)

    def _set_server_endpoints(self, server_endpoints):
        self.job_info.server_endpoints.extend(server_endpoints)

    def _set_origin_startup(self, origin_startup_prog):
        self.job_info.origin_startup = str(origin_startup_prog)

    def _set_origin_main(self, origin_main_prog):
        self.job_info.origin_main = str(origin_main_prog)

    def _distributed_main(self, distributed_main_prog):
        self.job_info.distributed_main = str(distributed_main_prog)

    def _optimizer_name(self, optimizer_name):
        self.job_info.optimizer_name = optimizer_name

    def _set_distributed_strategy(self, dist_strategy):
        self.job_info.strategy = dist_strategy


ReduceStrategyFluid = paddle.fluid.BuildStrategy.ReduceStrategy
ReduceStrategyFleet = int


class DistributedStrategyBase(object):
    __lock_attr = False

    def __init__(self):
        """
        DistributedStrategy is the main configuration entry for distributed training of Paddle.
        All of the distributed training configurations can be configured in DistributedStrategy,
        such as automatic mixed precision (AMP), Layer-wise Adaptive Rate Scaling (LARS), 
        asynchronous update parameter server(ASGD), etc.

        DistributedStrategy can be serialized into protobuf file or deserialized from protobuf file

        Users who run local training usually configure BuildStrategy and ExecutionStrategy, and 
        DistributedStrategy supports configurations from BuildStrategy and ExecutionStrategy

        """
        __lock_attr = True
        self.strategy = distributed_strategy_pb2.DistributedStrategy()

    def __setattr__(self, key, value):
        if self.__lock_attr and not hasattr(self, key):
            raise TypeError("%s is not a attribute of %s" %
                            (key, self.__class__.__name__))
        object.__setattr__(self, key, value)

    def save_to_prototxt(self, output):
        """
        Serialize current DistributedStrategy to string and save to output file

        Examples:

          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.dgc = True
            strategy.recompute = True
            strategy.recompute_configs = {"checkpoints": ["x"]}
            strategy.save_to_prototxt("dist_strategy.prototxt")
        """
        with open(output, "w") as fout:
            fout.write(str(self.strategy))

    def load_from_prototxt(self, pb_file):
        """
        Load from prototxt file for DistributedStrategy initialization

        Examples:

          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.load_from_prototxt("dist_strategy.prototxt")
        """
        with open(pb_file, 'r') as f:
            self.strategy = google.protobuf.text_format.Merge(
                str(f.read()), self.strategy)

    def _is_strict_auto(self):
        global non_auto_func_called
        if self.strategy.auto and non_auto_func_called:
            return True
        return False

    def __repr__(self):
        spacing = 2
        max_k = 38
        max_v = 38

        length = max_k + max_v + spacing

        h1_format = "    " + "|{{:^{}s}}|\n".format(length)
        h2_format = "    " + "|{{:>{}s}}{}{{:^{}s}}|\n".format(
            max_k, " " * spacing, max_v)

        border = "    +" + "".join(["="] * length) + "+"
        line = "    +" + "".join(["-"] * length) + "+"

        draws = border + "\n"
        draws += h1_format.format("")
        draws += h1_format.format("DistributedStrategy Overview")
        draws += h1_format.format("")

        fields = self.strategy.DESCRIPTOR.fields
        str_res = ""

        env_draws = line + "\n"
        for f in fields:
            if "build_strategy" in f.name or "execution_strategy" in f.name:
                continue
            if "_configs" in f.name:
                continue
            else:
                if isinstance(getattr(self.strategy, f.name), bool):
                    if hasattr(self.strategy, f.name + "_configs"):
                        if getattr(self.strategy, f.name):
                            draws += border + "\n"
                            draws += h1_format.format(
                                "{}=True <-> {}_configs".format(f.name, f.name))
                            draws += line + "\n"
                            my_configs = getattr(self.strategy,
                                                 f.name + "_configs")
                            config_fields = my_configs.DESCRIPTOR.fields
                            for ff in config_fields:
                                if isinstance(
                                        getattr(my_configs,
                                                ff.name), google.protobuf.pyext.
                                        _message.RepeatedScalarContainer):
                                    values = getattr(my_configs, ff.name)
                                    for i, v in enumerate(values):
                                        if i == 0:
                                            draws += h2_format.format(
                                                ff.name, str(v))
                                        else:
                                            draws += h2_format.format(
                                                "", str(v))
                                else:
                                    draws += h2_format.format(
                                        ff.name,
                                        str(getattr(my_configs, ff.name)))
                    else:
                        env_draws += h2_format.format(
                            f.name, str(getattr(self.strategy, f.name)))
                else:
                    env_draws += h2_format.format(
                        f.name, str(getattr(self.strategy, f.name)))

        result_res = draws + border + "\n" + h1_format.format(
            "Environment Flags, Communication Flags")
        result_res += env_draws

        build_strategy_str = border + "\n"
        build_strategy_str += h1_format.format("Build Strategy")
        build_strategy_str += line + "\n"

        fields = self.strategy.build_strategy.DESCRIPTOR.fields
        for f in fields:
            build_strategy_str += h2_format.format(
                f.name, str(getattr(self.strategy.build_strategy, f.name)))
        build_strategy_str += border + "\n"

        execution_strategy_str = h1_format.format("Execution Strategy")
        execution_strategy_str += line + "\n"

        fields = self.strategy.execution_strategy.DESCRIPTOR.fields
        for f in fields:
            execution_strategy_str += h2_format.format(
                f.name, str(getattr(self.strategy.execution_strategy, f.name)))
        execution_strategy_str += border + "\n"

        result_res += build_strategy_str + execution_strategy_str
        return result_res
