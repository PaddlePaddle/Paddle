#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from .proto import distributed_strategy_pb2
from ..fluid.framework import Variable


class DistributedStrategy(object):
    def __init__(self):
        self.strategy = distributed_strategy_pb2.DistributedStrategy()

    @property
    def amp(self):
        return self.strategy.amp

    @amp.setter
    def amp(self, flag):
        if isinstance(flag, bool):
            self.strategy.amp = flag
        else:
            print("WARNING: amp should have value of bool type")

    @property
    def amp_loss_scaling(self):
        return self.strategy.amp_loss_scaling

    @amp_loss_scaling.setter
    def amp_loss_scaling(self, value):
        if isinstance(value, int):
            self.strategy.amp_loss_scaling = value
        else:
            print("WARNING: amp_loss_scaling should have value of int type")

    @property
    def recompute(self):
        return self.strategy.recompute

    @recompute.setter
    def recompute(self, flag):
        if isinstance(flag, bool):
            self.strategy.recompute = flag
        else:
            print("WARNING: recompute should have value of bool type")

    @property
    def recompute_checkpoints(self):
        return self.strategy.recompute_checkpoints

    @recompute_checkpoints.setter
    def recompute_checkpoints(self, checkpoints):
        if isinstance(checkpoints, list):
            str_list = True
            var_list = True
            for item in checkpoints:
                if not isinstance(item, str):
                    str_list = False
                if not isinstance(item, Variable):
                    var_list = False

            assert (str_list and var_list) == False
            if str_list:
                self.strategy.ClearField("recompute_checkpoints")
                self.strategy.recompute_checkpoints.extend(checkpoints)
            elif var_list:
                names = [x.name for x in checkpoints]
                self.strategy.ClearField("recompute_checkpoints")
                self.strategy.recompute_checkpoints.extend(names)
            else:
                print(
                    "WARNING: recompute_checkpoints should have value of list[Variable] or list[name] type"
                )
        else:
            print(
                "WARNING: recompute_checkpoints should have value of list[Variable] or list[name] type"
            )

    @property
    def pipeline(self):
        return self.strategy.pipeline

    @pipeline.setter
    def pipeline(self, flag):
        if isinstance(flag, bool):
            self.strategy.pipeline = flag
        else:
            print("WARNING: pipeline should have value of bool type")

    @property
    def pipeline_micro_batch(self):
        return self.strategy.pipeline_micro_batch

    @pipeline_micro_batch.setter
    def pipeline_micro_batch(self, value):
        if isinstance(value, int):
            self.strategy.pipeline_micro_batch = value
        else:
            print("WARNING: pipeline micro batch should have value of int type")

    @property
    def localsgd(self):
        return self.strategy.localsgd

    @localsgd.setter
    def localsgd(self, flag):
        if isinstance(flag, bool):
            self.strategy.localsgd = flag
        else:
            print("WARNING: localsgd should have value of bool type")

    @property
    def localsgd_k_step(self):
        return self.strategy.localsgd_k_step

    @localsgd_k_step.setter
    def localsgd_k_step(self, value):
        if isinstance(value, int):
            self.strategy.localsgd_k_step = value
        else:
            print("WARNING: localsgd_k_step should have value of int type")

    @property
    def dgc(self):
        return self.strategy.dgc

    @dgc.setter
    def dgc(self, flag):
        if isinstance(flag, bool):
            self.strategy.dgc = flag
        else:
            print("WARNING: dgc should have value of bool type")

    @property
    def hierachical_allreduce(self):
        return self.strategy.hierachical_allreduce

    @hierachical_allreduce.setter
    def hierachical_allreduce(self, flag):
        if isinstance(flag, bool):
            self.strategy.hierachical_allreduce = flag
        else:
            print(
                "WARNING: hierachical_allreduce should have value of bool type")

    @property
    def nccl_comm_num(self):
        return self.strategy.nccl_comm_num

    @nccl_comm_num.setter
    def nccl_comm_num(self, value):
        if isinstance(value, int):
            self.strategy.nccl_comm_num = value
        else:
            print("WARNING: nccl_comm_num should have value of int type")

    @property
    def gradient_merge(self):
        return self.strategy.gradient_merge

    @gradient_merge.setter
    def gradient_merge(self, flag):
        if isinstance(flag, bool):
            self.strategy.gradient_merge = flag
        else:
            print("WARNING: gradient_merge should have value of bool type")

    @property
    def gradient_merge_k_step(self):
        return self.strategy.gradient_merge_k_step

    @gradient_merge_k_step.setter
    def gradient_merge_k_step(self, value):
        if isinstance(value, int):
            self.strategy.gradient_merge_k_step = value
        else:
            print(
                "WARNING: gradient_merge_k_step should have value of int type")

    @property
    def sequential_execution(self):
        return self.strategy.sequential_execution

    @sequential_execution.setter
    def sequential_execution(self, flag):
        if isinstance(flag, bool):
            self.strategy.sequential_execution = flag
        else:
            print(
                "WARNING: sequential_execution should have value of bool type")

    @property
    def sync(self):
        return self.strategy.sync

    @sync.setter
    def sync(self, flag):
        if isinstance(flag, bool):
            self.strategy.sync = flag
        else:
            print("WARNING: sync should have value of bool type")

    @property
    def async(self):
        return self.strategy.async

    @async.setter
    def async(self, flag):
        if isinstance(flag, bool):
            self.strategy.async = flag
        else:
            print("WARNING: async should have value of bool type")

    @property
    def async_k_step(self):
        return self.strategy.async_k_step

    @async_k_step.setter
    def async_k_step(self, value):
        if isinstance(value, int):
            self.strategy.async_k_step = value
        else:
            print("WARNING: async_k_step should have value of int type")

    @property
    def auto(self):
        return self.strategy.auto

    @auto.setter
    def auto(self, flag):
        if isinstance(flag, bool):
            self.strategy.auto = flag
        else:
            print("WARNING: auto should have value of bool type")
