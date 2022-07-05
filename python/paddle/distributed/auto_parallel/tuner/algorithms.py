#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import copy
from abc import ABC, abstractmethod


class AlgorithmBase(ABC):
    """
    An Tuning alogrithm is a function find out an optimal configuration given the selected tuning optimization passes (and the arguments to be tuned). 
    Different optimization passes commibation will correspond to a different algorithm.
    """
    _REGISTERED_ALGORITHMS = {}

    name = None

    @staticmethod
    def _register(algo_name, algo_class):
        assert issubclass(algo_class, AlgorithmBase)
        AlgorithmBase._REGISTERED_ALGORITHMS[algo_name] = algo_class

    def __init__(self, config):
        self._config = config
        self._trial_count = 0
        if self._config.max_trial:
            self._max_trial = self._config.max_trial
        else:
            self._max_trial = float("inf")

    def collect_model_info(self, main_prog, startup_prog):
        """
        Collect the model static info (from programs) that could be used to pruning candidate trials and saving tuning time.
        For instance, model info like memory size of model state and memory size of activation could be 
        used to prune candidated trial and decide the next trial.
        """
        pass

    @abstractmethod
    def get_baseline_trial(self):
        """
        Return the trial with the loosest configurations to run the model base on the selected optimization passes. 
        If the baseline trial fails, it means there is NO available configuration that could run the model given the current selected optimization passes.
        For instance, it would be the configuration that save the Most memory. And if it fails, any other configurations would also fail with OOM.
        it is supported to be first trial and used to prune the trials and saving tuning time.
        """
        pass

    @abstractmethod
    def get_next_trial(self):
        pass

    @abstractmethod
    def update_statue(self, result):
        pass

    @abstractmethod
    def summary(self):
        pass


def register_algor(name):

    def impl(cls):
        AlgorithmBase._register(name, cls)
        cls.name = name
        return cls

    return impl


def new_algorithm(name, config):
    algor_class = AlgorithmBase._REGISTERED_ALGORITHMS.get(name)
    assert algor_class is not None, "Algorithm {} is not defined.".format(name)
    algor_obj = algor_class(config)
    return algor_obj


@register_algor("sharding")
class ShardingStageAlgorithm(AlgorithmBase):

    # TODO import trial class & copy strategy
    def __init__(self, config):
        super().__init__(config)
        self._max_stage = 3

        stage_range = self._config.sharding_configs.get("stage_range", None)
        if stage_range:
            assert set(stage_range).issubset(set([0, 1, 2, 3]))
            if self._max_stage in stage_range:
                stage_range.remove(self._max_stage)
            stage_range.sort()
        else:
            stage_range = [0, 1, 2]
        self.stage_range = stage_range[:]
        self._max_trial = min(self._max_trial, len(self.stage_range))

    def get_baseline_trial(self):

        new_strategy = copy.deepcopy(self._config.dist_strategy)
        config_dict = new_strategy.sharding_configs
        config_dict["stage"] = self._max_stage
        new_strategy.sharding_configs = config_dict
        self._trial_count += 1

        return new_strategy

    def get_next_trial(self):

        if self._trial_count <= self._max_trial:
            new_strategy = copy.deepcopy(self._config.dist_strategy)
            config_dict = new_strategy.sharding_configs
            config_dict["stage"] = self.stage_range[self._trial_count - 1]
            new_strategy.sharding_configs = config_dict
            self._trial_count += 1
            return new_strategy
        else:
            return None

    # TODO should return a trial class and trial name should be its member
    def get_trial_name(self):
        return "Sharing_stage_{}_trial".format(self._trial_count)

    def get_status(self):
        if self._trial_count >= 0 and self._trial_count <= self._max_trial:
            return "RUNNING"
        else:
            return "STOP"

    def update_statue(self, result):
        pass

    def summary(self):
        print(" algorithm.summary() " * 8)
