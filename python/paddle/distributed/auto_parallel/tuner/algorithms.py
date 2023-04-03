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
import logging
from abc import ABC, abstractmethod

from ..utils import get_logger, is_recompute_op
from .trial import OptimizationTunerTrial as Trial
from .trial import TrialStatus


class AlgorithmBase(ABC):
    """
    An Tuning algorithm is a class to find out an optimal configuration
    given the selected tuning optimization pass(es) and the arguments to be tuned.
    Different optimization pass(es) will correspond to a different algorithm,
    where different search space **pruning rules** will applied.

    In another word, the key "algorithm" for this class is the
    search space pruning rules specific for the given optimization scenario.
    """

    _REGISTERED_ALGORITHMS = {}

    name = None

    @staticmethod
    def _register(algo_name, algo_class):
        assert issubclass(algo_class, AlgorithmBase)
        AlgorithmBase._REGISTERED_ALGORITHMS[algo_name] = algo_class

    def __init__(self, config):
        self._config = config
        self._init_spaces()
        self._logger = get_logger(logging.INFO)
        self._changed_configs = []

    @property
    def changed_configs(self):
        return self._changed_configs[:]

    def collect_model_info(self, main_prog, startup_prog):
        """
        Collect the model static info (from programs) that could be used to
        pruning candidate trials and saving tuning time. For instance,
        model info like number of model parameters and activation memory could be
        used to prune candidated trial and decide the next trial.
        """
        pass

    @abstractmethod
    def _init_spaces(self):
        pass

    @abstractmethod
    def next_trial(self):
        pass

    @abstractmethod
    def update(self, results):
        """
        Update the algorithm with the results of last trial. Using this information is used to
        pruning the search space of the future trial.
        """
        pass

    def get_config_from_trial(self, trial):
        """
        Return a new fleet.DistributedStrategy with the configurations in trial.
        """
        assert len(self._changed_configs) > 0
        new_strategy = copy.deepcopy(self._config.dist_strategy)
        for name in self._changed_configs:
            config = getattr(trial.space, name)
            setattr(new_strategy, name, config)
        return new_strategy


def register_algor(name):
    def impl(cls):
        AlgorithmBase._register(name, cls)
        cls.name = name
        return cls

    return impl


def new_algorithm(name, config):
    algor_class = AlgorithmBase._REGISTERED_ALGORITHMS.get(name)
    assert algor_class is not None, f"Algorithm {name} is not defined."
    algor_obj = algor_class(config)
    return algor_obj


@register_algor("sharding")
class ShardingStageAlgorithm(AlgorithmBase):

    # TODO import trial class & copy strategy
    def __init__(self, config):
        super().__init__(config)
        self._changed_configs = ["sharding"]

    def _init_spaces(self):
        self._max_stage = 3
        self._trial_idx = 0

        stage_range = self._config.sharding.get("tuning_range", None)
        if stage_range:
            assert set(stage_range).issubset(
                {0, 1, 2, 3}
            ), "Sharding Stage should belong into range within 0 - 3 but got {}.".format(
                stage_range
            )
            stage_range.sort(reverse=True)
        else:
            stage_range = list(range(self._max_stage + 1)).sort(reverse=True)

        self._stage_range = stage_range[:]
        self._total_num_trial = len(self._stage_range)

    def next_trial(self):

        if self._trial_idx < self._total_num_trial:

            stage = self._stage_range[self._trial_idx]

            new_strategy = copy.deepcopy(self._config.dist_strategy)
            sharding = new_strategy.sharding
            sharding.stage = stage

            name = f"trial-sharding-stage{stage}"
            trial = Trial(new_strategy, name, self.changed_configs)

            return trial
        else:
            return Trial(None, None, None, status=TrialStatus.STOPPED)

    def update(self, results):

        et = results.get("ErrorType", None)
        if et and et == "ResourceExhaustedError":
            self._trial_idx = self._total_num_trial
            self._logger.info(
                "Last trial is failed with OOM, all remaining trials are pruned to save time !"
            )
        else:
            self._trial_idx += 1


@register_algor("recompute")
class ReccomputeCheckpointAlgorithm(AlgorithmBase):
    def __init__(self, config):
        super().__init__(config)
        self._changed_configs = ["recompute"]

    def collect_model_info(self, main_prog, startup_prog):
        segments = []
        for op in main_prog.global_block().ops:
            if not is_recompute_op(op):
                continue

            seg_name = op.attr('op_namescope')
            if seg_name not in segments:
                segments.append(seg_name)

        self._total_num_trial = len(segments)
        self._tuning_segments = list(range(len(segments)))
        self._trail_left = 0
        self._trail_right = len(segments) - 1
        self._trial_idx = int(0 + (len(segments) - 1) / 2)

    def _init_spaces(self):
        self._recompute_mode = "all"

    def next_trial(self):
        if self._trial_idx < self._total_num_trial:
            if self._recompute_mode == "all":
                self._recompute_flag = False
                new_strategy = copy.deepcopy(self._config.dist_strategy)
                name = "trial-recompute-all-segments"
                return Trial(new_strategy, name, self.changed_configs)
            elif self._recompute_mode == "none":
                self._recompute_flag = False
                new_strategy = copy.deepcopy(self._config.dist_strategy)
                recompute = new_strategy.recompute
                recompute.enable = False
                name = "trial-recompute-none-segments"
                return Trial(new_strategy, name, self.changed_configs)
            elif self._recompute_mode == "part":
                new_no_recompute = self._tuning_segments[: self._trial_idx]
                new_strategy = copy.deepcopy(self._config.dist_strategy)
                recompute = new_strategy.recompute
                recompute.no_recompute_segments.extend(new_no_recompute)
                name = "trial-recompute-part-segments-idx{}".format(
                    self._trial_idx
                )
                return Trial(new_strategy, name, self.changed_configs)
        else:
            return Trial(None, None, None, status=TrialStatus.STOPPED)

    def update(self, results):

        et = results.get("ErrorType", None)
        if self._recompute_mode == "all":
            if et and et == "ResourceExhaustedError":
                self._trial_idx = self._total_num_trial
                self._logger.info(
                    "Recompute all candidate segments is failed with OOM, please reduce model size or batch size."
                )
            else:
                self._recompute_mode = "none"
        elif self._recompute_mode == "none":
            if et and et == "ResourceExhaustedError":
                self._recompute_mode = "part"
            else:
                self._trial_idx = self._total_num_trial
                self._logger.info(
                    "Recompute is unnecessary for this model size, which will reduce the Throughput."
                )
        else:
            if self._trail_left >= self._trail_right:
                self._trial_idx = self._total_num_trial
            elif et and et == "ResourceExhaustedError":
                self._trail_left = self._trail_left
                self._trail_right = self._trial_idx - 1
                self._trial_idx = int(
                    self._trail_left
                    + (self._trail_right - self._trail_left) / 2
                )
            else:
                self._trail_left = self._trial_idx + 1
                self._trail_right = self._trail_right
                self._trial_idx = int(
                    self._trail_left
                    + (self._trail_right - self._trail_left) / 2
                )
