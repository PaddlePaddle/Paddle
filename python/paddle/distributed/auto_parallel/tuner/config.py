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

import os
import copy
import pathlib

import paddle
from paddle.distributed import fleet

_tuning_supported_passes = ["sharding", "recompute"]
_strategy_config_suffiex = "_configs"


def _get_pass_config(strategy, pass_name):
    config_name = pass_name + _strategy_config_suffiex
    config = getattr(strategy, config_name)
    return config


class TuningConfig(object):
    """
    A uniform config wrap:
    distributed strategy: the user defined configuration for optimization pass
    tuning config: configuration for the tuning process: mode (profile or cost model), log dir, extra tuning config for optimization like search range for specific 
    """

    def __init__(self, user_config, strategy):

        if not isinstance(strategy, fleet.DistributedStrategy):
            raise TypeError(
                "'strategy' must be object of class `fleet.DistributedStrategy`."
            )

        if not user_config:
            user_config = {}

        self._tuning_passes_configs = {}
        self._tuning_passes_name = set()
        self._dist_strategy = copy.deepcopy(strategy)
        self._mode = None
        self._profile_start_step = None
        self._profile_end_step = None
        self._log_dir = None
        self._max_trial = None

        self._initialize(user_config)

    @property
    def mode(self):
        return self._mode

    @property
    def profile_start_step(self):
        return self._profile_start_step

    @property
    def profile_end_step(self):
        return self._profile_end_step

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def tuning_passes_name(self):
        return self._tuning_passes_name

    @property
    def max_trial(self):
        return self._max_trial

    @property
    def dist_strategy(self):
        return self._dist_strategy

    # initialize config with user define value or default value
    def _initialize(self, user_config):

        self._mode = user_config.get("mode", "PROFILE")

        self._profile_start_step = user_config.get("profile_start_step", 10)

        self._profile_end_step = user_config.get("profile_end_step", 30)

        log_dir = user_config.get("log_dir", None)
        if not log_dir:
            log_dir = os.path.join(os.getcwd(), "tuning_results")
        self._log_dir = log_dir
        if not os.path.exists(self._log_dir):
            if paddle.distributed.get_rank() == 0:
                pathlib.Path(self._log_dir).mkdir(parents=True, exist_ok=True)

        for p in _tuning_supported_passes:
            if getattr(self._dist_strategy, p) and _get_pass_config(
                    self._dist_strategy, p)["enable_tuning"]:
                # TODO distinguish different args of each passes
                self._tuning_passes_name.add(p)

                config_name = p + _strategy_config_suffiex
                p_dict = getattr(self._dist_strategy, config_name)
                self._tuning_passes_configs[config_name] = p_dict

                # TODO verify the user defined configs
                user_config_for_pass = user_config.get(p, None)
                if user_config_for_pass:
                    for k, v in user_config_for_pass.items():
                        self._tuning_passes_configs[config_name][k] = v

    # (NOTE)tuning config ONLY wraps dist strategy for pass config which is to be tuned
    # (NOTE) could NOT be deepcopy
    def __getattr__(self, item):
        if item in self._tuning_passes_configs:
            return self._tuning_passes_configs[item]
        else:
            return getattr(self._dist_strategy, item)

    # def __setattr__(self, key, value):
    #     if key not in self._tuning_passes_configs:
    #         raise TypeError("{} is not a config of tuning passes {}".format(key, [p for p in self._tuning_passes_name]))
    #     self._tuning_passes_configs[key] = value
