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

<<<<<<< HEAD
import copy
import os

=======
import os
import copy
import pathlib

import paddle
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
from ..strategy import Strategy

_tuning_supported_passes = ["sharding", "recompute"]


def _get_pass_config(strategy, pass_name):
    config = getattr(strategy, pass_name)
    return config


<<<<<<< HEAD
class TuningConfig:
    """
    A uniform config wrap:
    distributed strategy: the user defined configuration for optimization pass
    tuning config: configuration for the tuning process: mode (profile or cost model), log dir, extra tuning config for optimization like search range for specific
    """

    def __init__(self, strategy):
=======
class TuningConfig(object):
    """
    A uniform config wrap:
    distributed strategy: the user defined configuration for optimization pass
    tuning config: configuration for the tuning process: mode (profile or cost model), log dir, extra tuning config for optimization like search range for specific 
    """

    def __init__(self, user_config, strategy):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        if not isinstance(strategy, Strategy):
            raise TypeError("'strategy' must be object of class `Strategy`.")

<<<<<<< HEAD
=======
        if not user_config:
            user_config = {}

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self._tuning_passes_name = set()
        self._dist_strategy = copy.deepcopy(strategy)
        self._mode = None
        self._profile_start_step = None
        self._profile_end_step = None
        self._project_dir = None
        self._max_num_trial = None
        self._early_stop = None
<<<<<<< HEAD
        self._debug = None

        self._initialize()
=======
        self._verbose = None

        self._initialize(user_config)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

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
    def project_dir(self):
        return self._project_dir

    @property
    def tuning_passes_name(self):
        return self._tuning_passes_name

    @property
    def max_num_trial(self):
        return self._max_num_trial

    @property
    def early_stop(self):
        return self._early_stop

    @property
<<<<<<< HEAD
    def debug(self):
        return self._debug
=======
    def verbose(self):
        return self._verbose
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    @property
    def dist_strategy(self):
        return self._dist_strategy

    # initialize config with user define value or default value
<<<<<<< HEAD
    def _initialize(self):
        tuning_strategy = self._dist_strategy.tuning

        self._mode = tuning_strategy.get("mode", "PROFILE")
        self._profile_start_step = tuning_strategy.get("profile_start_step", 10)
        self._profile_end_step = tuning_strategy.get("profile_end_step", 30)
        self._max_num_trial = tuning_strategy.get("max_num_trial", 50)
        self._early_stop = tuning_strategy.get("early_stop", None)
        self._debug = tuning_strategy.get("debug", False)

        project_dir = tuning_strategy.get("project_dir", None)
=======
    def _initialize(self, user_config):

        self._mode = user_config.get("mode", "PROFILE")

        self._profile_start_step = user_config.get("profile_start_step", 10)

        self._profile_end_step = user_config.get("profile_end_step", 30)

        self._max_num_trial = user_config.get("max_num_trial", 50)

        self._early_stop = user_config.get("early_stop", None)

        self._verbose = user_config.get("verbose", False)

        project_dir = user_config.get("project_dir", None)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if not project_dir:
            project_dir = os.path.join(os.getcwd(), "OptimizationTuning")
        self._project_dir = project_dir

        for p in _tuning_supported_passes:
<<<<<<< HEAD
            if (
                getattr(self._dist_strategy, p)
                and _get_pass_config(self._dist_strategy, p).enable_tuning
            ):
                # TODO distinguish different args of each passes
                self._tuning_passes_name.add(p)

                p_strategy = getattr(self._dist_strategy, p)
                self.__dict__[p] = p_strategy

                # # TODO verify the user defined configs
                # tuning_config_for_pass = tuning_strategy.get(p, None)
                # if tuning_config_for_pass:
                #     for k, v in tuning_config_for_pass.items():
                #         self.__dict__[p][k] = v
=======
            if getattr(self._dist_strategy, p) and _get_pass_config(
                    self._dist_strategy, p).enable_tuning:
                # TODO distinguish different args of each passes
                self._tuning_passes_name.add(p)

                config_name = p
                p_dict = getattr(self._dist_strategy, config_name)
                self.__dict__[config_name] = p_dict

                # TODO verify the user defined configs
                user_config_for_pass = user_config.get(p, None)
                if user_config_for_pass:
                    for k, v in user_config_for_pass.items():
                        self.__dict__[config_name][k] = v
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    # (NOTE)tuning config ONLY wraps dist strategy for pass config which is to be tuned
    def __getattr__(self, item):
        return getattr(self._dist_strategy, item)
