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

from .storable import Storable
from .trail import TrialStatus


class Tuner(Storable):
    def __init__(self,
                 tunable_space=None,
                 objective=None,
                 direction=None,
                 max_trials=None,
                 tuner_id=None,
                 seed=None,
                 logger=None):
        self._space = tunable_space
        self._objective = objective
        self._direction = direction
        self._max_trials = max_trials
        self._tuner_id = tuner_id
        self._seed = seed
        self._logger = logger
        self._trials = {}

    def tune(self):
        pass

    def populate_initial_space(self):
        pass

    def populate_space(self, trail_id):
        pass

    def get_space(self):
        pass

    def update_space(self):
        pass

    def _create_trial(self):
        pass

    def _eval_trial(self, trial_id, *args, **kwargs):
        pass

    def _update_trial(self, trial_id, metrics, step=0):
        pass

    def _end_trial(self, trial_id, status=TrialStatus.COMPLETED):
        pass

    def get_best_trials(self, num_trials=1):
        pass
