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

# Notice that the following codes are modified from KerasTuner to implement our own tuner. 
# Please refer to https://github.com/keras-team/keras-tuner/blob/master/keras_tuner/engine/trial.py.

import hashlib
import random
import time
from enum import Enum

from .storable import Storable
from .recorder import MetricsRecorder
from .tunable_space import TunableSpace


class TrialStatus:
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"
    INVALID = "INVALID"


class Trial(Storable):
    def __init__(self, tunable_space, trial_id=None,
                 status=TrialStatus.RUNNING):
        self._id = _generate_trial_id() if trial_id is None else trial_id
        self._space = tunable_space
        self._recorder = MetricsRecorder()
        self._score = None
        self._best_step = None
        self._status = status

    @property
    def id(self):
        return self._id

    @property
    def space(self):
        return self._space

    @property
    def recorder(self):
        return self._recorder

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, score):
        self._score = score

    @property
    def best_step(self):
        return self._best_step

    @best_step.setter
    def best_step(self, best_step):
        self._best_step = best_step

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        self._status = status

    def summary(self):
        print("Tunable space:")
        if self.space.values:
            for tv, value in self.space.values.items():
                print(tv + ":", value)

        if self.score is not None:
            print("Score: {}".format(self.score))

    def get_state(self):
        return {
            "id": self.id,
            "space": self.space.get_state(),
            "recorder": self.recorder.get_state(),
            "score": self.score,
            "best_step": self.best_step,
            "status": self.status,
        }

    def set_state(self, state):
        self._id = state["id"]
        self._space = TunableSpace.from_state(state["space"])
        self._recorder = MetricsRecorder.from_state(state["recorder"])
        self._score = state["score"]
        self._best_step = state["best_step"]
        self._status = state["status"]

    @classmethod
    def from_state(cls, state):
        trial = cls(tunable_space=None)
        trial.set_state(state)
        return trial


def _generate_trial_id():
    s = str(time.time()) + str(random.randint(1, int(1e7)))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]
