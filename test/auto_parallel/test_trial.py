# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

from paddle.distributed.auto_parallel.static.tuner import (
    trial as tr,
    tunable_space as ts,
)


class TestTrial(unittest.TestCase):
    def test_trial(self):
        space = ts.TunableSpace()
        space.choice("choice", [0, 1, 2, 3], default=2)
        trial = tr.Trial(space, trial_id="trial-1")
        trial.recorder.register("latency", direction="min")
        trial.recorder.update("latency", 0.1, step=0)
        trial.recorder.update("latency", 0.2, step=1)
        trial.best_step = 0

        self.assertEqual(trial.id, "trial-1")
        self.assertEqual(trial.space.get_value("choice"), 2)
        self.assertEqual(trial.best_step, 0)
        self.assertEqual(trial.status, "RUNNING")

    def test_serialization(self):
        space = ts.TunableSpace()
        space.int_range("int_range", start=1, stop=4, default=2)
        trial = tr.Trial(space, trial_id="trial-2", status="COMPLETED")
        trial.recorder.register("latency", direction="min")
        trial.recorder.update("latency", 0.1, step=0)
        trial.recorder.update("latency", 0.2, step=1)
        trial.best_step = 0

        new_trial = tr.Trial.from_state(trial.get_state())
        self.assertEqual(new_trial.id, "trial-2")
        self.assertEqual(new_trial.space.get_value("int_range"), 2)
        self.assertEqual(new_trial.best_step, 0)
        self.assertEqual(new_trial.status, "COMPLETED")


if __name__ == "__main__":
    unittest.main()
