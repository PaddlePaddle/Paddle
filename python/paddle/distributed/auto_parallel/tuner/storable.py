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

# Notice that the following codes are modified from KerasTuner for a different purpose. 
# Please refer to https://github.com/keras-team/keras-tuner/blob/master/keras_tuner/engine/metrics_tracking.py.

import json


class Storable(object):
    def get_state(self):
        raise NotImplementedError

    def set_state(self, state):
        raise NotImplementedError

    def save(self, path):
        state = self.get_state()
        state_json = json.dumps(state)
        with open(path, "w") as f:
            f.write(state_json)
        return str(path)

    def load(self, path):
        with open(path, "r") as f:
            state_data = f.read()
        state = json.loads(state_data)
        self.set_state(state)
