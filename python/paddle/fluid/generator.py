#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""This is definition of generator class, which is for managing the state of the algorithm that produces pseudo random numbers."""

from . import core

default_rng_seed_val = 34342423252


class Generator(object):
    """Generator class"""

    def __init__(self, device="CPU"):
        """init"""
        self.device = device
        seed_in = default_rng_seed_val
        if self.device == "CPU":
            self.generator = core.CPUGenerator(seed_in)
        elif self.device == "CUDA":
            self.generator = core.CUDAGenerator(seed_in)
        else:
            raise ValueError("generator class %s does not exist" %
                             generator_class)

    def get_state(self):
        pass

    def set_state(self, state):
        pass

    def manual_seed(self, seed):
        pass

    def seed(self):
        pass

    def initial_seed(self):
        pass
