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
# limitations under the License


class CostEstimator:
    def __init__(self,
                 program,
                 cluster=None,
                 dist_context=None,
                 mode="modeling"):
        self._program = program
        self._cluster = cluster
        self._dist_context = dist_context
        self._check_mode(mode)
        self._mode = mode
        self._global_cost = None
        self._local_cost = {}

    @property
    def program(self):
        return self._program

    @property
    def dist_context(self):
        return self._dist_context

    @property
    def cluster(self):
        return self._cluster

    @property
    def mode(self):
        return self._mode

    @property
    def global_cost(self):
        return self._global_cost

    @property
    def local_cost(self):
        return self._local_cost

    def get_op_cost(self):
        return 0

    def get_tensor_cost(self):
        return 0

    def get_global_cost(self):
        return 0

    def get_local_cost(self, rank=None):
        return 0

    def _check_mode(self, mode):
        if mode not in ["modeling", "profiling"]:
            raise ValueError(
                "Just support modeling and profiling, but got {}".format(mode))
