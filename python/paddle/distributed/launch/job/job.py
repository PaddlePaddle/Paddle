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


class JobMode:
    COLLECTIVE = 'collective'
    PS = 'ps'
    HETER = 'heter'


class Job:
    def __init__(self, jid='default', mode=JobMode.COLLECTIVE, nnodes="1"):
        self._mode = mode
        self._id = jid

        self._replicas = 0
        self._replicas_min = self._replicas
        self._replicas_max = self._replicas
        self._elastic = False

        self.set_replicas(str(nnodes))

    def __str__(self):
        return "Job: {}, mode {}, replicas {}[{}:{}], elastic {}".format(
            self.id,
            self.mode,
            self._replicas,
            self._replicas_min,
            self._replicas_max,
            self.elastic,
        )

    @property
    def mode(self):
        return self._mode

    @property
    def id(self):
        return self._id

    @property
    def elastic(self):
        return self._elastic

    @property
    def replicas(self):
        return self._replicas

    @property
    def replicas_min(self):
        return self._replicas_min

    @property
    def replicas_max(self):
        return self._replicas_max

    @replicas.setter
    def replicas(self, replicas):
        self._replicas = replicas

    def set_replicas(self, nnodes: str):
        np = str(nnodes) if nnodes else '1'

        if ':' in np:
            nps = np.split(':')
            self._replicas_min, self._replicas_max = int(nps[0]), int(nps[1])
            self._replicas = self._replicas_max  # default to max

            self._elastic = True
        else:
            self._replicas = int(np)
            self._replicas_min, self._replicas_max = (
                self._replicas,
                self._replicas,
            )

            self._elastic = False
