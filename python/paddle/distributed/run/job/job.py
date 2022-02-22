# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


class Job(object):
    def __init__(self):
        self.mode = JobMode.COLLECTIVE
        self.id = 'default'

        self.ips = []
        self.ports = []
        self.endpoints = []

        self.replicas = 0
        self.replicas_min = self.replicas
        self.replicas_max = self.replicas
        self.elastic = False

    def set_replicas(self, np: str):
        np = np if np else '1'

        if ':' in np:
            nps = np.split(':').strip()
            self.replicas_min, self.replicas_max = int(nps[0]), int(nps[1])
            self.replicas = self.replicas_max  # default to max

            self.elastic = True
        else:
            self.replicas = int(np)
            self.replicas_min, self.replicas_max = self.replicas, self.replicas

            self.elastic = False
