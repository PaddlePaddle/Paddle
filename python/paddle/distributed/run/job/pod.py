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

from collections import OrderedDict
from .container import Container

import random


class Status(object):
    def __init__(self):
        pass


class PodSepc(object):
    def __init__(self):
        self.name = ''.join(
            random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(6))

        self.endpoints = []

        # by controller
        self.init_containers: List[Container] = []
        self.containers: List[Container] = []
        self.resource: Resource = None
        self.status: Status = None
        self.rank = -1
        self.replicas = 0  # number of containers
        self.init_timeout = 120  # 2 min timeout for each init container


class Pod(PodSepc):
    def __init__(self):
        super().__init__()

    def json(self):
        pass

    def __str__(self):
        return "Pod: {}, replicas {}".format(self.name, self.replicas)

    def deploy(self):
        for i in self.init_containers:
            i.start(self.init_timeout)

        for c in self.containers:
            c.start()

    def stop(self, sigint):
        for c in self.containers:
            force = True if sigint == 9 else False
            c.terminate(force)

    def join(self):
        for c in self.containers:
            c.wait(None)

    def status(self):
        return None

    def logs(self, idx=0):
        self.containers[idx].logs()
