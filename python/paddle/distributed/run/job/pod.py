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

from .status import Status

import random
import time


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
        self.init_timeout = 120  # 2 min timeout for each init container
        self.restart = 0

        self.replicas = 0  # number of containers

        self._exit_code = 0


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

    def stop(self, sigint=0):
        for c in self.containers:
            force = True if sigint == 9 else False
            c.terminate(force)

    def join(self):
        for c in self.containers:
            c.wait(None)

    @property
    def exit_code(self):
        for c in self.containers:
            if c.exit_code() != 0:
                return c.exit_code()
        return 0

    def status(self):
        if self.is_failed():
            return Status.FAILED

        if self.is_completed():
            return Status.COMPLETED

        return Status.UNKNOWN

    def reset(self):
        self.init_containers = []
        self.containers = []

    def is_failed(self):
        for c in self.containers:
            if c.status() == Status.FAILED:
                return True
        return False

    def is_completed(self):
        for c in self.containers:
            if c.status() != Status.COMPLETED:
                return False
        return True

    def logs(self, idx=0):
        self.containers[idx].logs()

    def watch(self,
              all_list=[Status.COMPLETED],
              any_list=[Status.FAILED],
              interval=1,
              timeout=-1):
        '''
        watch return if any container status in any_list
        or all container status in all_list
        '''
        st = time.time()
        while timeout < 0 or st + timeout > time.time():
            for c in self.containers:
                if c.status() in any_list:
                    return c.status()

            s = [c.status() for c in self.containers]
            if len(set(s)) == 1 and s[0] in all_list:
                return s[0]

            time.sleep(interval)
