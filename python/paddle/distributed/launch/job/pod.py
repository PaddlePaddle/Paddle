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

from __future__ import annotations

from .container import Container

from .status import Status

import random
import time


class PodSepc:
    def __init__(self):
        self._name = ''.join(
            random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(6)
        )

        # by controller
        self._init_containers: list[Container] = []
        self._containers: list[Container] = []

        # self.resource: Resource = None
        # self.status: Status = None

        self._rank = -1
        self._init_timeout = None
        self._restart = -1
        self._replicas = 0  # number of containers
        self._exit_code = 0


class Pod(PodSepc):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Pod: {}, replicas {}, status {}".format(
            self.name, self.replicas, self.status
        )

    def failed_container(self):
        cs = []
        for c in self._containers:
            if c.status == Status.FAILED:
                cs.append(c)
        return cs

    @property
    def name(self):
        return self._name

    @property
    def replicas(self):
        return self._replicas

    @replicas.setter
    def replicas(self, r):
        self._replicas = max(r, 1)

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, r):
        self._rank = r

    @property
    def restart(self):
        return self._restart

    @property
    def containers(self):
        return self._containers

    def add_container(self, c):
        c.rank = len(self._containers)
        self._containers.append(c)

    @property
    def init_containers(self):
        return self._init_containers

    def add_init_container(self, c):
        c.rank = len(self._init_containers)
        self._init_containers.append(c)

    @property
    def exit_code(self):
        for c in self._containers:
            if c.exit_code != 0:
                return c.exit_code
        return 0

    def deploy(self):
        # init container should stop before run containers
        for i in self._init_containers:
            i.start()
            i.wait(self._init_timeout)

        for c in self._containers:
            c.start()

        self._restart += 1

    def stop(self, sigint=15, timeout=None):
        for c in self._containers:
            if isinstance(sigint, int) and timeout is None:
                c.send_signal(sigint)
            else:
                c.terminate()

        if isinstance(timeout, int):
            if not self.join(timeout):
                for c in self._containers:
                    c.terminate(force=True)
                return False
            else:
                return True

    def join(self, timeout=None):
        for c in self._containers:
            if not c.wait(timeout):
                return False
        return True

    @property
    def status(self):
        if self.is_failed():
            return Status.FAILED

        if self.is_completed():
            return Status.COMPLETED

        if self.is_running():
            return Status.RUNNING

        return Status.READY

    def reset(self):
        self._init_containers = []
        self._containers = []

    def is_failed(self):
        for c in self._containers:
            if c.status == Status.FAILED:
                return True
        return False

    def is_completed(self):
        for c in self._containers:
            if c.status != Status.COMPLETED:
                return False
        return True

    def is_running(self):
        for c in self._containers:
            if c.status != Status.RUNNING:
                return False
        return True

    def logs(self, idx=None):
        if idx is None:
            self._containers[0].logs()
        else:
            self._containers[idx].logs()

    def tail(self, idx=None):
        if idx is None:
            self._containers[0].tail()
        else:
            self._containers[idx].tail()

    def watch(
        self,
        all_list=[Status.COMPLETED],
        any_list=[Status.FAILED],
        interval=1,
        timeout=-1,
    ):
        '''
        watch return if any container status in any_list
        or all container status in all_list
        '''
        end = time.time() + timeout
        while timeout < 0 or time.time() < end:
            for c in self._containers:
                if c.status in any_list:
                    return c.status

            s = [c.status for c in self._containers]
            if len(set(s)) == 1 and s[0] in all_list:
                return s[0]

            time.sleep(interval)
