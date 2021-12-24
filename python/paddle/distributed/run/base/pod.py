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


class Status(object):
    def __init__(self):
        pass


class _PodProto(object):
    def __init__(self):
        self.name = ""
        self.spec = ""
        self.enable_elastic = False
        self.init_containers: List[Container] = []
        self.containers: List[Container] = []
        self.resource: Resource = None
        self.status: Status = None
        self.rank = 0
        self.replicas = 0

    def json(self):
        pass

    def add_container(self, c, is_init=False):
        assert (isinstance(c, Container))
        if is_init:
            self.init_containers.append(c)
        else:
            self.containers.append(c)

    def add(self, item):
        if isinstance(c, Container):
            self.containers.append(c)


class Pod(_PodProto):
    def create(self):
        for i in self.init_containers:
            i.run()

        for i in self.containers:
            i.run()

    def stop(self):
        pass

    def status(self):
        return None

    def detach(self):
        pass

    def logs(self):
        pass
