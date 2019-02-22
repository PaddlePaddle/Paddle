#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

import abc
from abc import abstractmethod
from .... import executor
from .graph import IRGraph, ImitationGraph

__all__ = ['get_executor']


class GraphExecutor(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, place):
        self.place = place

    @abstractmethod
    def run(self, graph, feches=None, feed=None):
        pass


class IRGraphExecutor(GraphExecutor):
    def run(self, grah, fetches, feed=None):
        pass


class ImitationGraphExecutor(GraphExecutor):
    def __init__(self, place):
        super(ImitationGraphExecutor, self).__init__(place)
        self.exe = executor.Executor(place)

    def run(self, graph, scope=None, fetches=None, feed=None):
        assert isinstance(graph, ImitationGraph)
        fetch_list = None
        if fetches:
            fetch_list = [
                graph.program.global_block().var(name) for name in fetches
            ]
        results = self.exe.run(graph.program,
                               scope=scope,
                               fetch_list=fetch_list,
                               feed=feed)
        return results


def get_executor(graph, place):
    if isinstance(graph, ImitationGraph):
        return ImitationGraphExecutor(place)
    if isinstance(graph, IRGraph):
        return IRGraphExecutor(place)
