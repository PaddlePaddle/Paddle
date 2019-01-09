# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from ....framework import Program
import copy

__all__ = ['Graph', 'ImitationGraph', 'IRGraph']


class Graph(object):
    """
    Base class for all graph.
    """

    def __init__(self):
        pass

    def all_parameters(self):
        """
        Return all the parameters in current graph.
        """
        pass


class ImitationGraph(Graph):
    def __init__(self, program=None, scope=None, in_nodes=[], out_nodes=[]):
        super(ImitationGraph, self).__init__()
        self.program = Program() if program is None else program
        self.scope = scope
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes

    def all_parameters(self):
        return self.program.global_block().all_parameters()

    def get_var(self, var_name):
        return self.program.global_block().var(var_name)

    def clone(self):
        return ImitationGraph(self.program.clone(), self.scope,
                              copy.deepcopy(self.in_nodes),
                              copy.deepcopy(self.out_nodes))

    def ops(self):
        return self.program.global_block().ops

    def program(self):
        return self.program


class IRGraph(Graph):
    pass
