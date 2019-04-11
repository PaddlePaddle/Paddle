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

from ....compiler import CompiledProgram
from ....data_feeder import DataFeeder
from .... import executor
from .graph_wrapper import GraphWrapper

__all__ = ['SlimGraphExecutor']


class SlimGraphExecutor(object):
    """
    Wrapper of executor used to run GraphWrapper.
    """

    def __init__(self, place):
        self.exe = executor.Executor(place)
        self.place = place

    def run(self, graph, scope, data=None):
        """
        Runing a graph with a batch of data.
        Args:
            graph(GraphWrapper): The graph to be executed.
            scope(fluid.core.Scope): The scope to be used.
            data(list<tuple>): A batch of data. Each tuple in this list is a sample.
                               It will feed the items of tuple to the in_nodes of graph.
        Returns:
            results(list): A list of result with the same order indicated by graph.out_nodes.
        """
        assert isinstance(graph, GraphWrapper)
        if data is not None:
            feeder = DataFeeder(
                feed_list=graph.in_nodes.values(),
                place=self.place,
                program=graph.program)
            feed = feeder.feed(data)

        fetch_list = graph.out_nodes.values()
        program = graph.compiled_graph if graph.compiled_graph else graph.program
        results = self.exe.run(program,
                               scope=scope,
                               fetch_list=fetch_list,
                               feed=feed)
        return results
