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

from ....core import CPUPlace
from ..graph import get_executor

__all__ = ['Context', 'CompressPass']


class Context(object):
    """
    The context in the process of compression.
    Args:
        exe: The executor used to execute graph.
        graph: The graph to be compressed.
        scope: The scope used to execute graph.
        program_exe: The program_exe is used to execute the program
                     created for modifying the variables in scope.
    """

    def __init__(self, exe, graph, scope, program_exe=None):
        # The total number of epoches to be trained.
        self.epoch = 0
        # Current epoch
        self.epoch_id = 0
        # Current batch
        self.batch_id = 0
        self.exe = exe
        self.graph = graph
        self.scope = scope
        self.program_exe = program_exe


class CompressPass(object):
    """
    The pass used to compress model.
    Args:
        place: The device used in compression.
        data_reader: The data_reader used to run graph.
        data_feeder: The data_feeder used to run graph.
        scope: The scope used to run graph.
        metrics: The metrics for evaluating model.
        epoch: The total epoches of trainning in compression.
        program_exe: The program_exe is used to execute the program
                     created for modifying the variables in scope.
    """

    def __init__(self,
                 place=None,
                 data_reader=None,
                 data_feeder=None,
                 scope=None,
                 metrics=None,
                 epoch=None,
                 program_exe=None):
        self.strategies = []
        self.place = CPUPlace() if place is None else place
        self.data_reader = data_reader
        self.data_feeder = data_feeder
        self.scope = scope
        self.metrics = metrics
        self.epoch = epoch
        self.program_exe = program_exe

    def add_strategy(self, strategy):
        """
        Add a strategy to current compress pass.
        Args:
            strategy: The strategy to be added into current compress pass.
        """
        self.strategies.append(strategy)
        self.epoch = max(strategy.end_epoch, self.epoch)

    def apply(self, graph):
        """
        Compress a model.
        Args:
            graph: The target graph to be compressed.
        """
        self.executor = get_executor(graph, self.place)
        context = Context(
            self.executor, graph, self.scope, program_exe=self.program_exe)

        for strategy in self.strategies:
            strategy.on_compress_begin(context)

        for epoch in range(self.epoch):

            for strategy in self.strategies:
                strategy.on_epoch_begin(context)

            for data in self.data_reader():

                for strategy in self.strategies:
                    strategy.on_batch_begin(context)
                fetches = None
                if self.metrics:
                    fetches = self.metrics.values()
                feed = None
                if self.data_feeder:
                    feed = self.data_feeder.feed(data)
                results = self.executor.run(graph,
                                            fetches=fetches,
                                            scope=self.scope,
                                            feed=feed)
                if results:
                    print("results: {}".format(
                        zip(self.metrics.keys(), results)))
                for strategy in self.strategies:
                    strategy.on_batch_end(context)
                context.batch_id += 1

            for strategy in self.strategies:
                strategy.on_epoch_end(context)
            context.epoch_id += 1

        for strategy in self.strategies:
            strategy.on_compress_end(context)
