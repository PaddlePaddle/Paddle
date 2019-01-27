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
from ....data_feeder import DataFeeder
from ..graph import get_executor, ImitationGraph
from config import ConfigFactory
import numpy as np
from collections import Iterable
import time
import os

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

    def __init__(self,
                 place,
                 train_graph=None,
                 train_reader=None,
                 eval_graph=None,
                 eval_reader=None,
                 optimizer=None):
        # The total number of epoches to be trained.
        self.epoch = 0
        # Current epoch
        self.epoch_id = 0
        # Current batch
        self.batch_id = 0

        self.k_v = {}

        self.place = place
        self.train_graph = train_graph
        self.train_reader = train_reader
        self.eval_graph = eval_graph
        self.eval_reader = eval_reader
        self.executor = None
        self.optimizer = optimizer

    def run_eval_graph(self):
        assert self.eval_graph is not None
        assert self.eval_reader is not None
        self.executor = get_executor(self.eval_graph, self.place, parallel=True)
        results = []
        batch_id = 0
        s_time = time.time()
        for data in self.eval_reader():
            result = self.executor.run(self.eval_graph, data=data)
            results.append(result)
            if batch_id % 20 == 0:
                e_time = time.time()
                print("time: {}s; batch[{}] eval: {}={}".format(
                    e_time - s_time, batch_id,
                    self.eval_graph.out_nodes.keys(), list(result)))
                s_time = time.time()
                break
            batch_id += 1
        result = np.mean(np.array(results), axis=0)
        print("final eval result: {}={}".format(
            self.eval_graph.out_nodes.keys(), result))
        if not isinstance(result, Iterable):
            result = [result]
        return result, self.eval_graph.out_nodes.keys()

    def put(self, key, value):
        self.k_v[key] = value

    def get(self, key):
        return self.k_v.get(key)


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
                 place,
                 scope,
                 train_program,
                 train_reader=None,
                 train_feed_list=None,
                 train_fetch_list=None,
                 eval_program=None,
                 eval_reader=None,
                 eval_feed_list=None,
                 eval_fetch_list=None,
                 teacher_programs=[],
                 optimizer=None):
        self.strategies = []
        self.epoch = 0
        self.place = CPUPlace() if place is None else place
        self.train_graph = ImitationGraph(
            train_program,
            scope=scope,
            in_nodes=train_feed_list,
            out_nodes=train_fetch_list)
        self.eval_graph = ImitationGraph(
            eval_program,
            scope=scope,
            in_nodes=eval_feed_list,
            out_nodes=eval_fetch_list)
        self.train_reader = train_reader
        self.eval_reader = eval_reader
        self.teacher_graphs = []
        for teacher in teacher_programs:
            self.teacher_graphs.append(ImitationGraph(teacher, scope=scope))

        self.checkpoint = None
        self.model_save_dir = None
        self.eval_epoch = 1

        self.optimizer = optimizer

    def add_strategy(self, strategy):
        """
        Add a strategy to current compress pass.
        Args:
            strategy: The strategy to be added into current compress pass.
        """
        print "add strategy: %s" % (type(strategy))
        self.strategies.append(strategy)
        self.epoch = max(strategy.end_epoch, self.epoch)

    def config(self, config_file):
        factory = ConfigFactory(config_file)
        self.epoch = factory.compress_pass['epoch']
        for strategy in factory.compress_pass['strategies']:
            self.add_strategy(strategy)

    def _load_checkpoint(self, context):
        if self.checkpoint:
            exe = get_executor(context.train_graph, parallel=False)
            fluid.io.load_persistables(
                exe.exe,
                self.checkpoint,
                main_program=context.train_graph.program)
            print("Loaded checkpoint from: {}".format(self.checkpoint))

    def _save_checkpoint(self, context):
        if context.epoch_id % 5 == 0 and self.model_save_dir:
            model_path = os.path.join(self.model_save_dir,
                                      str(context.epoch_id))
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            exe = get_executor(context.train_graph, parallel=False)
            fluid.io.save_persistables(
                exe.exe, model_path, main_program=context.train_graph.program)
            print('Saved checkpoint to: {}'.format(model_path))

    def _train_one_epoch(self, context):
        if context.train_graph is None:
            print("train_graph is None; Please config train_graph_pass.")
            return
        for data in context.train_reader():
            for strategy in self.strategies:
                strategy.on_batch_begin(context)
            feed = None
            results = self.executor.run(context.train_graph, data=data)
            results = [float(result) for result in results]
            if context.batch_id % 20 == 0:
                print("epoch:{}; batch_id:{}; {} = {}".format(
                    context.epoch, context.batch_id,
                    context.train_graph.out_nodes.keys(), results))
            for strategy in self.strategies:
                strategy.on_batch_end(context)
            context.batch_id += 1
        self._save_checkpoint(context)

    def _eval(self, context):
        results, names = context.run_eval_graph()
        print("epoch:{}; batch_id:{}; eval results: {}={}".format(
            context.epoch, context.batch_id, names, results))

    def run(self):

        context = Context(
            place=self.place,
            train_graph=self.train_graph,
            train_reader=self.train_reader,
            eval_graph=self.eval_graph,
            eval_reader=self.eval_reader,
            optimizer=self.optimizer)

        self._load_checkpoint(context)

        self.executor = get_executor(self.train_graph, self.place)
        context.put('executor', self.executor)

        if self.teacher_graphs:
            context.put('teachers', self.teacher_graphs)

        for strategy in self.strategies:
            strategy.on_compression_begin(context)

        for epoch in range(self.epoch):

            for strategy in self.strategies:
                strategy.on_epoch_begin(context)

            self._train_one_epoch(context)

            for strategy in self.strategies:
                strategy.on_epoch_end(context)
            context.epoch_id += 1

            if self.eval_epoch and epoch % self.eval_epoch == 0:
                self._eval(context)

        for strategy in self.strategies:
            strategy.on_compression_end(context)

        return context.eval_graph
