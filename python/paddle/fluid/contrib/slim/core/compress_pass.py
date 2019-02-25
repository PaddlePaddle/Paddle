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
from .... import io
from .... import profiler
from ....data_feeder import DataFeeder
from ..graph import *
from config import ConfigFactory
import numpy as np
from collections import Iterable
import time
import os
import logging
import sys
import pickle

__all__ = ['Context', 'CompressPass']

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def cached_reader(reader, sampled_rate, cache_path, cached_id):
    np.random.seed(cached_id)
    cache_path = cache_path + "/" + str(cached_id)
    logger.info('read data from: {}'.format(cache_path))

    def s_reader():
        if os.path.isdir(cache_path):
            for file_name in open(cache_path + "/list"):
                yield np.load(cache_path + '/' + file_name.strip())
        else:
            os.makedirs(cache_path)
            list_file = open(cache_path + "/list", 'w')
            batch = 0
            dtype = None
            for data in reader():
                if batch == 0 or (np.random.uniform() < sampled_rate):
                    np.save(cache_path + '/batch' + str(batch), data)
                    list_file.write('batch' + str(batch) + '.npy\n')
                    batch += 1
                    yield data

    return s_reader


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
                 teacher_graphs=None,
                 optimizer=None):
        # The total number of epoches to be trained.
        self.epoch = 0
        # Current epoch
        #        self.epoch_id = -1
        self.epoch_id = -1
        # Current batch
        self.batch_id = 0

        self.k_v = {}

        self.place = place
        self.train_graph = train_graph
        self.train_reader = train_reader
        self.eval_graph = eval_graph
        self.eval_reader = eval_reader
        self.executor = None
        self.teacher_graphs = teacher_graphs
        self.optimizer = optimizer
        self.cache_path = './eval_cache'
        self.eval_results = {}

    def to_file(self, file_name):
        data = {}
        data['epoch_id'] = self.epoch_id
        data['eval_results'] = self.eval_results
        with open(file_name, 'wb') as context_file:
            pickle.dump(data, context_file)

    def from_file(self, file_name):
        with open(file_name) as context_file:
            data = pickle.load(context_file)
            self.epoch_id = data['epoch_id']
            self.eval_results = data['eval_results']

    def eval_converged(self, metric_name, delta=0.001):
        if (metric_name not in self.eval_results
            ) or len(self.eval_results[metric_name]) < 2:
            return False
        results = self.eval_results[metric_name][-2:]
        logger.info('Latest evaluations: {}'.format(results))
        return abs(results[1] - results[0]) / results[0] < delta

    def run_eval_graph(self, sampled_rate=None, cached_id=0):
        assert self.eval_graph is not None
        assert self.eval_reader is not None
        eval_graph = self.eval_graph.clone(for_test=True)
        executor = get_executor(self.eval_graph, self.place, parallel=True)
        results = []
        batch_id = 0
        s_time = time.time()
        reader = self.eval_reader
        if sampled_rate:
            reader = cached_reader(reader, sampled_rate, self.cache_path,
                                   cached_id)
        for data in reader():
            result = executor.run(eval_graph, data=data)
            result = [np.mean(r) for r in result]
            results.append(result)
            if batch_id % 20 == 0:
                e_time = time.time()
                logger.info("time: {:.2f}s; batch[{}] eval: {}={}".format(
                    e_time - s_time, batch_id,
                    self.eval_graph.out_nodes.keys(
                    ), [round(r, 2) for r in result]))
                s_time = time.time()
            batch_id += 1
        result = np.mean(np.array(results), axis=0)
        logger.info("final eval result: {}={}".format(
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
                 optimizer=None,
                 checkpoint_path='./checkpoints'):
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
        self.checkpoint_path = checkpoint_path
        self.eval_epoch = 1

        self.optimizer = optimizer

    def add_strategy(self, strategy):
        """
        Add a strategy to current compress pass.
        Args:
            strategy: The strategy to be added into current compress pass.
        """
        self.strategies.append(strategy)
        self.epoch = max(strategy.end_epoch, self.epoch)

    def config(self, config_file):
        factory = ConfigFactory(config_file)
        self.epoch = factory.compress_pass['epoch']
        for strategy in factory.compress_pass['strategies']:
            self.add_strategy(strategy)

    def _load_checkpoint(self, context):
        logger.info('_load_checkpoint')
        strategies = self.strategies
        if self.checkpoint_path:
            checkpoints = [
                dir for dir in os.listdir(self.checkpoint_path)
                if os.path.isdir(os.path.join(self.checkpoint_path, dir))
            ]
            logger.info('self.checkpoint_path: {}'.format(self.checkpoint_path))
            logger.info('checkpoints: {}'.format(checkpoints))
            if len(checkpoints) > 0:
                latest = max(checkpoints)
                latest_ck_path = os.path.join(self.checkpoint_path, str(latest))

                model_path = os.path.join(latest_ck_path, 'model')
                context_path = os.path.join(latest_ck_path, 'context')
                strategy_path = os.path.join(latest_ck_path, 'strategies')
                context.from_file(context_path)
                with open(strategy_path, 'rb') as strategy_file:
                    strategies = pickle.load(strategy_file)

                exe = get_executor(
                    context.train_graph, context.place, parallel=False)
                load_persistables(context.train_graph, model_path, exe)
                update_param_shape(context.eval_graph)
                update_depthwise_conv(context.eval_graph)
                logger.info("Loaded checkpoint from: {}".format(
                    self.checkpoint_path))
        return context, strategies

    def _save_checkpoint(self, context):
        if context.epoch_id % 1 == 0 and self.checkpoint_path:
            checkpoint_path = os.path.join(self.checkpoint_path,
                                           str(context.epoch_id))
            model_path = os.path.join(checkpoint_path, 'model')
            context_path = os.path.join(checkpoint_path, 'context')
            strategy_path = os.path.join(checkpoint_path, 'strategies')
            if not os.path.isdir(model_path):
                os.makedirs(model_path)

            exe = get_executor(
                context.train_graph, context.place, parallel=False)
            save_persistables(context.train_graph, model_path, exe)
            context.to_file(context_path)
            with open(strategy_path, 'wb') as strategy_file:
                pickle.dump(self.strategies, strategy_file)
            logger.info('Saved checkpoint to: {}'.format(checkpoint_path))

    def _train_one_epoch(self, context):
        if context.train_graph is None:
            logger.info("train_graph is None; Please config train_graph_pass.")
            return

        executor = get_executor(self.train_graph, self.place, parallel=True)
        #        with profiler.profiler('GPU', 'total'):
        for data in context.train_reader():
            for strategy in self.strategies:
                strategy.on_batch_begin(context)
            feed = None
            results = executor.run(context.train_graph, data=data)
            results = [float(np.mean(result)) for result in results]
            if context.batch_id % 20 == 0:
                logger.info("epoch:{}; batch_id:{}; {} = {}".format(
                    context.epoch_id, context.batch_id,
                    context.train_graph.out_nodes.keys(), results))
            for strategy in self.strategies:
                strategy.on_batch_end(context)
            context.batch_id += 1

#        context.epoch_id += 1
        context.batch_id = 0

    def _eval(self, context):
        results, names = context.run_eval_graph()
        for name, result in zip(names, results):
            if name not in context.eval_results:
                context.eval_results[name] = []
            context.eval_results[name].append(result)

    def run(self):

        context = Context(
            place=self.place,
            train_graph=self.train_graph,
            train_reader=self.train_reader,
            eval_graph=self.eval_graph,
            eval_reader=self.eval_reader,
            teacher_graphs=self.teacher_graphs,
            optimizer=self.optimizer)

        context, self.strategies = self._load_checkpoint(context)

        self.executor = get_executor(
            self.train_graph, self.place, parallel=True)
        context.put('executor', self.executor)

        if self.teacher_graphs:
            context.put('teachers', self.teacher_graphs)

        for strategy in self.strategies:
            strategy.on_compression_begin(context)
        start = context.epoch_id + 1
        for epoch in range(start, self.epoch):
            context.epoch_id = epoch
            for strategy in self.strategies:
                strategy.on_epoch_begin(context)

            self._train_one_epoch(context)

            for strategy in self.strategies:
                strategy.on_epoch_end(context)

            if self.eval_epoch and epoch % self.eval_epoch == 0:
                self._eval(context)
            self._save_checkpoint(context)

        for strategy in self.strategies:
            strategy.on_compression_end(context)

        return context.eval_graph
