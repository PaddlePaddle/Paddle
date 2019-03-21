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
from .... import compiler
from .... import io
from .... import profiler
from .... import scope_guard
from ....data_feeder import DataFeeder
from ..graph import *
from .config import ConfigFactory
import numpy as np
from collections import Iterable
import time
import os
import logging
import sys
import pickle
import functools

__all__ = ['Context', 'Compressor']

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def cached_reader(reader, sampled_rate, cache_path, cached_id):
    """
    Sample partial data from reader and cache them into local file system.
    Args:
        reader: Iterative data source.
        sampled_rate(float): The sampled rate used to sample partial data for evaluation. None means using all data in eval_reader. default: None.
        cache_path(str): The path to cache the sampled data.
        cached_id(int): The id of dataset sampled. Evaluations with same cached_id use the same sampled dataset. default: 0.
    """
    np.random.seed(cached_id)
    cache_path = os.path.join(cache_path, str(cached_id))
    logger.debug('read data from: {}'.format(cache_path))

    def s_reader():
        if os.path.isdir(cache_path):
            for file_name in open(os.path.join(cache_path, "list")):
                yield np.load(os.path.join(cache_path, file_name.strip()))
        else:
            os.makedirs(cache_path)
            list_file = open(os.path.join(cache_path, "list"), 'w')
            batch = 0
            dtype = None
            for data in reader():
                if batch == 0 or (np.random.uniform() < sampled_rate):
                    np.save(
                        os.path.join(cache_path, 'batch' + str(batch)), data)
                    list_file.write('batch' + str(batch) + '.npy\n')
                    batch += 1
                    yield data

    return s_reader


class Context(object):
    """
    The context in the process of compression.
    """

    def __init__(self,
                 place,
                 scope,
                 train_graph=None,
                 train_reader=None,
                 eval_graph=None,
                 eval_reader=None,
                 teacher_graphs=None,
                 train_optimizer=None,
                 distiller_optimizer=None):
        """
        Args:
            place: The device place where the compression job running.
            scope: The scope used in compression job.
            train_graph: The graph with loss as output node.
            eval_graph: The graph used for evaluation.
            eval_reader: The data reader used for evaluation.
            teacher_graphs: The teacher graphs used in distillation strategies.
            train_optimizer: The optimizer used to append backward ops and
                             optimization ops into train_graph.
            distiller_optimizer: The optimizer used by distillation strategies.
        """
        # The total number of epoches to be trained.
        self.epoch = 0
        # Current epoch
        self.epoch_id = 0
        # Current batch
        self.batch_id = 0

        self.k_v = {}

        self.place = place
        self.scope = scope
        self.train_graph = train_graph
        self.train_reader = train_reader
        self.eval_graph = eval_graph
        self.eval_reader = eval_reader
        self.executor = None
        self.teacher_graphs = teacher_graphs
        self.train_optimizer = train_optimizer
        self.distiller_optimizer = distiller_optimizer
        self.optimize_graph = None
        self.cache_path = './eval_cache'
        self.eval_results = {}

    def to_file(self, file_name):
        """
        Save the context into file.
        """
        data = {}
        data['epoch_id'] = self.epoch_id
        data['eval_results'] = self.eval_results
        with open(file_name, 'wb') as context_file:
            pickle.dump(data, context_file)

    def from_file(self, file_name):
        """
        Load the context from file.
        """
        with open(file_name) as context_file:
            if sys.version_info < (3, 0):
                data = pickle.load(context_file)
            else:
                data = pickle.load(context_file, encoding='bytes')
            self.epoch_id = data['epoch_id']
            self.eval_results = data['eval_results']

    def eval_converged(self, metric_name, delta=0.001):
        """
        Check whether the training has been converged.
        Args:
            metric_name(str): The metric used to check convergence.
            delta(float): '(metric[k] - metric[k-1] / metric[k-1]) < delta'
                          means that the training has been converged.
        Returns:
            bool: True means the training has been converged.
        """
        # TODO(wanghaoshuang@baidu.com): enhence this method.
        if (metric_name not in self.eval_results
            ) or len(self.eval_results[metric_name]) < 2:
            return False
        results = self.eval_results[metric_name][-2:]
        logger.info('Latest evaluations: {}'.format(results))
        return abs(results[1] - results[0]) / results[0] < delta

    def run_eval_graph(self, sampled_rate=None, cached_id=0):
        """
        Evaluate the current mode in context.
        Args:
            sampled_rate(float): The sampled rate used to sample partial data
            for evaluation. None means using all data in eval_reader. default: None.
            cached_id(int): The id of dataset sampled. Evaluations with same
                            cached_id use the same sampled dataset. default: 0.
        """
        logger.info('Running evaluation')
        assert self.eval_graph is not None
        assert self.eval_reader is not None
        eval_graph = self.eval_graph.clone(for_test=True)

        executor = SlimGraphExecutor(self.place)
        results = []
        batch_id = 0
        s_time = time.time()
        reader = self.eval_reader
        if sampled_rate:
            reader = cached_reader(reader, sampled_rate, self.cache_path,
                                   cached_id)
        for data in reader():
            result = executor.run(eval_graph, self.scope, data=data)
            result = [np.mean(r) for r in result]
            results.append(result)
            if batch_id % 20 == 0:
                logger.info("batch-{}; {}={}".format(
                    batch_id, eval_graph.out_nodes.keys(), result))
            batch_id += 1
        result = np.mean(np.array(results), axis=0)
        logger.info("Final eval result: {}={}".format(eval_graph.out_nodes.keys(
        ), result))
        if not isinstance(result, Iterable):
            result = [result]
        logger.info('Finish evaluation')
        return result, eval_graph.out_nodes.keys()

    def put(self, key, value):
        self.k_v[key] = value

    def get(self, key):
        return self.k_v.get(key)


class Compressor(object):
    """
    The pass used to compress model.
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
                 checkpoint_path='./checkpoints',
                 train_optimizer=None,
                 distiller_optimizer=None):
        """
        Args:
            place(fluid.Place): The device place where the compression job running.
            scope(fluid.core.Scope): The scope used to run graph.
            train_program(Program): The main program to be compressed. It must have loss op.
            train_reader: The data reader used for training.
            train_feed_list(dict): A dict to indicate the input variable of the training program.
                                   The key is user-defined and human-readable name.
                                   The value is the name of Variable.
            train_fetch_list(dict): A dict to indicate the output variable of the training program.
                                   The key is user-defined and human-readable name.
                                   The value is the name of Variable.
            eval_program(Program): The program used for evaluation.
            eval_reader: The data reader used for evaluation.
            eval_feed_list(dict): A dict to indicate the input variable of the evaluation program.
                                   The key is user-defined and human-readable name.
                                   The value is the name of Variable.
            eval_fetch_list(dict): A dict to indicate the output variable of the evaluation program.
                                   The key is user-defined and human-readable name.
                                   The value is the name of Variable.
            teacher_programs: The teacher graphs used in distillation strategies.
            train_optimizer: The optimizer used to append backward ops and
                             optimization ops into train_graph.
            distiller_optimizer: The optimizer used by distillation strategies. In distillation strategy,
                                 this optimizer is used to minimize the combined loss of student-net and
                                 teacher-net while train_optimizer is used to minimize loss of
                                 student-net in fine-tune stage. 

        """
        assert isinstance(
            train_feed_list, list
        ), "train_feed_list should be a list of tuple, such as [('image', image.name), ('label', gt.name)]"
        assert isinstance(
            eval_feed_list, list
        ), "eval_feed_list should be a list of tuple, such as [('image', image.name), ('label', gt.name)]"
        self.strategies = []
        self.epoch = 0
        self.place = CPUPlace() if place is None else place
        self.scope = scope
        self.train_graph = GraphWrapper(
            train_program, in_nodes=train_feed_list, out_nodes=train_fetch_list)
        self.eval_graph = GraphWrapper(
            eval_program, in_nodes=eval_feed_list, out_nodes=eval_fetch_list)
        self.train_reader = train_reader
        self.eval_reader = eval_reader
        self.teacher_graphs = []
        for teacher in teacher_programs:
            self.teacher_graphs.append(ImitationGraph(teacher, scope=scope))

        self.checkpoint = None
        self.checkpoint_path = checkpoint_path
        self.eval_epoch = 1

        self.train_optimizer = train_optimizer
        self.distiller_optimizer = distiller_optimizer
        self.init_model = None

    def _add_strategy(self, strategy):
        """
        Add a strategy to current compress pass.
        Args:
            strategy: The strategy to be added into current compress pass.
        """
        self.strategies.append(strategy)
        self.epoch = max(strategy.end_epoch, self.epoch)

    def config(self, config_file):
        """
        Configure the compress pass from file with yaml format.
        Args:
            config_file(str): The config file in local file system.
        """
        factory = ConfigFactory(config_file)
        self.epoch = factory.compressor['epoch']
        for strategy in factory.compressor['strategies']:
            self._add_strategy(strategy)
        if 'checkpoint_path' in factory.compressor:
            self.checkpoint_path = factory.compressor['checkpoint_path']

        if 'init_model' in factory.compressor:
            self.init_model = factory.compressor['init_model']

    def _init_model(self, context):
        """
        Load model that has been compressed. 
        """
        if self.init_model and os.path.exists(self.init_model):
            exe = SlimGraphExecutor(context.place)
            with scope_guard(context.scope):
                context.train_graph.load_persistables(self.init_model, exe)
            flops = context.eval_graph.flops()
            conv_flops = context.eval_graph.flops(only_conv=True)
            context.eval_graph.update_param_shape(context.scope)
            context.eval_graph.update_groups_of_conv()
            logger.info("conv flops: -{}".format(1 - float(
                context.eval_graph.flops(only_conv=True)) / conv_flops))
            logger.info("total flops: -{}".format(1 - float(
                context.eval_graph.flops()) / flops))
            context.train_graph.update_param_shape(context.scope)
            context.train_graph.update_groups_of_conv()
            context.train_graph.infer_shape()
            logger.info("Init model from: {}".format(self.init_model))

    def _load_checkpoint(self, context):
        """
        Load checkpoints from file.
        """
        logger.debug('_load_checkpoint')
        strategies = self.strategies
        if self.checkpoint_path:
            if not os.path.exists(self.checkpoint_path):
                logger.warning("Checkpints path doesn't exist: [{}]".format(
                    self.checkpoint_path))
                return context, strategies
            checkpoints = [
                dir for dir in os.listdir(self.checkpoint_path)
                if os.path.isdir(os.path.join(self.checkpoint_path, dir))
            ]
            logger.debug('self.checkpoint_path: {}'.format(
                self.checkpoint_path))
            logger.info('checkpoints: {}'.format(checkpoints))
            if len(checkpoints) > 0:
                latest = max([int(ck) for ck in checkpoints])
                latest_ck_path = os.path.join(self.checkpoint_path, str(latest))

                model_path = os.path.join(latest_ck_path, 'model')
                context_path = os.path.join(latest_ck_path, 'context')
                strategy_path = os.path.join(latest_ck_path, 'strategies')
                if os.path.exists(context_path):
                    context.from_file(context_path)
                    context.epoch_id += 1
                if os.path.exists(strategy_path):
                    with open(strategy_path, 'rb') as strategy_file:
                        if sys.version_info < (3, 0):
                            strategies = pickle.load(strategy_file)
                        else:
                            strategies = pickle.load(
                                strategy_file, encoding='bytes')

                if os.path.exists(model_path):
                    exe = SlimGraphExecutor(context.place)
                    with scope_guard(context.scope):
                        context.optimize_graph.load_persistables(model_path,
                                                                 exe)
                    context.optimize_graph.update_param_shape(context.scope)
                    context.optimize_graph.update_groups_of_conv()
                    context.eval_graph.update_param_shape(context.scope)
                    context.eval_graph.update_groups_of_conv()
                    logger.info("Loaded params from: {}".format(model_path))
        return context, strategies

    def _save_checkpoint(self, context):
        """
        Save checkpoints to file.
        """
        if context.epoch_id % 1 == 0 and self.checkpoint_path:
            checkpoint_path = os.path.join(self.checkpoint_path,
                                           str(context.epoch_id))
            model_path = os.path.join(checkpoint_path, 'model')
            context_path = os.path.join(checkpoint_path, 'context')
            strategy_path = os.path.join(checkpoint_path, 'strategies')
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            exe = SlimGraphExecutor(context.place)
            with scope_guard(context.scope):
                context.optimize_graph.save_persistables(model_path, exe)
            context.to_file(context_path)
            with open(strategy_path, 'wb') as strategy_file:
                pickle.dump(self.strategies, strategy_file)
            logger.info('Saved checkpoint to: {}'.format(checkpoint_path))

    def _train_one_epoch(self, context):
        """
        Train one epoch.
        """

        executor = SlimGraphExecutor(self.place)

        if context.optimize_graph.compiled_graph is None:
            context.optimize_graph.compiled_graph = compiler.CompiledProgram(
                context.optimize_graph.program).with_data_parallel(
                    loss_name=context.optimize_graph.out_nodes['loss'])

        for data in context.train_reader():
            for strategy in self.strategies:
                strategy.on_batch_begin(context)
            results = executor.run(context.optimize_graph,
                                   context.scope,
                                   data=data)
            results = [float(np.mean(result)) for result in results]
            if context.batch_id % 20 == 0:
                logger.info("epoch:{}; batch_id:{}; {} = {}".format(
                    context.epoch_id, context.batch_id,
                    context.optimize_graph.out_nodes.keys(
                    ), [round(r, 3) for r in results]))
            for strategy in self.strategies:
                strategy.on_batch_end(context)
            context.batch_id += 1
        context.batch_id = 0

    def _eval(self, context):
        """
        Runing evaluation.
        """
        results, names = context.run_eval_graph()
        for name, result in zip(names, results):
            if name not in context.eval_results:
                context.eval_results[name] = []
            context.eval_results[name].append(result)

    def run(self):
        """
        Execute compressiong pass.
        """
        context = Context(
            place=self.place,
            scope=self.scope,
            train_graph=self.train_graph,
            train_reader=self.train_reader,
            eval_graph=self.eval_graph,
            eval_reader=self.eval_reader,
            teacher_graphs=self.teacher_graphs,
            train_optimizer=self.train_optimizer,
            distiller_optimizer=self.distiller_optimizer)
        self.context = context
        if self.teacher_graphs:
            context.put('teachers', self.teacher_graphs)
        self._init_model(context)
        if not context.optimize_graph:
            if context.train_optimizer:
                context.train_optimizer._name = 'train_opt'
                context.optimize_graph = context.train_graph.get_optimize_graph(
                    context.train_optimizer, context.place, context.scope)
            else:
                context.optimize_graph = context.train_graph

        context, self.strategies = self._load_checkpoint(context)

        for strategy in self.strategies:
            strategy.on_compression_begin(context)
        start = context.epoch_id
        self._eval(context)
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
