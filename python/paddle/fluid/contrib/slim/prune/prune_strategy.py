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

from ..core.strategy import Strategy
from ..graph import VarWrapper, OpWrapper, GraphWrapper
from ....framework import Program, program_guard, Parameter
from ....log_helper import get_logger
from .... import layers
import prettytable as pt
import numpy as np
from scipy.optimize import leastsq
import copy
import re
import os
import pickle
import logging
import sys

__all__ = ['SensitivePruneStrategy', 'UniformPruneStrategy', 'PruneStrategy']

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class PruneStrategy(Strategy):
    """
    The base class of all pruning strategies.
    """

    def __init__(self,
                 pruner=None,
                 start_epoch=0,
                 end_epoch=0,
                 target_ratio=0.5,
                 metric_name=None,
                 pruned_params='conv.*_weights'):
        """
        Args:
            pruner(slim.Pruner): The pruner used to prune the parameters.
            start_epoch(int): The 'on_epoch_begin' function will be called in start_epoch. default: 0
            end_epoch(int): The 'on_epoch_end' function will be called in end_epoch. default: 0
            target_ratio(float): The flops ratio to be pruned from current model.
            metric_name(str): The metric used to evaluate the model.
                         It should be one of keys in out_nodes of graph wrapper.
            pruned_params(str): The pattern str to match the parameter names to be pruned.
        """
        super(PruneStrategy, self).__init__(start_epoch, end_epoch)
        self.pruner = pruner
        self.target_ratio = target_ratio
        self.metric_name = metric_name
        self.pruned_params = pruned_params
        self.pruned_list = []

    def _eval_graph(self, context, sampled_rate=None, cached_id=0):
        """
        Evaluate the current mode in context.
        Args:
            context(slim.core.Context): The context storing all information used to evaluate the current model.
            sampled_rate(float): The sampled rate used to sample partial data for evaluation. None means using all data in eval_reader. default: None.
            cached_id(int): The id of dataset sampled. Evaluations with same cached_id use the same sampled dataset. default: 0.
        """
        results, names = context.run_eval_graph(sampled_rate, cached_id)
        metric = np.mean(results[list(names).index(self.metric_name)])
        return metric

    def _prune_filters_by_ratio(self,
                                scope,
                                params,
                                ratio,
                                place,
                                lazy=False,
                                only_graph=False,
                                param_shape_backup=None,
                                param_backup=None):
        """
        Pruning filters by given ratio.
        Args:
            scope(fluid.core.Scope): The scope used to pruning filters.
            params(list<VarWrapper>): A list of filter parameters.
            ratio(float): The ratio to be pruned.
            place(fluid.Place): The device place of filter parameters.
            lazy(bool): True means setting the pruned elements to zero.
                        False means cutting down the pruned elements.
            only_graph(bool): True means only modifying the graph.
                              False means modifying graph and variables in  scope.
        """
        if params[0].name() in self.pruned_list[0]:
            return
        param_t = scope.find_var(params[0].name()).get_tensor()
        pruned_idx = self.pruner.cal_pruned_idx(
            params[0].name(), np.array(param_t), ratio, axis=0)
        for param in params:
            assert isinstance(param, VarWrapper)
            param_t = scope.find_var(param.name()).get_tensor()
            if param_backup is not None and (param.name() not in param_backup):
                param_backup[param.name()] = copy.deepcopy(np.array(param_t))
            pruned_param = self.pruner.prune_tensor(
                np.array(param_t), pruned_idx, pruned_axis=0, lazy=lazy)
            if not only_graph:
                param_t.set(pruned_param, place)
            ori_shape = param.shape()
            if param_shape_backup is not None and (
                    param.name() not in param_shape_backup):
                param_shape_backup[param.name()] = copy.deepcopy(param.shape())
            new_shape = list(param.shape())
            new_shape[0] = pruned_param.shape[0]
            param.set_shape(new_shape)
            _logger.debug(
                '|----------------------------------------+----+------------------------------+------------------------------|'
            )
            _logger.debug('|{:^40}|{:^4}|{:^30}|{:^30}|'.format(
                str(param.name()),
                str(ratio), str(ori_shape), str(param.shape())))
            self.pruned_list[0].append(param.name())
        return pruned_idx

    def _prune_parameter_by_idx(self,
                                scope,
                                params,
                                pruned_idx,
                                pruned_axis,
                                place,
                                lazy=False,
                                only_graph=False,
                                param_shape_backup=None,
                                param_backup=None):
        """
        Pruning parameters in given axis.
        Args:
            scope(fluid.core.Scope): The scope storing paramaters to be pruned.
            params(VarWrapper): The parameter to be pruned.
            pruned_idx(list): The index of elements to be pruned.
            pruned_axis(int): The pruning axis.
            place(fluid.Place): The device place of filter parameters.
            lazy(bool): True means setting the pruned elements to zero.
                        False means cutting down the pruned elements.
            only_graph(bool): True means only modifying the graph.
                              False means modifying graph and variables in  scope.
        """
        if params[0].name() in self.pruned_list[pruned_axis]:
            return
        for param in params:
            assert isinstance(param, VarWrapper)
            param_t = scope.find_var(param.name()).get_tensor()
            if param_backup is not None and (param.name() not in param_backup):
                param_backup[param.name()] = copy.deepcopy(np.array(param_t))
            pruned_param = self.pruner.prune_tensor(
                np.array(param_t), pruned_idx, pruned_axis, lazy=lazy)
            if not only_graph:
                param_t.set(pruned_param, place)
            ori_shape = param.shape()

            if param_shape_backup is not None and (
                    param.name() not in param_shape_backup):
                param_shape_backup[param.name()] = copy.deepcopy(param.shape())
            new_shape = list(param.shape())
            new_shape[pruned_axis] = pruned_param.shape[pruned_axis]
            param.set_shape(new_shape)
            _logger.debug(
                '|----------------------------------------+----+------------------------------+------------------------------|'
            )
            _logger.debug('|{:^40}|{:^4}|{:^30}|{:^30}|'.format(
                str(param.name()),
                str(pruned_axis), str(ori_shape), str(param.shape())))
            self.pruned_list[pruned_axis].append(param.name())

    def _forward_search_related_op(self, graph, param):
        """
        Forward search operators that will be affected by pruning of param.
        Args:
            graph(GraphWrapper): The graph to be searched.
            param(VarWrapper): The current pruned parameter.
        Returns:
            list<OpWrapper>: A list of operators.
        """
        assert isinstance(param, VarWrapper)
        visited = {}
        for op in graph.ops():
            visited[op.idx()] = False
        stack = []
        for op in graph.ops():
            if (not op.is_bwd_op()) and (param in op.all_inputs()):
                stack.append(op)
        visit_path = []
        while len(stack) > 0:
            top_op = stack[len(stack) - 1]
            if visited[top_op.idx()] == False:
                visit_path.append(top_op)
                visited[top_op.idx()] = True
            next_ops = None
            if top_op.type() == "conv2d" and param not in top_op.all_inputs():
                next_ops = None
            elif top_op.type() == "mul":
                next_ops = None
            else:
                next_ops = self._get_next_unvisited_op(graph, visited, top_op)
            if next_ops == None:
                stack.pop()
            else:
                stack += next_ops
        return visit_path

    def _get_next_unvisited_op(self, graph, visited, top_op):
        """
        Get next unvisited adjacent operators of given operators.
        Args:
            graph(GraphWrapper): The graph used to search. 
            visited(list): The ids of operators that has been visited.
            top_op: The given operator.
        Returns:
            list<OpWrapper>: A list of operators. 
        """
        assert isinstance(top_op, OpWrapper)
        next_ops = []
        for op in graph.next_ops(top_op):
            if (visited[op.idx()] == False) and (not op.is_bwd_op()):
                next_ops.append(op)
        return next_ops if len(next_ops) > 0 else None

    def _get_accumulator(self, graph, param):
        """
        Get accumulators of given parameter. The accumulator was created by optimizer.
        Args:
            graph(GraphWrapper): The graph used to search.
            param(VarWrapper): The given parameter.
        Returns:
            list<VarWrapper>: A list of accumulators which are variables.
        """
        assert isinstance(param, VarWrapper)
        params = []
        for op in param.outputs():
            if op.is_opt_op():
                for out_var in op.all_outputs():
                    if graph.is_persistable(out_var) and out_var.name(
                    ) != param.name():
                        params.append(out_var)
        return params

    def _forward_pruning_ralated_params(self,
                                        graph,
                                        scope,
                                        param,
                                        place,
                                        ratio=None,
                                        pruned_idxs=None,
                                        lazy=False,
                                        only_graph=False,
                                        param_backup=None,
                                        param_shape_backup=None):
        """
        Pruning all the parameters affected by the pruning of given parameter.
        Args:
            graph(GraphWrapper): The graph to be searched.
            scope(fluid.core.Scope): The scope storing paramaters to be pruned.
            param(VarWrapper): The given parameter.
            place(fluid.Place): The device place of filter parameters.
            ratio(float): The target ratio to be pruned.
            pruned_idx(list): The index of elements to be pruned.
            lazy(bool): True means setting the pruned elements to zero.
                        False means cutting down the pruned elements.
            only_graph(bool): True means only modifying the graph.
                              False means modifying graph and variables in  scope.
        """
        assert isinstance(
            graph,
            GraphWrapper), "graph must be instance of slim.core.GraphWrapper"
        assert isinstance(
            param, VarWrapper), "param must be instance of slim.core.VarWrapper"

        if param.name() in self.pruned_list[0]:
            return
        related_ops = self._forward_search_related_op(graph, param)

        if ratio is None:
            assert pruned_idxs is not None
            self._prune_parameter_by_idx(
                scope, [param] + self._get_accumulator(graph, param),
                pruned_idxs,
                pruned_axis=0,
                place=place,
                lazy=lazy,
                only_graph=only_graph,
                param_backup=param_backup,
                param_shape_backup=param_shape_backup)

        else:
            pruned_idxs = self._prune_filters_by_ratio(
                scope, [param] + self._get_accumulator(graph, param),
                ratio,
                place,
                lazy=lazy,
                only_graph=only_graph,
                param_backup=param_backup,
                param_shape_backup=param_shape_backup)
        corrected_idxs = pruned_idxs[:]

        for idx, op in enumerate(related_ops):
            if op.type() == "conv2d" and (param not in op.all_inputs()):
                for in_var in op.all_inputs():
                    if graph.is_parameter(in_var):
                        conv_param = in_var
                        self._prune_parameter_by_idx(
                            scope, [conv_param] + self._get_accumulator(
                                graph, conv_param),
                            corrected_idxs,
                            pruned_axis=1,
                            place=place,
                            lazy=lazy,
                            only_graph=only_graph,
                            param_backup=param_backup,
                            param_shape_backup=param_shape_backup)
            if op.type() == "depthwise_conv2d":
                for in_var in op.all_inputs():
                    if graph.is_parameter(in_var):
                        conv_param = in_var
                        self._prune_parameter_by_idx(
                            scope, [conv_param] + self._get_accumulator(
                                graph, conv_param),
                            corrected_idxs,
                            pruned_axis=0,
                            place=place,
                            lazy=lazy,
                            only_graph=only_graph,
                            param_backup=param_backup,
                            param_shape_backup=param_shape_backup)
            elif op.type() == "elementwise_add":
                # pruning bias
                for in_var in op.all_inputs():
                    if graph.is_parameter(in_var):
                        bias_param = in_var
                        self._prune_parameter_by_idx(
                            scope, [bias_param] + self._get_accumulator(
                                graph, bias_param),
                            pruned_idxs,
                            pruned_axis=0,
                            place=place,
                            lazy=lazy,
                            only_graph=only_graph,
                            param_backup=param_backup,
                            param_shape_backup=param_shape_backup)
            elif op.type() == "mul":  # pruning fc layer
                fc_input = None
                fc_param = None
                for in_var in op.all_inputs():
                    if graph.is_parameter(in_var):
                        fc_param = in_var
                    else:
                        fc_input = in_var

                idx = []
                feature_map_size = fc_input.shape()[2] * fc_input.shape()[3]
                range_idx = np.array(range(feature_map_size))
                for i in corrected_idxs:
                    idx += list(range_idx + i * feature_map_size)
                corrected_idxs = idx
                self._prune_parameter_by_idx(
                    scope, [fc_param] + self._get_accumulator(graph, fc_param),
                    corrected_idxs,
                    pruned_axis=0,
                    place=place,
                    lazy=lazy,
                    only_graph=only_graph,
                    param_backup=param_backup,
                    param_shape_backup=param_shape_backup)

            elif op.type() == "concat":
                concat_inputs = op.all_inputs()
                last_op = related_ops[idx - 1]
                for out_var in last_op.all_outputs():
                    if out_var in concat_inputs:
                        concat_idx = concat_inputs.index(out_var)
                offset = 0
                for ci in range(concat_idx):
                    offset += concat_inputs[ci].shape()[1]
                corrected_idxs = [x + offset for x in pruned_idxs]
            elif op.type() == "batch_norm":
                bn_inputs = op.all_inputs()
                mean = bn_inputs[2]
                variance = bn_inputs[3]
                alpha = bn_inputs[0]
                beta = bn_inputs[1]
                self._prune_parameter_by_idx(
                    scope, [mean] + self._get_accumulator(graph, mean),
                    corrected_idxs,
                    pruned_axis=0,
                    place=place,
                    lazy=lazy,
                    only_graph=only_graph,
                    param_backup=param_backup,
                    param_shape_backup=param_shape_backup)
                self._prune_parameter_by_idx(
                    scope, [variance] + self._get_accumulator(graph, variance),
                    corrected_idxs,
                    pruned_axis=0,
                    place=place,
                    lazy=lazy,
                    only_graph=only_graph,
                    param_backup=param_backup,
                    param_shape_backup=param_shape_backup)
                self._prune_parameter_by_idx(
                    scope, [alpha] + self._get_accumulator(graph, alpha),
                    corrected_idxs,
                    pruned_axis=0,
                    place=place,
                    lazy=lazy,
                    only_graph=only_graph,
                    param_backup=param_backup,
                    param_shape_backup=param_shape_backup)
                self._prune_parameter_by_idx(
                    scope, [beta] + self._get_accumulator(graph, beta),
                    corrected_idxs,
                    pruned_axis=0,
                    place=place,
                    lazy=lazy,
                    only_graph=only_graph,
                    param_backup=param_backup,
                    param_shape_backup=param_shape_backup)

    def _prune_parameters(self,
                          graph,
                          scope,
                          params,
                          ratios,
                          place,
                          lazy=False,
                          only_graph=False,
                          param_backup=None,
                          param_shape_backup=None):
        """
        Pruning the given parameters.
        Args:
            graph(GraphWrapper): The graph to be searched.
            scope(fluid.core.Scope): The scope storing paramaters to be pruned.
            params(list<str>): A list of parameter names to be pruned.
            ratios(list<float>): A list of ratios to be used to pruning parameters.
            place(fluid.Place): The device place of filter parameters.
            pruned_idx(list): The index of elements to be pruned.
            lazy(bool): True means setting the pruned elements to zero.
                        False means cutting down the pruned elements.
            only_graph(bool): True means only modifying the graph.
                              False means modifying graph and variables in  scope.

        """
        _logger.debug('\n################################')
        _logger.debug('#       pruning parameters       #')
        _logger.debug('################################\n')
        _logger.debug(
            '|----------------------------------------+----+------------------------------+------------------------------|'
        )
        _logger.debug('|{:^40}|{:^4}|{:^30}|{:^30}|'.format('parameter', 'axis',
                                                            'from', 'to'))
        assert len(params) == len(ratios)
        self.pruned_list = [[], []]
        for param, ratio in zip(params, ratios):
            assert isinstance(param, str) or isinstance(param, unicode)
            param = graph.var(param)
            self._forward_pruning_ralated_params(
                graph,
                scope,
                param,
                place,
                ratio=ratio,
                lazy=lazy,
                only_graph=only_graph,
                param_backup=param_backup,
                param_shape_backup=param_shape_backup)
            ops = param.outputs()
            for op in ops:
                if op.type() == 'conv2d':
                    brother_ops = self._search_brother_ops(graph, op)
                    for broher in brother_ops:
                        for p in graph.get_param_by_op(broher):
                            self._forward_pruning_ralated_params(
                                graph,
                                scope,
                                p,
                                place,
                                ratio=ratio,
                                lazy=lazy,
                                only_graph=only_graph,
                                param_backup=param_backup,
                                param_shape_backup=param_shape_backup)
        _logger.debug(
            '|----------------------------------------+----+------------------------------+------------------------------|'
        )

    def _search_brother_ops(self, graph, op_node):
        """
        Search brother operators that was affected by pruning of given operator.
        Args:
            graph(GraphWrapper): The graph to be searched.
            op_node(OpWrapper): The start node for searching.
        Returns: 
            list<VarWrapper>: A list of operators.
        """
        visited = [op_node.idx()]
        stack = []
        brothers = []
        for op in graph.next_ops(op_node):
            if (op.type() != 'conv2d') and (op.type() != 'fc') and (
                    not op._is_bwd_op()):
                stack.append(op)
                visited.append(op.idx())
        while len(stack) > 0:
            top_op = stack.pop()
            for parent in graph.pre_ops(top_op):
                if parent.idx() not in visited and (not parent._is_bwd_op()):
                    if ((parent.type == 'conv2d') or (parent.type == 'fc')):
                        brothers.append(parent)
                    else:
                        stack.append(parent)
                    visited.append(parent.idx())

            for child in graph.next_ops(top_op):
                if (child.type != 'conv2d') and (child.type != 'fc') and (
                        child.idx() not in visited) and (
                            not child._is_bwd_op()):
                    stack.append(child)
                    visited.append(child.idx())
        return brothers

    def _prune_graph(self, graph, target_graph):
        """
        Pruning parameters of graph according to target graph.
        Args:
            graph(GraphWrapper): The graph to be pruned.
            target_graph(GraphWrapper): The reference graph.
        Return: None
        """
        count = 1
        _logger.debug(
            '|----+----------------------------------------+------------------------------+------------------------------|'
        )
        _logger.debug('|{:^4}|{:^40}|{:^30}|{:^30}|'.format('id', 'parammeter',
                                                            'from', 'to'))
        for param in target_graph.all_parameters():
            var = graph.var(param.name())
            ori_shape = var.shape()
            var.set_shape(param.shape())
            _logger.debug(
                '|----+----------------------------------------+------------------------------+------------------------------|'
            )
            _logger.debug('|{:^4}|{:^40}|{:^30}|{:^30}|'.format(
                str(count),
                str(param.name()), str(ori_shape), str(param.shape())))
            count += 1
        _logger.debug(
            '|----+----------------------------------------+------------------------------+------------------------------|'
        )


class UniformPruneStrategy(PruneStrategy):
    """
    The uniform pruning strategy. The parameters will be pruned by uniform ratio.
    """

    def __init__(self,
                 pruner=None,
                 start_epoch=0,
                 end_epoch=0,
                 target_ratio=0.5,
                 metric_name=None,
                 pruned_params='conv.*_weights'):
        """
        Args:
            pruner(slim.Pruner): The pruner used to prune the parameters.
            start_epoch(int): The 'on_epoch_begin' function will be called in start_epoch. default: 0
            end_epoch(int): The 'on_epoch_end' function will be called in end_epoch. default: 0
            target_ratio(float): The flops ratio to be pruned from current model.
            metric_name(str): The metric used to evaluate the model.
                         It should be one of keys in out_nodes of graph wrapper.
            pruned_params(str): The pattern str to match the parameter names to be pruned.
        """
        super(UniformPruneStrategy, self).__init__(pruner, start_epoch,
                                                   end_epoch, target_ratio,
                                                   metric_name, pruned_params)

    def _get_best_ratios(self, context):
        """
        Search a group of ratios for pruning target flops.
        """
        _logger.info('_get_best_ratios')
        pruned_params = []
        for param in context.eval_graph.all_parameters():
            if re.match(self.pruned_params, param.name()):
                pruned_params.append(param.name())

        min_ratio = 0.
        max_ratio = 1.

        flops = context.eval_graph.flops()
        model_size = context.eval_graph.numel_params()

        while min_ratio < max_ratio:
            ratio = (max_ratio + min_ratio) / 2
            _logger.debug(
                '-----------Try pruning ratio: {:.2f}-----------'.format(ratio))
            ratios = [ratio] * len(pruned_params)
            param_shape_backup = {}
            self._prune_parameters(
                context.eval_graph,
                context.scope,
                pruned_params,
                ratios,
                context.place,
                only_graph=True,
                param_shape_backup=param_shape_backup)

            pruned_flops = 1 - (float(context.eval_graph.flops()) / flops)
            pruned_size = 1 - (float(context.eval_graph.numel_params()) /
                               model_size)
            _logger.debug('Pruned flops: {:.2f}'.format(pruned_flops))
            _logger.debug('Pruned model size: {:.2f}'.format(pruned_size))
            for param in param_shape_backup.keys():
                context.eval_graph.var(param).set_shape(param_shape_backup[
                    param])

            if abs(pruned_flops - self.target_ratio) < 1e-2:
                break
            if pruned_flops > self.target_ratio:
                max_ratio = ratio
            else:
                min_ratio = ratio
        _logger.info('Get ratios: {}'.format([round(r, 2) for r in ratios]))
        return pruned_params, ratios

    def on_epoch_begin(self, context):
        if context.epoch_id == self.start_epoch:
            params, ratios = self._get_best_ratios(context)

            self._prune_parameters(context.optimize_graph, context.scope,
                                   params, ratios, context.place)

            model_size = context.eval_graph.numel_params()
            flops = context.eval_graph.flops()
            _logger.debug('\n################################')
            _logger.debug('#          pruning eval graph    #')
            _logger.debug('################################\n')
            self._prune_graph(context.eval_graph, context.optimize_graph)
            context.optimize_graph.update_groups_of_conv()
            context.eval_graph.update_groups_of_conv()

            _logger.info(
                '------------------finish pruning--------------------------------'
            )
            _logger.info('Pruned size: {:.2f}'.format(1 - (float(
                context.eval_graph.numel_params()) / model_size)))
            _logger.info('Pruned flops: {:.2f}'.format(1 - (float(
                context.eval_graph.flops()) / flops)))
            #            metric = self._eval_graph(context)
            #            _logger.info('Metric after pruning: {:.2f}'.format(metric))
            _logger.info(
                '------------------UniformPruneStrategy.on_compression_begin finish--------------------------------'
            )


class SensitivePruneStrategy(PruneStrategy):
    """
    Sensitive pruning strategy. Different pruned ratio was applied on each layer.
    """

    def __init__(self,
                 pruner=None,
                 start_epoch=0,
                 end_epoch=0,
                 delta_rate=0.20,
                 target_ratio=0.5,
                 metric_name='top1_acc',
                 pruned_params='conv.*_weights',
                 sensitivities_file='./sensitivities.data',
                 sensitivities={},
                 num_steps=1,
                 eval_rate=None):
        """
        Args:
            pruner(slim.Pruner): The pruner used to prune the parameters.
            start_epoch(int): The 'on_epoch_begin' function will be called in start_epoch. default: 0.
            end_epoch(int): The 'on_epoch_end' function will be called in end_epoch. default: 10.
            delta_rate(float): The delta used to generate ratios when calculating sensitivities. default: 0.2
            target_ratio(float): The flops ratio to be pruned from current model. default: 0.5
            metric_name(str): The metric used to evaluate the model.
                         It should be one of keys in out_nodes of graph wrapper. default: 'top1_acc'
            pruned_params(str): The pattern str to match the parameter names to be pruned. default: 'conv.*_weights'.
            sensitivities_file(str): The sensitivities file. default: './sensitivities.data'
            sensitivities(dict): The user-defined sensitivities. default: {}.
            num_steps(int): The number of pruning steps. default: 1.
            eval_rate(float): The rate of sampled data used to calculate sensitivities.
                              None means using all the data. default: None.
        """
        super(SensitivePruneStrategy, self).__init__(pruner, start_epoch,
                                                     end_epoch, target_ratio,
                                                     metric_name, pruned_params)
        self.delta_rate = delta_rate
        self.pruned_list = []
        self.sensitivities = sensitivities
        self.sensitivities_file = sensitivities_file
        self.num_steps = num_steps
        self.eval_rate = eval_rate
        self.pruning_step = 1 - pow((1 - target_ratio), 1.0 / self.num_steps)

    def _save_sensitivities(self, sensitivities, sensitivities_file):
        """
        Save sensitivities into file.
        """
        with open(sensitivities_file, 'wb') as f:
            pickle.dump(sensitivities, f)

    def _load_sensitivities(self, sensitivities_file):
        """
        Load sensitivities from file.
        """
        sensitivities = {}
        if sensitivities_file and os.path.exists(sensitivities_file):
            with open(sensitivities_file, 'rb') as f:
                if sys.version_info < (3, 0):
                    sensitivities = pickle.load(f)
                else:
                    sensitivities = pickle.load(f, encoding='bytes')

        for param in sensitivities:
            sensitivities[param]['pruned_percent'] = [
                round(p, 2) for p in sensitivities[param]['pruned_percent']
            ]
        self._format_sensitivities(sensitivities)
        return sensitivities

    def _format_sensitivities(self, sensitivities):
        """
        Print formated sensitivities in debug log level.
        """
        tb = pt.PrettyTable()
        tb.field_names = ["parameter", "size"] + [
            str(round(i, 2))
            for i in np.arange(self.delta_rate, 1, self.delta_rate)
        ]
        for param in sensitivities:
            if len(sensitivities[param]['loss']) == (len(tb.field_names) - 2):
                tb.add_row([param, sensitivities[param]['size']] + [
                    round(loss, 2) for loss in sensitivities[param]['loss']
                ])
        _logger.debug('\n################################')
        _logger.debug('#      sensitivities table     #')
        _logger.debug('################################\n')
        _logger.debug(tb)

    def _compute_sensitivities(self, context):
        """
        Computing the sensitivities of all parameters.
        """
        _logger.info("calling _compute_sensitivities.")
        cached_id = np.random.randint(1000)
        if self.start_epoch == context.epoch_id:
            sensitivities_file = self.sensitivities_file
        else:
            sensitivities_file = self.sensitivities_file + ".epoch" + str(
                context.epoch_id)
        sensitivities = self._load_sensitivities(sensitivities_file)

        for param in context.eval_graph.all_parameters():
            if not re.match(self.pruned_params, param.name()):
                continue
            if param.name() not in sensitivities:
                sensitivities[param.name()] = {
                    'pruned_percent': [],
                    'loss': [],
                    'size': param.shape()[0]
                }

        metric = None

        for param in sensitivities.keys():
            ratio = self.delta_rate
            while ratio < 1:
                ratio = round(ratio, 2)
                if ratio in sensitivities[param]['pruned_percent']:
                    _logger.debug('{}, {} has computed.'.format(param, ratio))
                    ratio += self.delta_rate
                    continue
                if metric is None:
                    metric = self._eval_graph(context, self.eval_rate,
                                              cached_id)

                param_backup = {}
                # prune parameter by ratio
                self._prune_parameters(
                    context.eval_graph,
                    context.scope, [param], [ratio],
                    context.place,
                    lazy=True,
                    param_backup=param_backup)
                self.pruned_list[0]
                # get accuracy after pruning and update self.sensitivities
                pruned_metric = self._eval_graph(context, self.eval_rate,
                                                 cached_id)
                loss = metric - pruned_metric
                _logger.info("pruned param: {}; {}; loss={}".format(
                    param, ratio, loss))
                for brother in self.pruned_list[0]:
                    if re.match(self.pruned_params, brother):
                        if brother not in sensitivities:
                            sensitivities[brother] = {
                                'pruned_percent': [],
                                'loss': []
                            }
                        sensitivities[brother]['pruned_percent'].append(ratio)
                        sensitivities[brother]['loss'].append(loss)

                self._save_sensitivities(sensitivities, sensitivities_file)

                # restore pruned parameters
                for param_name in param_backup.keys():
                    param_t = context.scope.find_var(param_name).get_tensor()
                    param_t.set(self.param_backup[param_name], context.place)

#                pruned_metric = self._eval_graph(context)

                ratio += self.delta_rate
        return sensitivities

    def _get_best_ratios(self, context, sensitivities, target_ratio):
        """
        Search a group of ratios for pruning target flops.
        """
        _logger.info('_get_best_ratios for pruning ratie: {}'.format(
            target_ratio))

        def func(params, x):
            a, b, c, d = params
            return a * x * x * x + b * x * x + c * x + d

        def error(params, x, y):
            return func(params, x) - y

        def slove_coefficient(x, y):
            init_coefficient = [10, 10, 10, 10]
            coefficient, loss = leastsq(error, init_coefficient, args=(x, y))
            return coefficient

        min_loss = 0.
        max_loss = 0.

        # step 1: fit curve by sensitivities
        coefficients = {}
        for param in sensitivities:
            losses = np.array([0] * 5 + sensitivities[param]['loss'])
            precents = np.array([0] * 5 + sensitivities[param][
                'pruned_percent'])
            coefficients[param] = slove_coefficient(precents, losses)
            loss = np.max(losses)
            max_loss = np.max([max_loss, loss])

        # step 2: Find a group of ratios by binary searching.
        flops = context.eval_graph.flops()
        model_size = context.eval_graph.numel_params()
        ratios = []
        while min_loss < max_loss:
            loss = (max_loss + min_loss) / 2
            _logger.info(
                '-----------Try pruned ratios while acc loss={:.4f}-----------'.
                format(loss))
            ratios = []
            # step 2.1: Get ratios according to current loss
            for param in sensitivities:
                coefficient = copy.deepcopy(coefficients[param])
                coefficient[-1] = coefficient[-1] - loss
                roots = np.roots(coefficient)
                for root in roots:
                    min_root = 1
                    if np.isreal(root) and root > 0 and root < 1:
                        selected_root = min(root.real, min_root)
                ratios.append(selected_root)
            _logger.info('Pruned ratios={}'.format(
                [round(ratio, 3) for ratio in ratios]))
            # step 2.2: Pruning by current ratios
            param_shape_backup = {}
            self._prune_parameters(
                context.eval_graph,
                context.scope,
                sensitivities.keys(),
                ratios,
                context.place,
                only_graph=True,
                param_shape_backup=param_shape_backup)

            pruned_flops = 1 - (float(context.eval_graph.flops()) / flops)
            pruned_size = 1 - (float(context.eval_graph.numel_params()) /
                               model_size)
            _logger.info('Pruned flops: {:.4f}'.format(pruned_flops))
            _logger.info('Pruned model size: {:.4f}'.format(pruned_size))
            for param in param_shape_backup.keys():
                context.eval_graph.var(param).set_shape(param_shape_backup[
                    param])

            # step 2.3: Check whether current ratios is enough
            if abs(pruned_flops - target_ratio) < 0.015:
                break
            if pruned_flops > target_ratio:
                max_loss = loss
            else:
                min_loss = loss
        return sensitivities.keys(), ratios

    def _current_pruning_target(self, context):
        '''
        Get the target pruning rate in current epoch.
        '''
        _logger.info('Left number of pruning steps: {}'.format(self.num_steps))
        if self.num_steps <= 0:
            return None
        if (self.start_epoch == context.epoch_id) or context.eval_converged(
                self.metric_name, 0.005):
            self.num_steps -= 1
            return self.pruning_step

    def on_epoch_begin(self, context):
        current_ratio = self._current_pruning_target(context)
        if current_ratio is not None:
            sensitivities = self._compute_sensitivities(context)
            params, ratios = self._get_best_ratios(context, sensitivities,
                                                   current_ratio)
            self._prune_parameters(context.optimize_graph, context.scope,
                                   params, ratios, context.place)

            model_size = context.eval_graph.numel_params()
            flops = context.eval_graph.flops()
            _logger.debug('################################')
            _logger.debug('#          pruning eval graph    #')
            _logger.debug('################################')
            self._prune_graph(context.eval_graph, context.optimize_graph)
            context.optimize_graph.update_groups_of_conv()
            context.eval_graph.update_groups_of_conv()
            context.optimize_graph.compile()  # to update the compiled program
            context.eval_graph.compile(
                for_parallel=False,
                for_test=True)  # to update the compiled program
            _logger.info(
                '------------------finish pruning--------------------------------'
            )
            _logger.info('Pruned size: {:.3f}'.format(1 - (float(
                context.eval_graph.numel_params()) / model_size)))
            _logger.info('Pruned flops: {:.3f}'.format(1 - (float(
                context.eval_graph.flops()) / flops)))
            metric = self._eval_graph(context)
            _logger.info('Metric after pruning: {:.2f}'.format(metric))
            _logger.info(
                '------------------SensitivePruneStrategy.on_epoch_begin finish--------------------------------'
            )
