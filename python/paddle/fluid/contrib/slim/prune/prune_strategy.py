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
from ....framework import Program, program_guard, Parameter
from .... import layers
from ..graph import get_executor
import prettytable as pt
import numpy as np
from scipy.optimize import leastsq
import copy
import re
import os
import pickle
import logging
import sys

__all__ = ['SensitivePruneStrategy', 'UniformPruneStrategy']

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

OPTIMIZER_OPS = [
    'momentum',
    'lars_momentum',
    'adagrad',
    'adam',
    'adamax',
    'decayed_adagrad',
    'adadelta',
    'rmsprop',
]


class PruneStrategy(Strategy):
    def __init__(self,
                 pruner=None,
                 start_epoch=0,
                 end_epoch=10,
                 target_ratio=0.5,
                 metric_name=None,
                 pruned_params='conv.*_weights'):
        super(PruneStrategy, self).__init__(start_epoch, end_epoch)
        self.pruner = pruner
        self.target_ratio = target_ratio
        self.metric_name = metric_name
        self.pruned_params = pruned_params
        self.pruned_list = []
        self.backup = {}
        self.param_shape_backup = {}

    def _eval_graph(self, context, sampled_rate=None, cached_id=0):
        results, names = context.run_eval_graph(sampled_rate, cached_id)
        metric = np.mean(results[names.index(self.metric_name)])
        return metric

    def _prune_parameter_by_ratio(self,
                                  scope,
                                  params,
                                  ratio,
                                  place,
                                  lazy=False,
                                  only_graph=False):
        """
        Only support for pruning in axis 0 by ratio.
        """
        if params[0].name in self.pruned_list[0]:
            return
        param_t = scope.find_var(params[0].name).get_tensor()
        pruned_idx = self.pruner.cal_pruned_idx(
            params[0].name, np.array(param_t), ratio, axis=0)
        for param in params:
            param_t = scope.find_var(param.name).get_tensor()
            if lazy:
                self.backup[param.name] = copy.deepcopy(np.array(param_t))
            pruned_param = self.pruner.prune_tensor(
                np.array(param_t), pruned_idx, pruned_axis=0, lazy=lazy)
            if not only_graph:
                param_t.set(pruned_param, place)
            ori_shape = param.shape
            if param.name not in self.param_shape_backup:
                self.param_shape_backup[param.name] = copy.deepcopy(param.shape)
            new_shape = list(param.shape)
            new_shape[0] = pruned_param.shape[0]
            param.desc.set_shape(new_shape)
            logger.debug(
                '|----------------------------------------+----+------------------------------+------------------------------|'
            )
            logger.debug('|{:^40}|{:^4}|{:^30}|{:^30}|'.format(
                param.name, 0, ori_shape, param.shape))
            self.pruned_list[0].append(param.name)
        return pruned_idx

    def _prune_parameter_by_idx(self,
                                scope,
                                params,
                                pruned_idx,
                                pruned_axis,
                                place,
                                lazy=False,
                                only_graph=False):
        if params[0].name in self.pruned_list[pruned_axis]:
            return
        for param in params:
            param_t = scope.find_var(param.name).get_tensor()
            if lazy:
                self.backup[param.name] = copy.deepcopy(np.array(param_t))
            pruned_param = self.pruner.prune_tensor(
                np.array(param_t), pruned_idx, pruned_axis, lazy=lazy)
            if not only_graph:
                param_t.set(pruned_param, place)
            ori_shape = param.shape
            if param.name not in self.param_shape_backup:
                self.param_shape_backup[param.name] = copy.deepcopy(param.shape)
            new_shape = list(param.shape)
            new_shape[pruned_axis] = pruned_param.shape[pruned_axis]
            param.desc.set_shape(new_shape)
            logger.debug(
                '|----------------------------------------+----+------------------------------+------------------------------|'
            )
            logger.debug('|{:^40}|{:^4}|{:^30}|{:^30}|'.format(
                param.name, pruned_axis, ori_shape, param.shape))
            self.pruned_list[pruned_axis].append(param.name)

    def _inputs(self, op):
        names = []
        for name in [op.input(name) for name in op.input_names]:
            if isinstance(name, list):
                names += name
            else:
                names.append(name)
        return names

    def _outputs(self, op):
        names = []
        for name in [op.output(name) for name in op.output_names]:
            if isinstance(name, list):
                names += name
            else:
                names.append(name)
        return names

    def _forward_search_related_op(self, graph, param):
        visited = {}
        for op in graph.ops():
            visited[op.idx] = False
        stack = []
        for op in graph.ops():
            if (not op.type.endswith('_grad')) and (
                    param.name in self._inputs(op)):
                stack.append(op)
        visit_path = []
        while len(stack) > 0:
            top_op = stack[len(stack) - 1]
            if visited[top_op.idx] == False:
                visit_path.append(top_op)
                visited[top_op.idx] = True
            next_ops = None
            if top_op.type == "conv2d" and param.name not in self._inputs(
                    top_op):
                next_ops = None
            elif top_op.type == "mul":
                next_ops = None
            else:
                next_ops = self._get_next_unvisited_op(graph, visited, top_op)
            if next_ops == None:
                stack.pop()
            else:
                stack += next_ops
        return visit_path

    def _get_next_unvisited_op(self, graph, visited, top_op):
        next_ops = []
        for out_name in self._outputs(top_op):
            for op in graph.ops():
                if (out_name in self._inputs(op)) and (
                        visited[op.idx] == False) and (
                            not op.type.endswith('_grad')):
                    next_ops.append(op)
        return next_ops if len(next_ops) > 0 else None

    def _get_accumulator(self, graph, param):
        params = []
        for op in graph.get_ops_by_param(param):
            if op.type in OPTIMIZER_OPS:
                for out_var in self._outputs(op):
                    out_var = graph.get_var(out_var)
                    if out_var.persistable and out_var.name != param.name:
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
                                        only_graph=False):
        if param.name in self.pruned_list[0]:
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
                only_graph=only_graph)

        else:
            pruned_idxs = self._prune_parameter_by_ratio(
                scope, [param] + self._get_accumulator(graph, param),
                ratio,
                place,
                lazy=lazy,
                only_graph=only_graph)
        corrected_idxs = pruned_idxs[:]

        for idx, op in enumerate(related_ops):

            if op.type == "conv2d" and (param.name not in self._inputs(op)):
                for param_name in self._inputs(op):
                    if isinstance(graph.get_var(param_name), Parameter):
                        conv_param = graph.get_var(param_name)
                        self._prune_parameter_by_idx(
                            scope, [conv_param] + self._get_accumulator(
                                graph, conv_param),
                            corrected_idxs,
                            pruned_axis=1,
                            place=place,
                            lazy=lazy,
                            only_graph=only_graph)
            if op.type == "depthwise_conv2d":
                for param_name in self._inputs(op):
                    if isinstance(graph.get_var(param_name), Parameter):
                        conv_param = graph.get_var(param_name)
                        self._prune_parameter_by_idx(
                            scope, [conv_param] + self._get_accumulator(
                                graph, conv_param),
                            corrected_idxs,
                            pruned_axis=0,
                            place=place,
                            lazy=lazy,
                            only_graph=only_graph)
            elif op.type == "elementwise_add":
                # pruning bias
                for param_name in self._inputs(op):
                    if isinstance(graph.get_var(param_name), Parameter):
                        bias_param = graph.get_var(param_name)
                        self._prune_parameter_by_idx(
                            scope, [bias_param] + self._get_accumulator(
                                graph, bias_param),
                            pruned_idxs,
                            pruned_axis=0,
                            place=place,
                            lazy=lazy,
                            only_graph=only_graph)
            elif op.type == "mul":  # pruning fc layer
                fc_input = None
                fc_param = None
                for i in self._inputs(op):
                    i_var = graph.get_var(i)
                    if isinstance(i_var, Parameter):
                        fc_param = i_var
                    else:
                        fc_input = i_var

                idx = []
                feature_map_size = fc_input.shape[2] * fc_input.shape[3]
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
                    only_graph=only_graph)

            elif op.type == "concat":
                concat_inputs = self._inputs(op)
                last_op = related_ops[idx - 1]
                for out_name in self._outputs(last_op):
                    if out_name in concat_inputs:
                        concat_idx = concat_inputs.index(out_name)
                offset = 0
                for ci in range(concat_idx):
                    offset += graph.get_var(concat_inputs[ci]).shape[1]
                corrected_idxs = [x + offset for x in pruned_idxs]
            elif op.type == "batch_norm":
                bn_inputs = self._inputs(op)
                mean = graph.get_var(bn_inputs[2])
                variance = graph.get_var(bn_inputs[3])
                alpha = graph.get_var(bn_inputs[0])
                beta = graph.get_var(bn_inputs[1])
                self._prune_parameter_by_idx(
                    scope, [mean] + self._get_accumulator(graph, mean),
                    corrected_idxs,
                    pruned_axis=0,
                    place=place,
                    lazy=lazy,
                    only_graph=only_graph)
                self._prune_parameter_by_idx(
                    scope, [variance] + self._get_accumulator(graph, variance),
                    corrected_idxs,
                    pruned_axis=0,
                    place=place,
                    lazy=lazy,
                    only_graph=only_graph)
                self._prune_parameter_by_idx(
                    scope, [alpha] + self._get_accumulator(graph, alpha),
                    corrected_idxs,
                    pruned_axis=0,
                    place=place,
                    lazy=lazy,
                    only_graph=only_graph)
                self._prune_parameter_by_idx(
                    scope, [beta] + self._get_accumulator(graph, beta),
                    corrected_idxs,
                    pruned_axis=0,
                    place=place,
                    lazy=lazy,
                    only_graph=only_graph)

    def _prune_parameters(self,
                          graph,
                          scope,
                          params,
                          ratios,
                          place,
                          lazy=False,
                          only_graph=False):
        """
        Pruning parameters.
        """
        logger.debug('\n################################')
        logger.debug('#       pruning parameters       #')
        logger.debug('################################\n')
        logger.debug(
            '|----------------------------------------+----+------------------------------+------------------------------|'
        )
        logger.debug('|{:^40}|{:^4}|{:^30}|{:^30}|'.format('parameter', 'axis',
                                                           'from', 'to'))

        self.pruned_list = [[], []]
        for param, ratio in zip(params, ratios):
            param = graph.get_var(param)

            self._forward_pruning_ralated_params(
                graph,
                scope,
                param,
                place,
                ratio=ratio,
                lazy=lazy,
                only_graph=only_graph)
            ops = graph.get_ops_by_param(param)
            for op in ops:
                if op.type == 'conv2d':
                    brother_ops = self._seach_brother_ops(graph, op)
                    for broher in brother_ops:
                        for p in graph.get_param_by_op(broher):
                            self._forward_pruning_ralated_params(
                                graph,
                                scope,
                                p,
                                place,
                                ratio=ratio,
                                lazy=lazy,
                                only_graph=only_graph)
        logger.debug(
            '|----------------------------------------+----+------------------------------+------------------------------|'
        )

    def _conv_op_name(self, op):
        return op.input('Filter')

    def _is_bwd_op(self, op):
        if (op.type in OPTIMIZER_OPS) or op.type.endswith('_grad'):
            return True

    def _seach_brother_ops(self, graph, op_node):
        """
        Args:
            graph: The graph to be searched.
            op_node: The start node for searching.
        """
        visited = [op_node.idx]
        stack = []
        brothers = []
        for op in graph.next_ops(op_node):
            if (op.type != 'conv2d') and (op.type != 'fc') and (
                    not self._is_bwd_op(op)):
                stack.append(op)
                visited.append(op.idx)
        while len(stack) > 0:
            top_op = stack.pop()
            for parent in graph.pre_ops(top_op):
                if parent.idx not in visited and (not self._is_bwd_op(parent)):
                    if ((parent.type == 'conv2d') or (parent.type == 'fc')):
                        brothers.append(parent)
                    else:
                        stack.append(parent)
                    visited.append(parent.idx)

            for child in graph.next_ops(top_op):
                if (child.type != 'conv2d') and (child.type != 'fc') and (
                        child.idx not in visited) and (
                            not self._is_bwd_op(child)):
                    stack.append(child)
                    visited.append(child.idx)
        return brothers

    def _is_optimizer_op(self, op):
        if op.type in OPTIMIZER_OPS:
            return True

    def _prune_graph(self, graph, target_graph):
        count = 1
        logger.debug(
            '|----+----------------------------------------+------------------------------+------------------------------|'
        )
        logger.debug('|{:^4}|{:^40}|{:^30}|{:^30}|'.format('id', 'parammeter',
                                                           'from', 'to'))
        for param in target_graph.all_parameters():
            var = graph.get_var(param.name)
            ori_shape = var.shape
            var.desc.set_shape(param.shape)
            logger.debug(
                '|----+----------------------------------------+------------------------------+------------------------------|'
            )
            logger.debug('|{:^4}|{:^40}|{:^30}|{:^30}|'.format(
                count, param.name, ori_shape, param.shape))
            count += 1
        logger.debug(
            '|----+----------------------------------------+------------------------------+------------------------------|'
        )

    def _update_depthwise_conv(self, graph):
        for op in graph.all_ops():
            if op.type == 'depthwise_conv2d':
                op.desc._set_attr('groups',
                                  graph.get_var(op.input('Filter')[0]).shape[0])
                logger.debug('input shape: {:30} filter shape: {:30}'.format(
                    graph.get_var(op.input('Input')[0]).shape,
                    graph.get_var(op.input('Filter')[0]).shape))

            if op.type == 'depthwise_conv2d' or op.type == 'conv2d':
                shape = np.array(
                    graph.scope.find_var(op.input('Filter')[0]).get_tensor(
                    )).shape
                logger.debug(
                    'op: {:15} input: {:30} filter: {:30} tensor:{:30}'.format(
                        op.type,
                        graph.get_var(op.input('Input')[0]).shape,
                        graph.get_var(op.input('Filter')[0]).shape, shape))


class UniformPruneStrategy(PruneStrategy):
    def __init__(self,
                 pruner=None,
                 start_epoch=0,
                 end_epoch=10,
                 target_ratio=0.5,
                 metric_name=None,
                 pruned_params='conv.*_weights'):
        super(UniformPruneStrategy, self).__init__(pruner, start_epoch,
                                                   end_epoch, target_ratio,
                                                   metric_name, pruned_params)

    def _get_best_ratios(self, context):
        logger.info('_get_best_ratios')
        pruned_params = []
        for param in context.eval_graph.all_parameters():
            if re.match(self.pruned_params, param.name):
                pruned_params.append(param.name)

        min_ratio = 0.
        max_ratio = 1.

        flops = context.eval_graph.flops()
        model_size = context.eval_graph.numel_params()

        while min_ratio < max_ratio:
            ratio = (max_ratio + min_ratio) / 2
            logger.debug(
                '-----------Try pruning ratio: {:.2f}-----------'.format(ratio))
            ratios = [ratio] * len(pruned_params)
            self._prune_parameters(
                context.eval_graph,
                context.eval_graph.scope,
                pruned_params,
                ratios,
                context.place,
                only_graph=True)

            pruned_flops = 1 - (float(context.eval_graph.flops()) / flops)
            pruned_size = 1 - (float(context.eval_graph.numel_params()) /
                               model_size)
            logger.debug('Pruned flops: {:.2f}'.format(pruned_flops))
            logger.debug('Pruned model size: {:.2f}'.format(pruned_size))
            for param in self.param_shape_backup.keys():
                context.eval_graph.get_var(param).desc.set_shape(
                    self.param_shape_backup[param])
            self.param_shape_backup = {}

            if abs(pruned_flops - self.target_ratio) < 1e-2:
                break
            if pruned_flops > self.target_ratio:
                max_ratio = ratio
            else:
                min_ratio = ratio
        logger.info('Get ratios: {}'.format([round(r, 2) for r in ratios]))
        return pruned_params, ratios

    def on_epoch_begin(self, context):
        if context.epoch_id == self.start_epoch:
            params, ratios = self._get_best_ratios(context)

            self._prune_parameters(context.optimize_graph,
                                   context.optimize_graph.scope, params, ratios,
                                   context.place)

            model_size = context.eval_graph.numel_params()
            flops = context.eval_graph.flops()
            logger.debug('\n################################')
            logger.debug('#          pruning eval graph    #')
            logger.debug('################################\n')
            self._prune_graph(context.eval_graph, context.optimize_graph)
            self._update_depthwise_conv(context.optimize_graph)
            self._update_depthwise_conv(context.eval_graph)

            logger.info(
                '------------------finish pruning--------------------------------'
            )
            logger.info('Pruned size: {:.2f}'.format(1 - (float(
                context.eval_graph.numel_params()) / model_size)))
            logger.info('Pruned flops: {:.2f}'.format(1 - (float(
                context.eval_graph.flops()) / flops)))
            metric = self._eval_graph(context)
            logger.info('Metric after pruning: {:.2f}'.format(metric))
            logger.info(
                '------------------UniformPruneStrategy.on_compression_begin finish--------------------------------'
            )


class SensitivePruneStrategy(PruneStrategy):
    def __init__(self,
                 pruner=None,
                 start_epoch=0,
                 end_epoch=10,
                 delta_rate=0.20,
                 target_ratio=0.5,
                 metric_name=None,
                 pruned_params='conv.*_weights',
                 sensitivities_file='./vgg11_sensitivities.data',
                 sensitivities={},
                 num_steps=1,
                 eval_rate=None):
        super(SensitivePruneStrategy, self).__init__(pruner, start_epoch,
                                                     end_epoch, target_ratio,
                                                     metric_name, pruned_params)
        self.delta_rate = delta_rate
        self.pruned_list = []
        self.sensitivities = sensitivities
        self.sensitivities_file = sensitivities_file
        self.backup = {}
        self.param_shape_backup = {}
        self.num_steps = num_steps
        self.eval_rate = eval_rate
        self.pruning_step = 1 - pow((1 - target_ratio), 1.0 / self.num_steps)

    def _save_sensitivities(self, sensitivities, sensitivities_file):
        with open(sensitivities_file, 'wb') as f:
            pickle.dump(sensitivities, f)

    def _load_sensitivities(self, sensitivities_file):
        sensitivities = {}
        if sensitivities_file and os.path.exists(sensitivities_file):
            with open(sensitivities_file, 'rb') as f:
                sensitivities = pickle.load(f)

        for param in sensitivities:
            sensitivities[param]['pruned_percent'] = [
                round(p, 2) for p in sensitivities[param]['pruned_percent']
            ]
        self._format_sensitivities(sensitivities)
        return sensitivities

    def _format_sensitivities(self, sensitivities):
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
        print('\n################################')
        print('#      sensitivities table     #')
        print('################################\n')
        print(tb)

    def _compute_sensitivities(self, context):
        """
        Computing the sensitivities of all parameters.
        """
        logger.info("calling _compute_sensitivities.")
        self.param_shape_backup = {}
        self.backup = {}
        cached_id = np.random.randint(1000)
        if self.start_epoch == context.epoch_id:
            sensitivities_file = self.sensitivities_file
        else:
            sensitivities_file = self.sensitivities_file + ".epoch" + str(
                context.epoch_id)
        sensitivities = self._load_sensitivities(sensitivities_file)

        for param in context.eval_graph.all_parameters():
            if not re.match(self.pruned_params, param.name):
                continue
            if param.name not in sensitivities:
                sensitivities[param.name] = {
                    'pruned_percent': [],
                    'loss': [],
                    'size': param.shape[0]
                }

        metric = None

        for param in sensitivities.keys():
            ratio = self.delta_rate
            while ratio < 1:
                ratio = round(ratio, 2)
                if ratio in sensitivities[param]['pruned_percent']:
                    logger.debug('{}, {} has computed.'.format(param, ratio))
                    ratio += self.delta_rate
                    continue
                if metric is None:
                    metric = self._eval_graph(context, self.eval_rate,
                                              cached_id)
                # prune parameter by ratio
                self._prune_parameters(
                    context.eval_graph,
                    context.eval_graph.scope, [param], [ratio],
                    context.place,
                    lazy=True)
                self.pruned_list[0]
                # get accuracy after pruning and update self.sensitivities
                pruned_metric = self._eval_graph(context, self.eval_rate,
                                                 cached_id)
                loss = metric - pruned_metric
                logger.info("pruned param: {}; {}; loss={}".format(param, ratio,
                                                                   loss))
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
                for param_name in self.backup.keys():
                    param_t = context.eval_graph.scope.find_var(
                        param_name).get_tensor()
                    param_t.set(self.backup[param_name], context.place)

#                pruned_metric = self._eval_graph(context)
                self.backup = {}

                ratio += self.delta_rate
        return sensitivities

    def _get_best_ratios(self, context, sensitivities, target_ratio):
        logger.info('_get_best_ratios')
        self.param_shape_backup = {}
        self.backup = {}

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
        coefficients = {}
        for param in sensitivities:
            losses = np.array([0] * 5 + sensitivities[param]['loss'])
            precents = np.array([0] * 5 + sensitivities[param][
                'pruned_percent'])
            coefficients[param] = slove_coefficient(precents, losses)
            loss = np.max(losses)
            max_loss = np.max([max_loss, loss])

        flops = context.eval_graph.flops()
        model_size = context.eval_graph.numel_params()
        ratios = []
        while min_loss < max_loss:
            loss = (max_loss + min_loss) / 2
            logger.info(
                '-----------Try pruned ratios while acc loss={:.2f}-----------'.
                format(loss))
            ratios = []
            for param in sensitivities:
                coefficient = copy.deepcopy(coefficients[param])
                coefficient[-1] = coefficient[-1] - loss
                roots = np.roots(coefficient)
                for root in roots:
                    min_root = 1
                    if np.isreal(root) and root > 0 and root < 1:
                        selected_root = min(root.real, min_root)
                ratios.append(selected_root)
            logger.info('Pruned ratios={}'.format(
                [round(ratio, 2) for ratio in ratios]))
            self._prune_parameters(
                context.eval_graph,
                context.eval_graph.scope,
                sensitivities.keys(),
                ratios,
                context.place,
                only_graph=True)

            pruned_flops = 1 - (float(context.eval_graph.flops()) / flops)
            pruned_size = 1 - (float(context.eval_graph.numel_params()) /
                               model_size)
            logger.info('Pruned flops: {:.2f}'.format(pruned_flops))
            logger.info('Pruned model size: {:.2f}'.format(pruned_size))
            for param in self.param_shape_backup.keys():
                context.eval_graph.get_var(param).desc.set_shape(
                    self.param_shape_backup[param])
            self.param_shape_backup = {}

            if abs(pruned_flops - target_ratio) < 1e-2:
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
        logger.info('Left number of pruning steps: {}'.format(self.num_steps))
        if self.num_steps <= 0:
            return None
        if (self.start_epoch == context.epoch_id) or context.eval_converged(
                self.metric_name, 0.001):
            self.num_steps -= 1
            return self.pruning_step

    def on_epoch_begin(self, context):
        current_ratio = self._current_pruning_target(context)
        if current_ratio is not None:
            sensitivities = self._compute_sensitivities(context)
            params, ratios = self._get_best_ratios(context, sensitivities,
                                                   current_ratio)
            self._prune_parameters(context.optimize_graph,
                                   context.optimize_graph.scope, params, ratios,
                                   context.place)

            self.param_shape_backup = {}
            self.backup = {}

            model_size = context.eval_graph.numel_params()
            flops = context.eval_graph.flops()
            logger.debug('\n################################')
            logger.debug('#          pruning eval graph    #')
            logger.debug('################################\n')
            self._prune_graph(context.eval_graph, context.optimize_graph)
            self._update_depthwise_conv(context.optimize_graph)
            self._update_depthwise_conv(context.eval_graph)

            logger.info(
                '------------------finish pruning--------------------------------'
            )
            logger.info('Pruned size: {:.2f}'.format(1 - (float(
                context.eval_graph.numel_params()) / model_size)))
            logger.info('Pruned flops: {:.2f}'.format(1 - (float(
                context.eval_graph.flops()) / flops)))
            metric = self._eval_graph(context)
            logger.info('Metric after pruning: {:.2f}'.format(metric))
            logger.info(
                '------------------SensitivePruneStrategy.on_epoch_begin finish--------------------------------'
            )
