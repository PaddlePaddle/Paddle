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
import copy
import re
import os
import pickle

__all__ = ['SensitivePruneStrategy']

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


class SensitivePruneStrategy(Strategy):
    def __init__(self,
                 pruner=None,
                 start_epoch=0,
                 end_epoch=10,
                 delta_rate=0.20,
                 target_ratio=0.5,
                 metric_name=None,
                 pruned_params='conv.*_weights',
                 sensitivities_file='./vgg11_sensitivities.data',
                 sensitivities={}):
        super(SensitivePruneStrategy, self).__init__(start_epoch, end_epoch)
        self.pruner = pruner
        self.delta_rate = delta_rate
        self.target_ratio = target_ratio
        self.metric_name = metric_name
        self.pruned_params = pruned_params
        self.pruned_list = []
        self.sensitivities = sensitivities
        self.sensitivities_file = sensitivities_file
        self.backup = {}

    def _eval_graph(self, context):
        results, names = context.run_eval_graph()
        metric = np.mean(results[names.index(self.metric_name)])
        return metric

    def _save_sensitivities(self):
        with open(self.sensitivities_file, 'wb') as f:
            pickle.dump(self.sensitivities, f)

    def _load_sensitivities(self):
        if self.sensitivities_file and os.path.exists(self.sensitivities_file):
            with open(self.sensitivities_file, 'rb') as f:
                self.sensitivities = pickle.load(f)
        else:
            self.sensitivities = {}
        for param in self.sensitivities:
            self.sensitivities[param]['pruned_percent'] = [
                round(p, 2) for p in self.sensitivities[param]['pruned_percent']
            ]
        self._format_sensitivities()

    def _format_sensitivities(self):
        tb = pt.PrettyTable()
        tb.field_names = ["parameter", "size"] + [
            str(round(i, 2))
            for i in np.arange(self.delta_rate, 1, self.delta_rate)
        ]
        for param in self.sensitivities:
            if len(self.sensitivities[param]['loss']) == (
                    len(tb.field_names) - 2):
                tb.add_row([param, self.sensitivities[param]['size']] + [
                    round(loss, 2) for loss in self.sensitivities[param]['loss']
                ])
        print('\n################################')
        print('#      sensitivities table     #')
        print('################################\n')
        print(tb)

    def _compute_sensitivities(self, context):
        """
        Computing the sensitivities of all parameters.
        """
        print("calling _compute_sensitivities.")

        self._load_sensitivities()
        metric = self._eval_graph(context)

        for param in context.eval_graph.all_parameters():
            if not re.match(self.pruned_params, param.name):
                continue
            if param.name not in self.sensitivities:
                self.sensitivities[param.name] = {
                    'pruned_percent': [],
                    'loss': [],
                    'size': param.shape[0]
                }

        for param in self.sensitivities.keys():
            ratio = self.delta_rate
            while ratio < 1:
                ratio = round(ratio, 2)
                if ratio in self.sensitivities[param]['pruned_percent']:
                    print('{}, {} has computed.'.format(param, ratio))
                    ratio += self.delta_rate
                    continue
                # prune parameter by ratio
                self._prune_parameters(
                    context.eval_graph,
                    context.eval_graph.scope, [param], [ratio],
                    context.place,
                    lazy=True)
                self.pruned_list[0]
                # get accuracy after pruning and update self.sensitivities
                pruned_metric = self._eval_graph(context)
                loss = metric - pruned_metric
                print "pruned param: {}; {}; loss={}".format(param, ratio, loss)
                for brother in self.pruned_list[0]:
                    if re.match(self.pruned_params, brother):
                        if brother not in self.sensitivities:
                            self.sensitivities[brother] = {
                                'pruned_percent': [],
                                'loss': []
                            }
                        self.sensitivities[brother]['pruned_percent'].append(
                            ratio)
                        self.sensitivities[brother]['loss'].append(loss)

                self._save_sensitivities()

                # restore pruned parameters
                for param_name in self.backup.keys():
                    param_t = context.eval_graph.scope.find_var(
                        param_name).get_tensor()
                    param_t.set(self.backup[param_name], context.place)

                ratio += self.delta_rate

    def _get_best_ratios(self):
        losses = []
        sizes = []
        for param in self.sensitivities:
            losses.append(self.sensitivities[param]['loss'][self.sensitivities[
                param]['pruned_percent'].index(self.target_ratio)])
            sizes.append(self.sensitivities[param]['size'])
        losses = np.array(losses) / np.max(np.array(losses))
        sensitivities = (1.0 / losses) * np.array(sizes)
        ratios = sensitivities / np.sum(sensitivities)
        size_ratios = np.array(sizes, dtype='float32') / np.sum(np.array(sizes))
        ratios = (ratios / size_ratios) * self.target_ratio
        tb = pt.PrettyTable()
        tb.field_names = ['parameter', 'pruned ratio']
        for param, ratio in zip(self.sensitivities.keys(), ratios):
            tb.add_row([param, ratio])
        print('\n################################')
        print('#      best pruning ratios     #')
        print('################################\n')
        print(tb)
        return self.sensitivities.keys(), ratios

    def _prune_parameter_by_ratio(self, scope, params, ratio, place,
                                  lazy=False):
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
            pruned_param = self.pruner.prune_tensor(
                np.array(param_t), pruned_idx, pruned_axis=0, lazy=lazy)
            param_t.set(pruned_param, place)
            ori_shape = param.shape
            param.desc.set_shape(pruned_param.shape)
            print(
                '|----------------------------------------+----+------------------------------+------------------------------|'
            )
            print('|{:^40}|{:^4}|{:^30}|{:^30}|'.format(param.name, 0,
                                                        ori_shape, param.shape))
            if lazy:
                self.backup[param.name] = copy.deepcopy(np.array(param_t))
            self.pruned_list[0].append(param.name)
        return pruned_idx

    def _prune_parameter_by_idx(self,
                                scope,
                                params,
                                pruned_idx,
                                pruned_axis,
                                place,
                                lazy=False):
        if params[0].name in self.pruned_list[pruned_axis]:
            return
        for param in params:
            param_t = scope.find_var(param.name).get_tensor()
            pruned_param = self.pruner.prune_tensor(
                np.array(param_t), pruned_idx, pruned_axis, lazy=lazy)
            param_t.set(pruned_param, place)
            ori_shape = param.shape
            param.desc.set_shape(pruned_param.shape)
            print(
                '|----------------------------------------+----+------------------------------+------------------------------|'
            )
            print('|{:^40}|{:^4}|{:^30}|{:^30}|'.format(param.name, pruned_axis,
                                                        ori_shape, param.shape))
            if lazy:
                self.backup[param.name] = copy.deepcopy(np.array(param_t))
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
                                        lazy=False):
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
                lazy=lazy)

        else:
            pruned_idxs = self._prune_parameter_by_ratio(
                scope, [param] + self._get_accumulator(graph, param),
                ratio,
                place,
                lazy=lazy)
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
                            lazy=lazy)
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
                            lazy=lazy)
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
                    lazy=lazy)

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
                    lazy=lazy)
                self._prune_parameter_by_idx(
                    scope, [variance] + self._get_accumulator(graph, variance),
                    corrected_idxs,
                    pruned_axis=0,
                    place=place,
                    lazy=lazy)
                self._prune_parameter_by_idx(
                    scope, [alpha] + self._get_accumulator(graph, alpha),
                    corrected_idxs,
                    pruned_axis=0,
                    place=place,
                    lazy=lazy)
                self._prune_parameter_by_idx(
                    scope, [beta] + self._get_accumulator(graph, beta),
                    corrected_idxs,
                    pruned_axis=0,
                    place=place,
                    lazy=lazy)

    def _prune_parameters(self, graph, scope, params, ratios, place,
                          lazy=False):
        """
        Pruning parameters.
        """
        print('\n################################')
        print('#       pruning parameters       #')
        print('################################\n')
        print(
            '|----------------------------------------+----+------------------------------+------------------------------|'
        )
        print('|{:^40}|{:^4}|{:^30}|{:^30}|'.format('parameter', 'axis', 'from',
                                                    'to'))

        self.pruned_list = [[], []]
        for param, ratio in zip(params, ratios):
            param = graph.get_var(param)

            self._forward_pruning_ralated_params(
                graph, scope, param, place, ratio=ratio, lazy=lazy)
            ops = graph.get_ops_by_param(param)
            for op in ops:
                if op.type == 'conv2d':
                    brother_ops = self._seach_brother_ops(graph, op)
                    for broher in brother_ops:
                        for p in graph.get_param_by_op(broher):
                            self._forward_pruning_ralated_params(
                                graph, scope, p, place, ratio=ratio, lazy=lazy)
        print(
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
        print(
            '|----+----------------------------------------+------------------------------+------------------------------|'
        )
        print('|{:^4}|{:^40}|{:^30}|{:^30}|'.format('id', 'parammeter', 'from',
                                                    'to'))
        for param in target_graph.all_parameters():
            var = graph.get_var(param.name)
            ori_shape = var.shape
            var.desc.set_shape(param.shape)
            print(
                '|----+----------------------------------------+------------------------------+------------------------------|'
            )
            print('|{:^4}|{:^40}|{:^30}|{:^30}|'.format(count, param.name,
                                                        ori_shape, param.shape))
            count += 1
        print(
            '|----+----------------------------------------+------------------------------+------------------------------|'
        )

    def on_compression_begin(self, context):
        self._compute_sensitivities(context)
        self._load_sensitivities()
        params, ratios = self._get_best_ratios()

        self._prune_parameters(context.train_graph, context.train_graph.scope,
                               params, ratios, context.place)

        print('\n################################')
        print('#          pruning eval graph    #')
        print('################################\n')
        self._prune_graph(context.eval_graph, context.train_graph)
        print(
            '------------------SensitivePruneStrategy.on_compression_begin finish--------------------------------'
        )
