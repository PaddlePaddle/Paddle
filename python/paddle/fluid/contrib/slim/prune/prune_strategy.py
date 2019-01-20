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
import numpy as np
import copy
import re
import os
import pickle

__all__ = ['SensitivePruneStrategy']


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
        print "load sensitivities: %s" % self.sensitivities

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
                    'loss': []
                }
            self.sensitivities[param.name]['size'] = param.shape[0]
            ratio = self.delta_rate
            while ratio < 1:
                ratio = round(ratio, 2)
                if ratio in self.sensitivities[param.name]['pruned_percent']:
                    ratio += self.delta_rate
                    continue
                else:
                    print('{} not in {}'.format(
                        ratio,
                        self.sensitivities[param.name]['pruned_percent']))
                # prune parameter by ratio
                self._prune_parameters(
                    context.eval_graph,
                    context.eval_graph.scope, [param.name], [ratio],
                    context.place,
                    lazy=True)
                # get accuracy after pruning and update self.sensitivities
                pruned_metric = self._eval_graph(context)
                loss = metric - pruned_metric
                print "pruned param: {}; {}; loss={}".format(param.name, ratio,
                                                             loss)
                self.sensitivities[param.name]['pruned_percent'].append(ratio)
                self.sensitivities[param.name]['loss'].append(loss)
                self._save_sensitivities()

                # restore pruned parameters
                for param_name in self.backup.keys():
                    param_t = context.eval_graph.scope.find_var(
                        param_name).get_tensor()
                    param_t.set(self.backup[param_name], context.place)

                ratio += self.delta_rate
                if pruned_metric <= 0:
                    break

    def _get_best_ratios(self):
        losses = []
        sizes = []
        for param in self.sensitivities:
            losses.append(self.sensitivities[param]['loss'][self.sensitivities[
                param]['pruned_percent'].index(self.target_ratio)])
            sizes.append(self.sensitivities[param]['size'])
        losses = np.array(losses) / np.max(np.array(losses))
        sensitivities = (1.0 / losses) * np.array(sizes)
        print "sensitivities: %s" % (sensitivities, )
        ratios = sensitivities / np.sum(sensitivities)
        size_ratios = np.array(sizes, dtype='float32') / np.sum(np.array(sizes))
        ratios = (ratios / size_ratios) * self.target_ratio

        return self.sensitivities.keys(), ratios

    def _prune_parameter(self, scope, param, ratio, place, lazy=False):
        param_t = scope.find_var(param.name).get_tensor()
        pruned_idx, pruned_axis = self.pruner.cal_pruned_idx(param.name,
                                                             np.array(param_t),
                                                             ratio)
        pruned_param = self.pruner.prune_tensor(
            np.array(param_t), pruned_idx, pruned_axis, lazy=lazy)
        param.desc.set_shape(pruned_param.shape)
        if lazy:
            self.backup[param.name] = copy.deepcopy(np.array(param_t))
        param_t.set(pruned_param, place)
        return pruned_idx, pruned_axis

    def _prune_parameter_by_idx(self,
                                scope,
                                param,
                                pruned_idx,
                                pruned_axis,
                                place,
                                lazy=False):
        param_t = scope.find_var(param.name).get_tensor()
        pruned_param = self.pruner.prune_tensor(
            np.array(param_t), pruned_idx, pruned_axis, lazy=lazy)
        param.desc.set_shape(pruned_param.shape)
        if lazy:
            self.backup[param.name] = copy.deepcopy(np.array(param_t))
        param_t.set(pruned_param, place)

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
            if param.name in self._inputs(op):
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
                if out_name in self._inputs(op) and visited[op.idx] == False:
                    next_ops.append(op)
        return next_ops if len(next_ops) > 0 else None

    def _prune_add_op(self,
                      graph,
                      scope,
                      op,
                      pruned_idxs,
                      place,
                      excludes=None,
                      lazy=False):
        # pruning bias
        for param_name in self._inputs(op):
            if isinstance(graph.get_var(param_name), Parameter):
                bias_param = graph.get_var(param_name)
                self._prune_parameter_by_idx(
                    scope,
                    bias_param,
                    pruned_idxs,
                    pruned_axis=0,
                    place=place,
                    lazy=lazy)

        for pre_op in graph.pre_ops(op):
            if excludes is not None and pre_op.idx in excludes:
                continue
            if pre_op.type == 'conv2d':
                # pruning parameters of conv2d
                for param_name in self._inputs(pre_op):
                    param = graph.get_var(param_name)
                    if isinstance(param, Parameter):
                        self._prune_parameter_by_idx(
                            scope,
                            param,
                            pruned_idxs,
                            pruned_axis=0,
                            place=place,
                            lazy=lazy)
                        self.pruned_list.append(param.name)
            if pre_op.type == 'elementwise_add':
                self._prune_add_op(
                    graph,
                    scope,
                    pre_op,
                    pruned_idxs,
                    place,
                    excludes=excludes,
                    lazy=lazy)

    def _pruning_ralated_op(self, graph, scope, param, ratio, place,
                            lazy=False):
        related_ops = self._forward_search_related_op(graph, param)

        pruned_idxs = None
        corrected_idxs = None

        for idx, op in enumerate(related_ops):

            if op.type == "conv2d":
                if param.name in self._inputs(op):
                    pruned_idxs, pruned_axis = self._prune_parameter(
                        scope, param, ratio, place, lazy=lazy)
                    corrected_idxs = pruned_idxs[:]
                else:
                    for param_name in self._inputs(op):
                        if isinstance(graph.get_var(param_name), Parameter):
                            conv_param = graph.get_var(param_name)
                            self._prune_parameter_by_idx(
                                scope,
                                conv_param,
                                corrected_idxs,
                                pruned_axis=1,
                                place=place,
                                lazy=lazy)
            elif op.type == "elementwise_add":
                last_op = related_ops[idx - 1]
                self._prune_add_op(
                    graph,
                    scope,
                    op,
                    corrected_idxs,
                    place,
                    excludes=[last_op.idx],
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
                    scope,
                    fc_param,
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

            elif type == "batch_norm":
                bn_inputs = self._inputs(op)
                mean = graph.get_var(bn_inputs[2])
                variance = graph.get_var(bn_inputs[3])
                alpha = graph.get_var(bn_inputs[0])
                beta = graph.get_var(bn_inputs[1])
                self._prune_parameter_by_idx(
                    scope,
                    mean,
                    corrected_idxs,
                    pruned_axis=0,
                    place=place,
                    lazy=lazy)
                self._prune_parameter_by_idx(
                    scope,
                    variance,
                    corrected_idxs,
                    pruned_axis=0,
                    place=place,
                    lazy=lazy)
                self._prune_parameter_by_idx(
                    scope,
                    alpha,
                    corrected_idxs,
                    pruned_axis=0,
                    place=place,
                    lazy=lazy)
                self._prune_parameter_by_idx(
                    scope,
                    beta,
                    corrected_idxs,
                    pruned_axis=0,
                    place=place,
                    lazy=lazy)

    def _prune_parameters(self, graph, scope, params, ratios, place,
                          lazy=False):
        """
        Pruning parameters.
        """
        for param, ratio in zip(params, ratios):
            param = graph.get_var(param)
            if param.name not in self.pruned_list:
                self._pruning_ralated_op(
                    graph, scope, param, ratio, place, lazy=lazy)
                self.pruned_list.append(param.name)

    def on_compression_begin(self, context):
        self._compute_sensitivities(context)
        params, ratios = self._get_best_ratios()
        print("best pruning ratios: {}".format(zip(params, ratios)))
        return
        self._prune_parameters(context.graph, context.graph.scope, params,
                               ratios, context.place)
        print('SensitivePruneStrategy.on_compression_begin finish.')

        if context.get('train_graph_pass'):
            context.train_graph = context.get('train_graph_pass').apply(
                context.graph)
        if context.get('eval_graph_pass'):
            context.eval_graph = context.get('eval_graph_pass').apply(
                context.graph)
        executor = get_executor(context.train_graph, context.place)
        context.put('executor', executor)
