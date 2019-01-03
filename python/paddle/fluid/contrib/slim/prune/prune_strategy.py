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
from ....framework import Program, program_guard
from .... import layers
import numpy as np

__all__ = ['SensitivePruneStrategy', 'PruneStrategy']


class SensitivePruneStrategy(Strategy):
    def __init__(self,
                 pruner=None,
                 start_epoch=0,
                 end_epoch=10,
                 delta_rate=0.20,
                 acc_loss_threshold=0.2,
                 sensitivities=None):
        super(SensitivePruneStrategy, self).__init__(start_epoch, end_epoch)
        self.pruner = pruner
        self.delta_rate = delta_rate
        self.acc_loss_threshold = acc_loss_threshold
        self.sensitivities = sensitivities

    def _compute_sensitivities(self, context):
        """
        Computing the sensitivities of all parameters.
        """
        scope = context.scope
        param_backup = None
        accuracy = context.exe.run(context.graph, scope=scope)[0]
        for param in context.graph.all_parameters():
            ratio = 1
            while ratio > 0:
                # prune parameter by ratio
                prune_program = fluid.Program()
                prune_program.global_block().clone_variable(param)
                with fluid.program_guide(prune_program):
                    param_backup = fluid.layers.assign(param)
                    zero_mask = pruner.prune(param, ratio=ratio)
                    pruned_param = param * zero_mask
                    layers.assign(input=pruned_param, output=param)
                context.program_exe.run(prune_program, scope=context.scope)

                # get accuracy after pruning and update self.sensitivities
                pruned_accuracy = context.exe.run(context.graph, scope=scope)[0]
                acc_loss = accuracy - pruned_accuracy
                self.sensitivities[param.name]['pruned_percent'].append(ratio)
                self.sensitivities[param.name]['acc_loss'].append(acc_loss)

                # restore pruned parameter
                restore_program = fluid.Program()
                restore_program.global_block().clone_variable(param)
                restore_program.global_block().clone_variable(param_backup)
                with fluid.program_guide(restore_program):
                    param_backup = fluid.layers.assign(
                        input=param_backup, out=param)
                context.program_exe.run(restore_program, scope=context.scope)

                ratio -= delta_rate

    def _get_best_ratios(self):
        no_loss_ratios = {}
        for param in self.sensitivities:
            for ratio, loss in izip(self.sensitivities[param]['pruned_percent'],
                                    self.sensitivities[param]['acc_loss']):
                if loss == 0:
                    no_loss_ratios[param] = ratio
        ratio_sum = np.sum(no_loss_ratios.values())
        return no_loss_ratios.keys(), self.target_ratio * no_loss_ratios.values(
        ) / ratio_sum

    def _prune_parameter(self, context, param, ratio):
        """
        
        """
        param_t = context.scope.find_var(param.name).get_tensor()
        pruned_idx, pruned_axis = pruner.cal_pruned_idx(param.name,
                                                        np.array(param_t),
                                                        ratio)
        pruned_param = pruner.prune_tensor(
            np.array(param_t), pruned_idx, pruned_axis)
        param.desc.set_shape(pruned_param.shape)
        param_t.set(pruned_param, context.place)
        return pruned_idx, pruned_axis

    def _prune_parameter_by_idx(self, context, param, pruned_idx, pruned_axis):
        param_t = context.scope.find_var(param.name).get_tensor()
        pruned_param = pruner.prune_tensor(
            np.array(param_t), pruned_idx, pruned_axis)
        param.desc.set_shape(pruned_param.shape)
        param_t.set(pruned_param, context.place)

    def _forward_search_related_layer(self, graph, param):
        visited = {}
        for op in context.graph.ops():
            visited[op.index()] = False
        stack = []
        for op in context.graph.ops():
            if param.name in op.input_names():
                stack.append(op)
        visit_path = []
        while len(stack) > 0:
            top_op = stack[len(stack) - 1]
            if visited[top_op.index] == False:
                visit_path.append(top_op)
                visited[top_op.index] = True
            next_ops = None
            if top_op.type(
            ) == "conv2d" and param.name not in top_op.input_names():
                next_ops = None
            else:
                next_ops = self._get_next_unvisited_op(context.graph, visited,
                                                       top_op)

            if next_op == None:
                statck.pop()
            else:
                statck += next_ops
        return visit_path

    def _get_next_unvisited_op(graph, visited, top_op):
        next_ops = []
        for out_name in top_op.output_names():
            for op in graph.ops():
                if out_name in op.input_names() and visited[op.index(
                )] == False:
                    next_ops.append(op)
        return next_ops

    def _pruning_ralated_op(self, graph, param, ratio):
        related_ops = self.forward_search_related_op(graph, param)
        pruned_idxs = None
        corrected_idxs = None

        for idx, op in enumerate(related_ops):

            if op.type() == "conv2d":
                if param.name in op.input_names():
                    pruned_idxs, pruned_axis = self._prune_parameter(
                        context, param, ratio)
                    corrected_idxs = pruned_idxs[:]
                else:
                    for param_name in op.input_names():
                        if param_name in graph.all_parameters():
                            conv_param = graph.get_param(param_name)
                            self._prune_parameter_by_idx(
                                context,
                                conv_param,
                                corrected_idxs,
                                pruned_axis=1)
            elif op.type() == "add":
                for param_name in op.input_names():
                    if param_name in graph.all_parameters():
                        bias_param = graph.get_param(param_name)
                        self._prune_parameter_by_idx(
                            context, bias_param, corrected_idxs, pruned_axis=0)

            elif op.type() == "concat":
                concat_inputs = op.input_names()
                last_op = related_ops[idx - 1]
                for out_name in last_op.output_names():
                    if out_name in concat_inputs:
                        concat_idx = concat_inputs.index(out_name)
                offset = 0
                for ci in range(concat_idx):
                    offset += graph.get_param(concat_inputs[ci]).shape[1]
                corrected_idxs = [x + offset for x in pruned_idxs]

            elif type == "batch_norm":
                mean = graph.get_param(op.input_names()[2])
                variance = graph.get_param(op.input_names()[3])
                alpha = graph.get_param(op.input_names()[0])
                beta = graph.get_param(op.input_names()[1])
                self._prune_parameter(
                    context, mean, corrected_idxs, pruned_axis=0)
                self._prune_parameter(
                    context, variance, corrected_idxs, pruned_axis=0)
                self._prune_parameter(
                    context, alpha, corrected_idxs, pruned_axis=0)
                self._prune_parameter(
                    context, beta, corrected_idxs, pruned_axis=0)

    def _prune_parameters(self, context, params, ratios):
        """
        Pruning parameters.
        """
        for param, ratio in izip(params, ratios):
            self._pruning_ralated_layer(param, ratio)

    def on_compression_begin(self, context):
        self._compute_sensitivities(context)
        params, ratios = self._get_best_ratios()
        self._prune_parameters(context, params, ratios)


class PruneStrategy(Strategy):
    """
    The strategy that pruning weights by threshold or ratio iteratively.
    """

    def __init__(self,
                 pruner,
                 mini_batch_pruning_frequency=1,
                 start_epoch=0,
                 end_epoch=10):
        super(PruneStrategy, self).__init__(start_epoch, end_epoch)
        self.pruner = pruner
        self.mini_batch_pruning_frequency = mini_batch_pruning_frequency

    def _triger(self, context):
        return (context.batch_id % self.mini_batch_pruning_frequency == 0 and
                self.start_epoch <= context.epoch_id < self.end_epoch)

    def on_batch_end(self, context):
        if self._triger(context):
            prune_program = Program()
            with program_guard(prune_program):
                for param in context.graph.all_parameters():
                    prune_program.global_block().clone_variable(param)
                    p = prune_program.global_block().var(param.name)
                    zeros_mask = self.pruner.prune(p)
                    pruned_param = p * zeros_mask
                    layers.assign(input=pruned_param, output=param)
            context.program_exe.run(prune_program, scope=context.scope)
