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
        pass

    def _prune_parameters(self, context, params, ratios):
        """
        Pruning parameters.
        """
        prune_program = fluid.Program()
        for param in params:
            prune_program.global_block().clone_variable(param)
        with fluid.program_guide(prune_program):
            zero_masks = pruner.prune(params, ratios=ratios)
            for param, mask in izip(params, zero_masks):
                pruned_param = param * zero_mask
                layers.assign(input=pruned_param, output=param)
        context.program_exe.run(prune_program, scope=context.scope)

    def _prune_graph(self, context):
        pass

    def on_compression_begin(self, context):
        self._compute_sensitivities(context)
        params, ratios = self._get_best_ratios()
        masks = self._prune_parameters(context, params, ratios)
        self._prune_graph(context, params, mask)


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
