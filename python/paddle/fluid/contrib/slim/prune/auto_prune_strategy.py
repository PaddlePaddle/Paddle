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
from .... import layers
from .prune_strategy import PruneStrategy
import prettytable as pt
import numpy as np
from scipy.optimize import leastsq
import copy
import re
import os
import pickle
import logging
import sys

__all__ = ['AutoPruneStrategy']

logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s')
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class AutoPruneStrategy(PruneStrategy):
    """
    Automatic pruning strategy.
    """

    def __init__(self,
                 pruner=None,
                 controller=None,
                 start_epoch=0,
                 end_epoch=0,
                 target_ratio=0.5,
                 delta=0.1,
                 metric_name='top1_acc',
                 pruned_params='conv.*_weights',
                 eval_rate=None):
        """
        """
        super(AutoPruneStrategy, self).__init__(pruner, start_epoch, end_epoch,
                                                target_ratio, metric_name,
                                                pruned_params)
        self._max_target_ratio = target_ratio + delta
        self._min_target_ratio = target_ratio - delta
        self.pruned_list = []
        self.backup = {}
        self.param_shape_backup = {}
        self.eval_rate = eval_rate
        self.controller = controller
        self._pruned_param_names = []
        self.iter = 0
        self._retrain_epoch = 0
        self._last_tokens = None
        self._current_tokens = None
        self._last_reward = -1
        self._current_reward = -1

    def on_compression_begin(self, context):
        pruned_params = []
        for param in context.eval_graph.all_parameters():
            if re.match(self.pruned_params, param.name()):
                self._pruned_param_names.append(param.name())
        range_table = [1] * len(self._pruned_param_names)
        self._current_tokens = self._controller.reset(range_table)

    def _get_prune_ratios(self, tokens):
        return self._pruned_param_names, tokens

    def _backup(self, context):
        """Backup graph before pruing."""
        pass

    def _restore(self, context):
        """Restore graph after pruning."""
        pass

    def on_epoch_begin(self, context):
        if context.epoch_id >= self.start_epoch and context.epoch_id <= self.end_epoch and (
                self._retrain_epoch == 0 or
            (context.epoch_id - self.start_epoch) % self._retrain_epoch == 0):
            self._current_tokens = self.controller.next_tokens(
                self._current_tokens)
            params, ratios = self._get_prune_ratios(self._current_tokens)
            self._backup()
            self._prune_parameters(context.optimize_graph, context.scope,
                                   params, ratios, context.place)
            self._prune_graph(context.eval_graph, context.optimize_graph)
            context.optimize_graph.update_groups_of_conv()
            context.eval_graph.update_groups_of_conv()
            context.optimize_graph.compile()  # to update the compiled program

            context.eval_graph.compile(
                for_parallel=False,
                for_test=True)  # to update the compiled program
            context.skip_training = (self._retrain_epoch == 0)

    def on_epoch_end(self, context):
        if context.epoch_id >= self.start_epoch and context.epoch_id < self.end_epoch and (
                self._retrain_epoch == 0 or
            (context.epoch_id - self.start_epoch) % self._retrain_epoch == 0):
            self.iter += 1
            self._current_reward = context.eval_results[-1]
            if self._controller.check(self._current_reward, self._last_reward,
                                      self.iter):
                self._last_reward = self._current_reward
                self._last_tokens = self._current_tokens
            if self._current_reward > self._max_reward:
                self._max_reward = self._current_reward
                self._best_tokens = self._current_tokens

            self._restore(context)

        elif context.epoch_id == self.end_epoch:
            self._restore(context)
            params, ratios = self._get_prune_ratios(self._best_tokens)
            self._prune_parameters(context.optimize_graph, context.scope,
                                   params, ratios, context.place)

            self._prune_graph(context.eval_graph, context.optimize_graph)
            context.optimize_graph.update_groups_of_conv()
            context.eval_graph.update_groups_of_conv()
            context.optimize_graph.compile()  # to update the compiled program

            context.eval_graph.compile(
                for_parallel=False,
                for_test=True)  # to update the compiled program
            context.skip_training = False
