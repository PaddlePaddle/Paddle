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

from .prune_strategy import PruneStrategy
import re
import logging

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
                 pruned_params='conv.*_weights'):
        """
        """
        super(AutoPruneStrategy, self).__init__(pruner, start_epoch, end_epoch,
                                                target_ratio, metric_name,
                                                pruned_params)
        self._max_target_ratio = target_ratio + delta
        self._min_target_ratio = target_ratio - delta
        self.controller = controller
        self._pruned_param_names = []
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
        self._current_tokens = self._get_init_tokens(context)

        constrain_func = functools.partial(
            self._constrain_func, context=context)

        self._controller.reset(
            range_table, constrain_func, init_tokens=self._current_tokens)

    def _constrain_func(self, tokens, context=None):
        """Check whether the tokens meet constraint."""
        ori_flops = context.eval_graph.flops()
        params, ratios = self._get_prune_ratios(tokens)
        param_shape_backup = {}
        self._prune_parameters(
            context.eval_graph,
            context.scope,
            params,
            ratios,
            context.place,
            only_graph=True,
            param_shape_backup=param_shape_backup)
        context.eval_graph.update_groups_of_conv()
        flops = context.eval_graph.flops()
        # restore params shape in eval graph
        for param in param_shape_backup.keys():
            context.eval_graph.var(param).set_shape(param_shape_backup[param])

        flops_ratio = (1 - flops / ori_flops)
        if flops_ratio >= self._min_target_ratio and flops_ratio <= self._max_target_ratio:
            return True
        else:
            return False

    def _get_init_tokens(self, context):
        return self._get_uniform_ratios(context)

    def _get_uniform_ratios(self, context):
        """
        Search a group of uniformratios for pruning target flops.
        """
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
                self._pruned_param_names,
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
            param_shape_backup = {}

            if abs(pruned_flops - self.target_ratio) < 1e-2:
                break
            if pruned_flops > self.target_ratio:
                max_ratio = ratio
            else:
                min_ratio = ratio
        _logger.info('Get ratios: {}'.format([round(r, 2) for r in ratios]))
        return ratios

    def _get_prune_ratios(self, tokens):
        return self._pruned_param_names, tokens

    def on_epoch_begin(self, context):
        if context.epoch_id >= self.start_epoch and context.epoch_id <= self.end_epoch and (
                self._retrain_epoch == 0 or
            (context.epoch_id - self.start_epoch) % self._retrain_epoch == 0):
            self._current_tokens = self.controller.next_tokens()
            params, ratios = self._get_prune_ratios(self._current_tokens)

            self._param_shape_backup = {}
            self._param_backup = {}
            self._prune_parameters(
                context.optimize_graph,
                context.scope,
                params,
                ratios,
                context.place,
                param_backup=self._param_backup,
                param_shape_backup=self._param_shape_backup)
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
            self._current_reward = context.eval_results[-1]
            self._controller.update(self._current_token, self._current_reward)

            # restore pruned parameters
            for param_name in self._param_backup.keys():
                param_t = context.scope.find_var(param_name).get_tensor()
                param_t.set(self.param_backup[param_name], context.place)
            # restore shape of parameters
            for param in self._param_shape_backup.keys():
                context.eval_graph.var(param).set_shape(
                    self._param_shape_backup[param])
                context.optimize_graph.var(param).set_shape(
                    self._param_shape_backup[param])

            context.optimize_graph.update_groups_of_conv()
            context.eval_graph.update_groups_of_conv()
            context.optimize_graph.compile()  # to update the compiled program

            context.eval_graph.compile(
                for_parallel=False,
                for_test=True)  # to update the compiled program

        elif context.epoch_id == self.end_epoch:
            # restore pruned parameters
            for param_name in self._param_backup.keys():
                param_t = context.scope.find_var(param_name).get_tensor()
                param_t.set(self.param_backup[param_name], context.place)
            # restore shape of parameters
            for param in self._param_shape_backup.keys():
                context.eval_graph.var(param).set_shape(
                    self._param_shape_backup[param])
                context.optimize_graph.var(param).set_shape(
                    self._param_shape_backup[param])

            context.optimize_graph.update_groups_of_conv()
            context.eval_graph.update_groups_of_conv()
            context.optimize_graph.compile()  # to update the compiled program

            context.eval_graph.compile(
                for_parallel=False,
                for_test=True)  # to update the compiled program

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
