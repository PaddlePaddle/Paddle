# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from .... import Executor
import logging

__all__ = ['DistillationStrategy']

logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s')
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class DistillationStrategy(Strategy):
    def __init__(self, distillers=None, start_epoch=0, end_epoch=0):
        """
        Args:
            distillers(list): A list of distiller used to combine student graph and teacher graph
                              by adding some loss.
            start_epoch(int): The epoch when to merge student graph and teacher graph for
                              distillation training. default: 0
            end_epoch(int): The epoch when to finish distillation training. default: 0
            
        """
        super(DistillationStrategy, self).__init__(start_epoch, end_epoch)
        self.distillers = distillers

    def on_compression_begin(self, context):
        # load from checkpoint
        if context.epoch_id > 0:
            if context.epoch_id > self.start_epoch and context.epoch_id < self.end_epoch:
                _logger.info('Restore DistillationStrategy')
                self._create_distillation_graph(context)
                _logger.info('Restore DistillationStrategy finish.')

    def on_epoch_begin(self, context):
        if self.start_epoch == context.epoch_id:
            _logger.info('DistillationStrategy::on_epoch_begin.')
            self._create_distillation_graph(context)
            _logger.info('DistillationStrategy set optimize_graph.')

    def _create_distillation_graph(self, context):
        """
        step 1: Merge student graph and teacher graph into distillation graph.
        step 2: Add loss into distillation graph by distillers.
        step 3: Append backward ops and optimize ops into distillation graph for training.
        """
        # step 1
        teacher = context.teacher_graphs[0]
        for var in teacher.program.list_vars():
            var.stop_gradient = True
        graph = context.train_graph.clone()
        graph.merge(teacher)
        graph.out_nodes['student_loss'] = graph.out_nodes['loss']

        # step 2
        for distiller in self.distillers:
            graph = distiller.distiller_loss(graph)

        # step 3
        startup_program = Program()
        with program_guard(graph.program, startup_program):
            context.distiller_optimizer._name = 'distillation_optimizer'
            context.distiller_optimizer.minimize(
                graph.var(graph.out_nodes['loss'])._var)
        exe = Executor(context.place)
        exe.run(startup_program, scope=context.scope)

        # backup graph for fine-tune after distillation
        context.put('distillation_backup_optimize_graph',
                    context.optimize_graph)
        context.optimize_graph = graph

    def on_epoch_end(self, context):
        if context.epoch_id == (self.end_epoch - 1):
            _logger.info('DistillationStrategy::on_epoch_end.')
            # restore optimize_graph for fine-tune or other strategy in next stage.
            context.optimize_graph = context.get(
                'distillation_backup_optimize_graph')
            _logger.info(
                'DistillationStrategy set context.optimize_graph to None.')
