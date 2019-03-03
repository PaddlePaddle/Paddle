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
from .... import optimizer
from .... import Executor
import numpy as np
import copy
import re
import logging
import sys

__all__ = ['DistillationStrategy']

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


class DistillationStrategy(Strategy):
    def __init__(self, distillers=None, start_epoch=0, end_epoch=10):
        super(DistillationStrategy, self).__init__(start_epoch, end_epoch)
        self.distillers = distillers

    def on_compression_begin(self, context):
        # load from checkpoint
        if context.epoch_id > 0:
            if context.epoch_id > self.start_epoch and context.epoch_id < self.end_epoch:
                logger.info('Restore DistillationStrategy')
                self._create_distillation_graph(context)
                logger.info('Restore DistillationStrategy finish.')

    def on_epoch_begin(self, context):
        if self.start_epoch == context.epoch_id:
            logger.info('DistillationStrategy::on_epoch_begin.')
            self._create_distillation_graph(context)
            logger.info('DistillationStrategy set optimize_graph.')

    def _create_distillation_graph(self, context):
        teacher = context.teacher_graphs[0]
        for var in teacher.program.list_vars():
            var.stop_gradient = True
        graph = context.train_graph.clone()
        graph.merge(teacher)
        graph.out_nodes['student_loss'] = graph.out_nodes['loss']
        for distiller in self.distillers:
            graph = distiller.distiller_loss(graph)
        startup_program = Program()
        with program_guard(graph.program, startup_program):
            context.distiller_optimizer._name = 'distillation_optimizer'
            context.distiller_optimizer.minimize(
                graph.get_var(graph.out_nodes['loss']))
        exe = Executor(context.place)
        exe.run(startup_program, scope=graph.scope)

        context.put('distillation_backup_optimize_graph',
                    context.optimize_graph)
        context.optimize_graph = graph

    def on_epoch_end(self, context):
        if context.epoch_id == (self.end_epoch - 1):
            logger.info('DistillationStrategy::on_epoch_end.')
            context.optimize_graph = context.get(
                'distillation_backup_optimize_graph')
            logger.info(
                'DistillationStrategy set context.optimize_graph to None.')
