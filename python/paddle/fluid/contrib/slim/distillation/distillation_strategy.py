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
import numpy as np
import copy
import re
import logging
import sys

__all__ = ['FSPDistillationStrategy']

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


class FSPDistillationStrategy(Strategy):
    def __init__(self, distiller=None, start_epoch=0, end_epoch=10):
        super(FSPDistillationStrategy, self).__init__(start_epoch, end_epoch)
        self.distiller = distiller
        self.backup = None

    def on_compression_begin(self, context):
        # load from checkpoint
        if context.epoch_id > 0:
            if context.epoch_id > self.start_epoch and context.epoch_id < self.end_epoch:
                logger.info('Restore FSPDistillationStrategy')
                graph = self.distiller.distiller_graph(
                    context.train_graph, context.teacher_graphs,
                    context.distiller_optimizer, context.place)
                context.optimize_graph = graph
                logger.info('Restore FSPDistillationStrategy finish.')

    def on_epoch_begin(self, context):
        if self.start_epoch == context.epoch_id:
            logger.info('FSPDistillationStrategy::on_epoch_begin.')
            graph = self.distiller.distiller_graph(
                context.train_graph, context.teacher_graphs,
                context.distiller_optimizer, context.place)
            self.backup = context.optimize_graph
            context.optimize_graph = graph
            logger.info('FSPDistillationStrategy set optimize_graph.')

    def on_epoch_end(self, context):
        if context.epoch_id == (self.end_epoch - 1):
            logger.info('FSPDistillationStrategy::on_epoch_end.')
            context.optimize_graph = self.backup
            logger.info(
                'FSPDistillationStrategy set context.optimize_graph to None.')
