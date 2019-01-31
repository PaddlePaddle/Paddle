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

__all__ = ['FSPDistillationStrategy']


class FSPDistillationStrategy(Strategy):
    def __init__(self, distiller=None, start_epoch=0, end_epoch=10):
        super(FSPDistillationStrategy, self).__init__(start_epoch, end_epoch)
        self.distiller = distiller
        self.train_graph_backup = None

    def on_epoch_begin(self, context):
        if self.start_epoch == context.epoch_id:
            self.train_graph_backup = context.train_graph
            graph = self.distiller.distiller_graph(
                context.eval_graph, context.teacher_graphs, context.optimizer,
                context.place)
            context.train_graph = graph

    def on_epoch_end(self, context):
        if context.epoch_id == (self.end_epoch - 1):
            context.train_graph = self.train_graph_backup
