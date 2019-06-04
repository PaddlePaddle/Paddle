#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import sys
import os
import itertools

import numpy as np
import paddle.fluid as F
import paddle.fluid.layers as L

from paddle.fluid.incubate.atarashi import log
from paddle.fluid.incubate.atarashi.train import Saver


class Exporter(object):
    def export(self, exe, program, eval_result, state):
        raise NotImplementedError()


class BestExporter(Exporter):
    def __init__(self, export_dir, key, export_highest=True):
        self._export_dir = export_dir
        self._best = None
        if export_highest:
            self.cmp_fn = lambda old, new: old[key] < new[key]
        else:
            self.cmp_fn = lambda old, new: old[key] > new[key]

    def export(self, exe, program, eval_result, state):
        if self._best is None:
            self._best = eval_result
            log.debug('[Best Exporter]: skip step %d' % state.gstep)
            return
        if self.cmp_fn(self._best, eval_result):
            log.debug('[Best Exporter]: export to %s' % self._export_dir)
            saver = Saver(
                self._export_dir, exe, program=program, max_ckpt_to_keep=1)
            saver.save(state)

            self._best = eval_result
        else:
            log.debug('[Best Exporter]: skip step %s' % state.gstep)
