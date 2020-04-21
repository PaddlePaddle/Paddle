# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle.fluid as fluid

from metrics import Metric
from callbacks import ProgBarLogger


class TrainCallback(ProgBarLogger):
    def __init__(self, ppl, log_freq, verbose=2):
        super(TrainCallback, self).__init__(log_freq, verbose)
        self.ppl = ppl

    def on_train_begin(self, logs=None):
        super(TrainCallback, self).on_train_begin(logs)
        self.train_metrics = ["ppl"]  # remove loss to not print it

    def on_epoch_begin(self, epoch=None, logs=None):
        super(TrainCallback, self).on_epoch_begin(epoch, logs)
        self.ppl.reset()

    def on_train_batch_end(self, step, logs=None):
        logs["ppl"] = self.ppl.cal_acc_ppl(logs["loss"][0], logs["batch_size"])
        if step > 0 and step % self.ppl.reset_freq == 0:
            self.ppl.reset()
        super(TrainCallback, self).on_train_batch_end(step, logs)

    def on_eval_begin(self, logs=None):
        super(TrainCallback, self).on_eval_begin(logs)
        self.eval_metrics = ["ppl"]
        self.ppl.reset()

    def on_eval_batch_end(self, step, logs=None):
        logs["ppl"] = self.ppl.cal_acc_ppl(logs["loss"][0], logs["batch_size"])
        super(TrainCallback, self).on_eval_batch_end(step, logs)


class PPL(Metric):
    def __init__(self, reset_freq=100, name=None):
        super(PPL, self).__init__()
        self._name = name or "ppl"
        self.reset_freq = reset_freq
        self.reset()

    def add_metric_op(self, pred, label):
        seq_length = label[0]
        word_num = fluid.layers.reduce_sum(seq_length)
        return word_num

    def update(self, word_num):
        self.word_count += word_num[0]
        return word_num

    def reset(self):
        self.total_loss = 0
        self.word_count = 0

    def accumulate(self):
        return self.word_count

    def name(self):
        return self._name

    def cal_acc_ppl(self, batch_loss, batch_size):
        self.total_loss += batch_loss * batch_size
        ppl = np.exp(self.total_loss / self.word_count)
        return ppl