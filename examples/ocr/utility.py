"""Contains common utility functions."""
#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import distutils.util
import numpy as np
import paddle.fluid as fluid
import six

from hapi.metrics import Metric
from hapi.callbacks import ProgBarLogger


def print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


class SeqAccuracy(Metric):
    def __init__(self, name=None, *args, **kwargs):
        super(SeqAccuracy, self).__init__(*args, **kwargs)
        self._name = 'seq_acc'
        self.reset()

    def add_metric_op(self, output, label, mask, *args, **kwargs):
        pred = fluid.layers.flatten(output, axis=2)
        score, topk = fluid.layers.topk(pred, 1)
        return topk, label, mask

    def update(self, topk, label, mask, *args, **kwargs):
        topk = topk.reshape(label.shape[0], -1)
        seq_len = np.sum(mask, -1)
        acc = 0
        for i in range(label.shape[0]):
            l = int(seq_len[i] - 1)
            pred = topk[i][:l - 1]
            ref = label[i][:l - 1]
            if np.array_equal(pred, ref):
                self.total += 1
                acc += 1
            self.count += 1
        return float(acc) / label.shape[0]

    def reset(self):
        self.total = 0.
        self.count = 0.

    def accumulate(self):
        return float(self.total) / self.count

    def name(self):
        return self._name


class MyProgBarLogger(ProgBarLogger):
    def __init__(self, log_freq=1, verbose=2, train_bs=None, eval_bs=None):
        super(MyProgBarLogger, self).__init__(log_freq, verbose)
        self.train_bs = train_bs
        self.eval_bs = eval_bs if eval_bs else train_bs

    def on_train_batch_end(self, step, logs=None):
        logs = logs or {}
        logs['loss'] = [l / self.train_bs for l in logs['loss']]
        super(MyProgBarLogger, self).on_train_batch_end(step, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['loss'] = [l / self.train_bs for l in logs['loss']]
        super(MyProgBarLogger, self).on_epoch_end(epoch, logs)

    def on_eval_batch_end(self, step, logs=None):
        logs = logs or {}
        logs['loss'] = [l / self.eval_bs for l in logs['loss']]
        super(MyProgBarLogger, self).on_eval_batch_end(step, logs)

    def on_eval_end(self, logs=None):
        logs = logs or {}
        logs['loss'] = [l / self.eval_bs for l in logs['loss']]
        super(MyProgBarLogger, self).on_eval_end(logs)


def index2word(ids):
    return [chr(int(k + 33)) for k in ids]


def postprocess(seq, bos_idx=0, eos_idx=1):
    if type(seq) is np.ndarray:
        seq = seq.tolist()
    eos_pos = len(seq) - 1
    for i, idx in enumerate(seq):
        if idx == eos_idx:
            eos_pos = i
            break
    seq = [
        idx for idx in seq[:eos_pos + 1] if idx != bos_idx and idx != eos_idx
    ]
    return seq


class SeqBeamAccuracy(Metric):
    def __init__(self, name=None, *args, **kwargs):
        super(SeqBeamAccuracy, self).__init__(*args, **kwargs)
        self._name = 'seq_acc'
        self.reset()

    def add_metric_op(self, output, label, mask, *args, **kwargs):
        return output, label, mask

    def update(self, preds, labels, masks, *args, **kwargs):
        preds = preds[:, :, np.newaxis] if len(preds.shape) == 2 else preds
        preds = np.transpose(preds, [0, 2, 1])
        seq_len = np.sum(masks, -1)
        acc = 0
        for i in range(labels.shape[0]):
            l = int(seq_len[i] - 1)
            #ref = labels[i][: l - 1]
            ref = np.array(postprocess(labels[i]))
            pred = preds[i]
            for idx, beam in enumerate(pred):
                beam_pred = np.array(postprocess(beam))
                if np.array_equal(beam_pred, ref):
                    self.total += 1
                    acc += 1
                    break
            self.count += 1
        return float(acc) / labels.shape[0]

    def reset(self):
        self.total = 0.
        self.count = 0.

    def accumulate(self):
        return float(self.total) / self.count

    def name(self):
        return self._name
