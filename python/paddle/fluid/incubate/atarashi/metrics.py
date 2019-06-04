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

import sys
import os
import numpy as np
import itertools

import paddle.fluid as F
import paddle.fluid.layers as L
import sklearn.metrics

from paddle.fluid.incubate.atarashi import log


class Metrics(object):
    def __init__(self):
        self.saver = []

    @property
    def tensor(self):
        pass

    def update(self, *args):
        pass

    def eval(self):
        pass


class Mean(Metrics):
    def __init__(self, t):
        self.t = t
        self.reset()

    def reset(self):
        self.saver = np.array([])

    @property
    def tensor(self):
        return self.t.name,

    def update(self, args):
        t, = args
        t = t.reshape([-1])
        self.saver = np.concatenate([self.saver, t])

    def eval(self):
        return self.saver.mean()


class Ppl(Mean):
    def eval(self):
        return np.exp(self.saver.mean())


class Acc(Mean):
    def __init__(self, label, pred):
        self.eq = L.equal(pred, label)
        self.reset()

    @property
    def tensor(self):
        return self.eq.name,


class Auc(Metrics):
    def __init__(self, label, pred):
        self.pred = pred
        self.label = label
        self.reset()

    def reset(self):
        self.pred_saver = np.array([], dtype=np.float32)
        self.label_saver = np.array([], dtype=np.bool)

    @property
    def tensor(self):
        return [self.pred.name, self.label.name]

    def update(self, args):
        pred, label = args
        pred = pred.reshape([-1]).astype(np.float32)
        label = label.reshape([-1]).astype(np.bool)
        self.pred_saver = np.concatenate([self.pred_saver, pred])
        self.label_saver = np.concatenate([self.label_saver, label])

    def eval(self):
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(
            self.label_saver.astype(np.int64), self.pred_saver)
        auc = sklearn.metrics.auc(fpr, tpr)
        return auc


class PrecisionAtThreshold(Auc):
    def __init__(self, label, pred, threshold=0.5):
        super().__init__(label, pred)
        self.threshold = threshold

    def eval(self):
        infered = self.pred_saver > self.threshold
        correct_num = np.array(infered & self.label_saver).sum()
        infer_num = infered.sum()
        return correct_num / (infer_num + 1.e-6)


#class SemanticRecallMetrics(Metrics):
#    def __init__(self, qid, vec, type_id):
#        self.qid = qid
#        self.vec = vec
#        self.type_id = type_id
#        self.reset()
#
#    def reset(self):
#        self.saver = []
#
#    @property
#    def tensor(self):
#        return [self.qid, self.vec, self.type_id]
#
#    def update(self, args):
#        qid, vec, type_id = args
#        self.saver.append((qid, vec, type_id))
#
#    def eval(self):
#        dic = {}
#        for qid, vec, type_id in self.saver():
#            dic.setdefault(i, {}).setdefault(k, []).append(vec)
#        
#        for qid in dic:
#            assert len(dic[qid]) == 3
#            qvec = np.arrray(dic[qid][0])
#            assert len(qvec) == 1
#            ptvec = np.array(dic[qid][1])
#            ntvec = np.array(dic[qid][2])
#
#            np.matmul(qvec, np.transpose(ptvec))
#            np.matmul(qvec, np.transpose(ntvec))
#            
