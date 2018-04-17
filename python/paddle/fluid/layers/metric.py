#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""
All layers just related to metric.
"""

import warnings
from ..layer_helper import LayerHelper
from ..initializer import Normal, Constant
from ..framework import Variable
from ..param_attr import ParamAttr
import nn

__all__ = ['accuracy', 'auc']


def accuracy(input, label, k=1, correct=None, total=None):
    """
    This function computes the accuracy using the input and label.
    The output is the top k inputs and their indices.
    """
    helper = LayerHelper("accuracy", **locals())
    topk_out, topk_indices = nn.topk(input, k=k)
    acc_out = helper.create_tmp_variable(dtype="float32")
    if correct is None:
        correct = helper.create_tmp_variable(dtype="int64")
    if total is None:
        total = helper.create_tmp_variable(dtype="int64")
    helper.append_op(
        type="accuracy",
        inputs={
            "Out": [topk_out],
            "Indices": [topk_indices],
            "Label": [label]
        },
        outputs={
            "Accuracy": [acc_out],
            "Correct": [correct],
            "Total": [total],
        })
    return acc_out


def auc(input, label, curve='ROC', num_thresholds=200):
    warnings.warn(
        "This interface not recommended, fluid.layers.auc compute the auc at every minibatch, \
        but can not aggregate them and get the pass AUC, because pass \
        auc can not be averaged with weighted from the minibatch auc value. \
        Please use fluid.metrics.Auc, it can compute the auc value via Python natively, \
        which can get every minibatch and every pass auc value.", Warning)
    helper = LayerHelper("auc", **locals())
    topk_out = helper.create_tmp_variable(dtype=input.dtype)
    topk_indices = helper.create_tmp_variable(dtype="int64")
    topk_out, topk_indices = nn.topk(input, k=k)
    auc_out = helper.create_tmp_variable(dtype="float32")
    if correct is None:
        correct = helper.create_tmp_variable(dtype="int64")
    if total is None:
        total = helper.create_tmp_variable(dtype="int64")
    helper.append_op(
        type="accuracy",
        inputs={
            "Out": [topk_out],
            "Indices": [topk_indices],
            "Label": [label]
        },
        attrs={"curve": curve,
               "num_thresholds": num_thresholds},
        outputs={"AUC": [auc_out], })
    return auc_out
