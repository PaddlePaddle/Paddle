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

from ..layer_helper import LayerHelper
from ..initializer import Normal, Constant
from ..framework import Variable
from ..param_attr import ParamAttr

__all__ = ['accuracy']


def accuracy(input, label, k=1, correct=None, total=None):
    """
    This function computes the accuracy using the input and label.
    The output is the top_k inputs and their indices.
    """
    helper = LayerHelper("accuracy", **locals())
    topk_out = helper.create_tmp_variable(dtype=input.dtype)
    topk_indices = helper.create_tmp_variable(dtype="int64")
    helper.append_op(
        type="top_k",
        inputs={"X": [input]},
        outputs={"Out": [topk_out],
                 "Indices": [topk_indices]},
        attrs={"k": k})
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
