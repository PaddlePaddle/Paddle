#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from .metrics import *
from . import metrics

from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import check_variable_and_dtype
from ..fluid.framework import core, _varbase_creator, in_dygraph_mode
from ..fluid.layers.metric_op import auc
from ..fluid.layers.nn import chunk_eval, mean_iou, topk

__all__ = metrics.__all__ + [
    'accuracy',
    'auc',
    'chunk_eval',
    'mean_iou',
]


def accuracy(input, label, k=1, correct=None, total=None, name=None):
    """
    accuracy layer.
    Refer to the https://en.wikipedia.org/wiki/Precision_and_recall

    This function computes the accuracy using the input and label.
    If the correct label occurs in top k predictions, then correct will increment by one.
    Note: the dtype of accuracy is determined by input. the input and label dtype can be different.

    Args:
        input(Tensor): The input of accuracy layer, which is the predictions of network. A Tensor with type float32,float64.
            The shape is ``[sample_number, class_dim]`` .
        label(Tensor): The label of dataset. Tensor with type int32,int64. The shape is ``[sample_number, 1]`` .
        k(int): The top k predictions for each class will be checked. Data type is int64 or int32.
        correct(Tensor): The correct predictions count. A Tensor with type int64 or int32.
        total(Tensor): The total entries count. A tensor with type int64 or int32.
        name(str, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Tensor: The correct rate. A Tensor with type float32.

    Examples:
        .. code-block:: python

            import paddle

            predictions = paddle.to_tensor([[0.2, 0.1, 0.4, 0.1, 0.1], [0.2, 0.3, 0.1, 0.15, 0.25]], dtype='float32')
            label = paddle.to_tensor([[2], [0]], dtype="int64")
            result = paddle.metric.accuracy(input=predictions, label=label, k=1)
            # [0.5]
    """
    if in_dygraph_mode():
        if correct is None:
            correct = _varbase_creator(dtype="int32")
        if total is None:
            total = _varbase_creator(dtype="int32")

        topk_out, topk_indices = topk(input, k=k)
        _acc, _, _ = core.ops.accuracy(topk_out, topk_indices, label, correct,
                                       total)
        return _acc

    helper = LayerHelper("accuracy", **locals())
    check_variable_and_dtype(input, 'input', ['float16', 'float32', 'float64'],
                             'accuracy')
    topk_out, topk_indices = topk(input, k=k)
    acc_out = helper.create_variable_for_type_inference(dtype="float32")
    if correct is None:
        correct = helper.create_variable_for_type_inference(dtype="int32")
    if total is None:
        total = helper.create_variable_for_type_inference(dtype="int32")
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
