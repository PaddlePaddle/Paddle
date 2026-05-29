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

"""Backward-compatible functional accuracy that delegates to the C++ accuracy op."""

from __future__ import annotations

import paddle
from paddle import _C_ops, _legacy_C_ops
from paddle.base.data_feeder import check_variable_and_dtype
from paddle.base.framework import _create_tensor, in_pir_mode
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_mode


def accuracy(
    input: paddle.Tensor,
    label: paddle.Tensor,
    k: int = 1,
    correct: paddle.Tensor | None = None,
    total: paddle.Tensor | None = None,
    name: str | None = None,
) -> paddle.Tensor:
    """Compute top-k accuracy using the C++ accuracy op.

    Args:
        input: Predictions with shape [batch_size, num_classes].
        label: Ground truth labels with shape [batch_size, 1].
        k: Number of top predictions to consider. Default: 1.
        correct: Optional pre-allocated tensor for correct count.
        total: Optional pre-allocated tensor for total count.
        name: Optional name for the operation.

    Returns:
        Tensor containing the accuracy value.
    """
    if label.dtype == paddle.int32:
        label = paddle.cast(label, paddle.int64)
    if in_dynamic_mode():
        if correct is None:
            correct = _create_tensor(dtype="int32")
        if total is None:
            total = _create_tensor(dtype="int32")

        topk_out, topk_indices = paddle.topk(input, k=k)
        _acc, _, _ = _legacy_C_ops.accuracy(
            topk_out, topk_indices, label, correct, total
        )

        return _acc
    elif in_pir_mode():
        topk_out, topk_indices = paddle.topk(input, k=k)
        _acc, _, _ = _C_ops.accuracy(topk_out, topk_indices, label)
        return _acc

    helper = LayerHelper("accuracy", **locals())
    check_variable_and_dtype(
        input, 'input', ['float16', 'uint16', 'float32', 'float64'], 'accuracy'
    )
    topk_out, topk_indices = paddle.topk(input, k=k)
    acc_out = helper.create_variable_for_type_inference(dtype="float32")
    if correct is None:
        correct = helper.create_variable_for_type_inference(dtype="int32")
    if total is None:
        total = helper.create_variable_for_type_inference(dtype="int32")
    helper.append_op(
        type="accuracy",
        inputs={"Out": [topk_out], "Indices": [topk_indices], "Label": [label]},
        outputs={
            "Accuracy": [acc_out],
            "Correct": [correct],
            "Total": [total],
        },
    )
    return acc_out


class Accuracy:
    """Backward-compatible ``paddle.metric.Accuracy`` for code that does not
    pass ``task=``.  Delegates to :class:`MulticlassAccuracy`."""

    def __new__(cls, topk=(1,), name=None, **kwargs):
        from paddle.metric.classification import MulticlassAccuracy

        return MulticlassAccuracy(top_k=topk[0] if topk else 1, name=name, **kwargs)


class Auc:
    """Backward-compatible ``paddle.metric.Auc`` for code that uses the old
    ``curve='ROC'`` / ``num_thresholds`` API.  Delegates to
    :class:`BinaryAUROC`."""

    def __new__(cls, curve='ROC', num_thresholds=4095, name='auc', **kwargs):
        from paddle.metric.classification import BinaryAUROC

        return BinaryAUROC(
            thresholds=num_thresholds, name=name, **kwargs
        )
