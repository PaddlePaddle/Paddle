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
from ..framework import (
    Variable,
    _non_static_mode,
    _varbase_creator,
    _in_legacy_dygraph,
    in_dygraph_mode,
)
from .. import core
from ..param_attr import ParamAttr
from . import nn
from . import tensor
from ..data_feeder import check_variable_and_dtype
from paddle import _C_ops, _legacy_C_ops

__all__ = ['auc']


def auc(
    input,
    label,
    curve='ROC',
    num_thresholds=2**12 - 1,
    topk=1,
    slide_steps=1,
    ins_tag_weight=None,
):
    """
    **Area Under the Curve (AUC) Layer**

    This implementation computes the AUC according to forward output and label.
    It is used very widely in binary classification evaluation.

    Note: If input label contains values other than 0 and 1, it will be cast
    to `bool`. Find the relevant definitions `here <https://en.wikipedia.org\
    /wiki/Receiver_operating_characteristic#Area_under_the_curve>`_.

    There are two types of possible curves:

        1. ROC: Receiver operating characteristic;
        2. PR: Precision Recall

    Args:
        input(Tensor): A floating-point 2D Tensor, values are in the range
                         [0, 1]. Each row is sorted in descending order. This
                         input should be the output of topk. Typically, this
                         Tensor indicates the probability of each label.
                         A Tensor with type float32,float64.
        label(Tensor): A 2D int Tensor indicating the label of the training
                         data. The height is batch size and width is always 1.
                         A Tensor with type int32,int64.
        curve(str): Curve type, can be 'ROC' or 'PR'. Default 'ROC'.
        num_thresholds(int): The number of thresholds to use when discretizing
                             the roc curve. Default 4095.
        topk(int): only topk number of prediction output will be used for auc.
        slide_steps: when calc batch auc, we can not only use step currently but the previous steps can be used. slide_steps=1 means use the current step, slide_steps=3 means use current step and the previous second steps, slide_steps=0 use all of the steps.
        ins_tag_weight(Tensor): A 2D int Tensor indicating the data's tag weight, 1 means real data, 0 means fake data. Default None, and it will be assigned to a tensor of value 1.
                         A Tensor with type float32,float64.

    Returns:
        Tensor: A tuple representing the current AUC.
        The return tuple is auc_out, batch_auc_out, [
        batch_stat_pos, batch_stat_neg, stat_pos, stat_neg ]
        Data type is Tensor, supporting float32, float64.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np
            paddle.enable_static()

            data = paddle.static.data(name="input", shape=[-1, 32,32], dtype="float32")
            label = paddle.static.data(name="label", shape=[-1], dtype="int")
            fc_out = paddle.static.nn.fc(x=data, size=2)
            predict = paddle.nn.functional.softmax(x=fc_out)
            result=paddle.static.auc(input=predict, label=label)

            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)

            exe.run(paddle.static.default_startup_program())
            x = np.random.rand(3,32,32).astype("float32")
            y = np.array([1,0,1])
            output= exe.run(feed={"input": x,"label": y},
                             fetch_list=[result[0]])
            print(output)

            #you can learn the usage of ins_tag_weight by the following code.
            '''
            import paddle
            import numpy as np
            paddle.enable_static()

            data = paddle.static.data(name="input", shape=[-1, 32,32], dtype="float32")
            label = paddle.static.data(name="label", shape=[-1], dtype="int")
            ins_tag_weight = paddle.static.data(name='ins_tag', shape=[-1,16], lod_level=0, dtype='float64')
            fc_out = paddle.static.nn.fc(x=data, size=2)
            predict = paddle.nn.functional.softmax(x=fc_out)
            result=paddle.static.auc(input=predict, label=label, ins_tag_weight=ins_tag_weight)

            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)

            exe.run(paddle.static.default_startup_program())
            x = np.random.rand(3,32,32).astype("float32")
            y = np.array([1,0,1])
            z = np.array([1,0,1])
            output= exe.run(feed={"input": x,"label": y, "ins_tag_weight":z},
                             fetch_list=[result[0]])
            print(output)
            '''

    """
    helper = LayerHelper("auc", **locals())

    if ins_tag_weight is None:
        ins_tag_weight = tensor.fill_constant(
            shape=[1, 1], dtype="float32", value=1.0
        )
    check_variable_and_dtype(input, 'input', ['float32', 'float64'], 'auc')
    check_variable_and_dtype(label, 'label', ['int32', 'int64'], 'auc')
    check_variable_and_dtype(
        ins_tag_weight, 'ins_tag_weight', ['float32', 'float64'], 'auc'
    )
    auc_out = helper.create_variable_for_type_inference(dtype="float64")
    batch_auc_out = helper.create_variable_for_type_inference(dtype="float64")
    # make tp, tn, fp, fn persistable, so that can accumulate all batches.

    # for batch auc
    # we create slide_step+1 buckets, the first slide_steps buckets store
    # historical batch-level values, and the last bucket stores the sum values of
    # previous slide_step buckets.
    # The index of bucket that the newest batch will use is determined by batch_id mod slide_steps,
    # and batch_id is store in the last posision of following variable
    batch_stat_pos = helper.create_global_variable(
        persistable=True,
        dtype='int64',
        shape=[(1 + slide_steps) * (num_thresholds + 1) + 1],
    )
    batch_stat_neg = helper.create_global_variable(
        persistable=True,
        dtype='int64',
        shape=[(1 + slide_steps) * (num_thresholds + 1) + 1],
    )

    # for global auc
    # Needn't maintain the batch id
    stat_pos = helper.create_global_variable(
        persistable=True, dtype='int64', shape=[1, num_thresholds + 1]
    )
    stat_neg = helper.create_global_variable(
        persistable=True, dtype='int64', shape=[1, num_thresholds + 1]
    )

    for var in [batch_stat_pos, batch_stat_neg, stat_pos, stat_neg]:
        helper.set_variable_initializer(
            var, Constant(value=0.0, force_cpu=False)
        )

    # "InsTagWeight": [ins_tag_weight]
    # Batch AUC
    helper.append_op(
        type="auc",
        inputs={
            "Predict": [input],
            "Label": [label],
            "StatPos": [batch_stat_pos],
            "StatNeg": [batch_stat_neg],
        },
        attrs={
            "curve": curve,
            "num_thresholds": num_thresholds,
            "slide_steps": slide_steps,
        },
        outputs={
            "AUC": [batch_auc_out],
            "StatPosOut": [batch_stat_pos],
            "StatNegOut": [batch_stat_neg],
        },
    )
    # Global AUC
    helper.append_op(
        type="auc",
        inputs={
            "Predict": [input],
            "Label": [label],
            "StatPos": [stat_pos],
            "StatNeg": [stat_neg],
        },
        attrs={
            "curve": curve,
            "num_thresholds": num_thresholds,
            "slide_steps": 0,
        },
        outputs={
            "AUC": [auc_out],
            "StatPosOut": [stat_pos],
            "StatNegOut": [stat_neg],
        },
    )
    return (
        auc_out,
        batch_auc_out,
        [batch_stat_pos, batch_stat_neg, stat_pos, stat_neg],
    )
