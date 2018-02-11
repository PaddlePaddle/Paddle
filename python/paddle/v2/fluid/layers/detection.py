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
All layers just related to the detection neural network.
"""

from ..layer_helper import LayerHelper
from ..param_attr import ParamAttr
from ..framework import Variable
from layer_function_generator import autodoc
from tensor import concat
from ops import reshape
from ..nets import img_conv_with_bn
from nn import transpose
import math

__all__ = [
    'detection_output',
    'multi_box_head',
]


def detection_output(scores,
                     loc,
                     prior_box,
                     prior_box_var,
                     background_label=0,
                     nms_threshold=0.3,
                     nms_top_k=400,
                     keep_top_k=200,
                     score_threshold=0.01,
                     nms_eta=1.0):
    """
    **Detection Output Layer**

    This layer applies the NMS to the output of network and computes the 
    predict bounding box location. The output's shape of this layer could
    be zero if there is no valid bounding box.

    Args:
        scores(Variable): A 3-D Tensor with shape [N, C, M] represents the
            predicted confidence predictions. N is the batch size, C is the
            class number, M is number of bounding boxes. For each category
            there are total M scores which corresponding M bounding boxes.
        loc(Variable): A 3-D Tensor with shape [N, M, 4] represents the
            predicted locations of M bounding bboxes. N is the batch size,
            and each bounding box has four coordinate values and the layout
            is [xmin, ymin, xmax, ymax].
        prior_box(Variable): A 2-D Tensor with shape [M, 4] holds M boxes,
            each box is represented as [xmin, ymin, xmax, ymax],
            [xmin, ymin] is the left top coordinate of the anchor box,
            if the input is image feature map, they are close to the origin
            of the coordinate system. [xmax, ymax] is the right bottom
            coordinate of the anchor box.
        prior_box_var(Variable): A 2-D Tensor with shape [M, 4] holds M group
            of variance.
        background_label(float): The index of background label,
            the background label will be ignored. If set to -1, then all
            categories will be considered.
        nms_threshold(float): The threshold to be used in NMS.
        nms_top_k(int): Maximum number of detections to be kept according
            to the confidences aftern the filtering detections based on
            score_threshold.
        keep_top_k(int): Number of total bboxes to be kept per image after
            NMS step. -1 means keeping all bboxes after NMS step.
        score_threshold(float): Threshold to filter out bounding boxes with
            low confidence score. If not provided, consider all boxes.
        nms_eta(float): The parameter for adaptive NMS.

    Returns:
        The detected bounding boxes which are a Tensor.

    Examples:
        .. code-block:: python

        pb = layers.data(name='prior_box', shape=[10, 4],
                         append_batch_size=False, dtype='float32')
        pbv = layers.data(name='prior_box_var', shape=[10, 4],
                          append_batch_size=False, dtype='float32')
        loc = layers.data(name='target_box', shape=[21, 4],
                          append_batch_size=False, dtype='float32')
        scores = layers.data(name='scores', shape=[2, 21, 10],
                          append_batch_size=False, dtype='float32')
        nmsed_outs = fluid.layers.detection_output(scores=scores,
                                       loc=loc,
                                       prior_box=pb,
                                       prior_box_var=pbv)
    """

    helper = LayerHelper("detection_output", **locals())
    decoded_box = helper.create_tmp_variable(dtype=loc.dtype)
    helper.append_op(
        type="box_coder",
        inputs={
            'PriorBox': prior_box,
            'PriorBoxVar': prior_box_var,
            'TargetBox': loc
        },
        outputs={'OutputBox': decoded_box},
        attrs={'code_type': 'decode_center_size'})
    nmsed_outs = helper.create_tmp_variable(dtype=decoded_box.dtype)

    helper.append_op(
        type="multiclass_nms",
        inputs={'Scores': scores,
                'BBoxes': decoded_box},
        outputs={'Out': nmsed_outs},
        attrs={
            'background_label': 0,
            'nms_threshold': nms_threshold,
            'nms_top_k': nms_top_k,
            'keep_top_k': keep_top_k,
            'score_threshold': score_threshold,
            'nms_eta': 1.0
        })
    return nmsed_outs


def multi_box_head(inputs,
                   num_classes,
                   min_sizes=None,
                   max_sizes=None,
                   min_ratio=None,
                   max_ratio=None,
                   aspect_ratios=None,
                   flip=False,
                   share_location=True,
                   kernel_size=1,
                   pad=1,
                   stride=1,
                   use_batchnorm=False,
                   base_size=None,
                   name=None):
    """
    **Multi Box Head**

    input many Variable, and return mbox_loc, mbox_conf

    Args:
       inputs(list): The list of input Variables, the format
            of all Variables is NCHW.
       num_classes(int): The number of calss.
       min_sizes(list, optional, default=None): The length of
            min_size is used to compute the the number of prior box.
            If the min_size is None, it will be computed according
            to min_ratio and max_ratio.
       max_sizes(list, optional, default=None): The length of max_size
            is used to compute the the number of prior box.
       min_ratio(int): If the min_sizes is None, min_ratio and min_ratio
            will be used to compute the min_sizes and max_sizes.
       max_ratio(int): If the min_sizes is None, min_ratio and min_ratio
            will be used to compute the min_sizes and max_sizes.
       aspect_ratios(list): The number of the aspect ratios is used to
            compute the number of prior box.
       base_size(int): the base_size is used to get min_size
            and max_size according to min_ratio and max_ratio.
       flip(bool, optional, default=False): Whether to flip
            aspect ratios.
       name(str, optional, None): Name of the prior box layer.

    Returns:

        mbox_loc(Variable): the output prior boxes of PriorBoxOp. The layout is
             [num_priors, 4]. num_priors is the total box count of each
              position of inputs.
        mbox_conf(Variable): the expanded variances of PriorBoxOp. The layout
             is [num_priors, 4]. num_priors is the total box count of each
             position of inputs

    Examples:
        .. code-block:: python


    """

    assert isinstance(inputs, list), 'inputs should be a list.'

    if min_sizes is not None:
        assert len(inputs) == len(min_sizes)

    if max_sizes is not None:
        assert len(inputs) == len(max_sizes)

    if min_sizes is None:
        # if min_sizes is None, min_sizes and max_sizes
        #  will be set according to max_ratio and min_ratio
        assert max_ratio is not None and min_ratio is not None
        min_sizes = []
        max_sizes = []
        num_layer = len(inputs)
        step = int(math.floor(((max_ratio - min_ratio)) / (num_layer - 2)))
        for ratio in xrange(min_ratio, max_ratio + 1, step):
            min_sizes.append(base_size * ratio / 100.)
            max_sizes.append(base_size * (ratio + step) / 100.)
        min_sizes = [base_size * .10] + min_sizes
        max_sizes = [base_size * .20] + max_sizes

    if aspect_ratios is not None:
        assert len(inputs) == len(aspect_ratios)

    mbox_locs = []
    mbox_confs = []
    for i, input in enumerate(inputs):
        min_size = min_sizes[i]
        if type(min_size) is not list:
            min_size = [min_size]

        max_size = []
        if max_sizes is not None:
            max_size = max_sizes[i]
            if type(max_size) is not list:
                max_size = [max_size]
            if max_size:
                assert len(max_size) == len(
                    min_size), "max_size and min_size should have same length."

        aspect_ratio = []
        if aspect_ratios is not None:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]

        num_priors_per_location = 0
        if max_sizes is not None:
            num_priors_per_location = len(min_size) + len(aspect_ratio) * len(
                min_size) + len(max_size)
        else:
            num_priors_per_location = len(min_size) + len(aspect_ratio) * len(
                min_size)
        if flip:
            num_priors_per_location += len(aspect_ratio) * len(min_size)

        # mbox_loc
        num_loc_output = num_priors_per_location * 4
        if share_location:
            num_loc_output *= num_classes

        mbox_loc = img_conv_with_bn(
            input=input,
            conv_num_filter=num_loc_output,
            conv_padding=pad,
            conv_stride=stride,
            conv_filter_size=kernel_size,
            conv_with_batchnorm=use_batchnorm)
        mbox_loc = transpose(mbox_loc, perm=[0, 2, 3, 1])
        mbox_locs.append(mbox_loc)

        #    get the number of prior box
        num_conf_output = num_priors_per_location * num_classes
        conf_loc = img_conv_with_bn(
            input=input,
            conv_num_filter=num_conf_output,
            conv_padding=pad,
            conv_stride=stride,
            conv_filter_size=kernel_size,
            conv_with_batchnorm=use_batchnorm)
        conf_loc = transpose(conf_loc, perm=[0, 2, 3, 1])
        mbox_confs.append(conf_loc)

    return mbox_locs, mbox_confs
