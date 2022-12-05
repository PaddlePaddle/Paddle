#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
All layers just related to the detection neural network.
"""

import paddle

from .layer_function_generator import templatedoc
from ..layer_helper import LayerHelper
from ..framework import Variable, _non_static_mode, static_only, in_dygraph_mode
from .. import core
from paddle.fluid.layers import softmax_with_cross_entropy
from . import tensor
from . import nn
from ..data_feeder import check_variable_and_dtype, check_type, check_dtype
import math
import numpy as np
from functools import reduce
from ..data_feeder import (
    convert_dtype,
    check_variable_and_dtype,
    check_type,
    check_dtype,
)
from paddle.utils import deprecated
from paddle import _C_ops, _legacy_C_ops
from ..framework import in_dygraph_mode

__all__ = [
    'bipartite_match',
    'target_assign',
    'detection_output',
    'ssd_loss',
    'rpn_target_assign',
    'retinanet_target_assign',
    'iou_similarity',
    'box_coder',
    'polygon_box_transform',
    'yolov3_loss',
    'yolo_box',
]


def retinanet_target_assign(
    bbox_pred,
    cls_logits,
    anchor_box,
    anchor_var,
    gt_boxes,
    gt_labels,
    is_crowd,
    im_info,
    num_classes=1,
    positive_overlap=0.5,
    negative_overlap=0.4,
):
    r"""
    **Target Assign Layer for the detector RetinaNet.**

    This OP finds out positive and negative samples from all anchors
    for training the detector `RetinaNet <https://arxiv.org/abs/1708.02002>`_ ,
    and assigns target labels for classification along with target locations for
    regression to each sample, then takes out the part belonging to positive and
    negative samples from category prediction( :attr:`cls_logits`) and location
    prediction( :attr:`bbox_pred`) which belong to all anchors.

    The searching principles for positive and negative samples are as followed:

    1. Anchors are assigned to ground-truth boxes when it has the highest IoU
    overlap with a ground-truth box.

    2. Anchors are assigned to ground-truth boxes when it has an IoU overlap
    higher than :attr:`positive_overlap` with any ground-truth box.

    3. Anchors are assigned to background when its IoU overlap is lower than
    :attr:`negative_overlap` for all ground-truth boxes.

    4. Anchors which do not meet the above conditions do not participate in
    the training process.

    Retinanet predicts a :math:`C`-vector for classification and a 4-vector for box
    regression for each anchor, hence the target label for each positive(or negative)
    sample is a :math:`C`-vector and the target locations for each positive sample
    is a 4-vector. As for a positive sample, if the category of its assigned
    ground-truth box is class :math:`i`, the corresponding entry in its length
    :math:`C` label vector is set to 1 and all other entries is set to 0, its box
    regression targets are computed as the offset between itself and its assigned
    ground-truth box. As for a negative sample, all entries in its length :math:`C`
    label vector are set to 0 and box regression targets are omitted because
    negative samples do not participate in the training process of location
    regression.

    After the assignment, the part belonging to positive and negative samples is
    taken out from category prediction( :attr:`cls_logits` ), and the part
    belonging to positive samples is taken out from location
    prediction( :attr:`bbox_pred` ).

    Args:
        bbox_pred(Variable): A 3-D Tensor with shape :math:`[N, M, 4]` represents
            the predicted locations of all anchors. :math:`N` is the batch size( the
            number of images in a mini-batch), :math:`M` is the number of all anchors
            of one image, and each anchor has 4 coordinate values. The data type of
            :attr:`bbox_pred` is float32 or float64.
        cls_logits(Variable): A 3-D Tensor with shape :math:`[N, M, C]` represents
            the predicted categories of all anchors. :math:`N` is the batch size,
            :math:`M` is the number of all anchors of one image, and :math:`C` is
            the number of categories (**Notice: excluding background**). The data type
            of :attr:`cls_logits` is float32 or float64.
        anchor_box(Variable): A 2-D Tensor with shape :math:`[M, 4]` represents
            the locations of all anchors. :math:`M` is the number of all anchors of
            one image, each anchor is represented as :math:`[xmin, ymin, xmax, ymax]`,
            :math:`[xmin, ymin]` is the left top coordinate of the anchor box,
            :math:`[xmax, ymax]` is the right bottom coordinate of the anchor box.
            The data type of :attr:`anchor_box` is float32 or float64.
        anchor_var(Variable): A 2-D Tensor with shape :math:`[M,4]` represents the expanded
            factors of anchor locations used in loss function. :math:`M` is number of
            all anchors of one image, each anchor possesses a 4-vector expanded factor.
            The data type of :attr:`anchor_var` is float32 or float64.
        gt_boxes(Variable): A 1-level 2-D LoDTensor with shape :math:`[G, 4]` represents
            locations of all ground-truth boxes. :math:`G` is the total number of
            all ground-truth boxes in a mini-batch, and each ground-truth box has 4
            coordinate values. The data type of :attr:`gt_boxes` is float32 or
            float64.
        gt_labels(variable): A 1-level 2-D LoDTensor with shape :math:`[G, 1]` represents
            categories of all ground-truth boxes, and the values are in the range of
            :math:`[1, C]`. :math:`G` is the total number of all ground-truth boxes
            in a mini-batch, and each ground-truth box has one category. The data type
            of :attr:`gt_labels` is int32.
        is_crowd(Variable): A 1-level 1-D LoDTensor with shape :math:`[G]` which
            indicates whether a ground-truth box is a crowd. If the value is 1, the
            corresponding box is a crowd, it is ignored during training. :math:`G` is
            the total number of all ground-truth boxes in a mini-batch. The data type
            of :attr:`is_crowd` is int32.
        im_info(Variable): A 2-D Tensor with shape [N, 3] represents the size
            information of input images. :math:`N` is the batch size, the size
            information of each image is a 3-vector which are the height and width
            of the network input along with the factor scaling the origin image to
            the network input. The data type of :attr:`im_info` is float32.
        num_classes(int32): The number of categories for classification, the default
            value is 1.
        positive_overlap(float32): Minimum overlap required between an anchor
            and ground-truth box for the anchor to be a positive sample, the default
            value is 0.5.
        negative_overlap(float32): Maximum overlap allowed between an anchor
            and ground-truth box for the anchor to be a negative sample, the default
            value is 0.4. :attr:`negative_overlap` should be less than or equal to
            :attr:`positive_overlap`, if not, the actual value of
            :attr:`positive_overlap` is :attr:`negative_overlap`.

    Returns:
        A tuple with 6 Variables:

        **predict_scores** (Variable): A 2-D Tensor with shape :math:`[F+B, C]` represents
        category prediction belonging to positive and negative samples. :math:`F`
        is the number of positive samples in a mini-batch, :math:`B` is the number
        of negative samples, and :math:`C` is the number of categories
        (**Notice: excluding background**). The data type of :attr:`predict_scores`
        is float32 or float64.

        **predict_location** (Variable): A 2-D Tensor with shape :math:`[F, 4]` represents
        location prediction belonging to positive samples. :math:`F` is the number
        of positive samples. :math:`F` is the number of positive samples, and each
        sample has 4 coordinate values. The data type of :attr:`predict_location`
        is float32 or float64.

        **target_label** (Variable): A 2-D Tensor with shape :math:`[F+B, 1]` represents
        target labels for classification belonging to positive and negative
        samples. :math:`F` is the number of positive samples, :math:`B` is the
        number of negative, and each sample has one target category. The data type
        of :attr:`target_label` is int32.

        **target_bbox** (Variable): A 2-D Tensor with shape :math:`[F, 4]` represents
        target locations for box regression belonging to positive samples.
        :math:`F` is the number of positive samples, and each sample has 4
        coordinate values. The data type of :attr:`target_bbox` is float32 or
        float64.

        **bbox_inside_weight** (Variable): A 2-D Tensor with shape :math:`[F, 4]`
        represents whether a positive sample is fake positive, if a positive
        sample is false positive, the corresponding entries in
        :attr:`bbox_inside_weight` are set 0, otherwise 1. :math:`F` is the number
        of total positive samples in a mini-batch, and each sample has 4
        coordinate values. The data type of :attr:`bbox_inside_weight` is float32
        or float64.

        **fg_num** (Variable): A 2-D Tensor with shape :math:`[N, 1]` represents the number
        of positive samples. :math:`N` is the batch size. **Notice: The number
        of positive samples is used as the denominator of later loss function,
        to avoid the condition that the denominator is zero, this OP has added 1
        to the actual number of positive samples of each image.** The data type of
        :attr:`fg_num` is int32.

    Examples:
        .. code-block:: python
          import paddle
          import paddle.fluid as fluid
          paddle.enable_static()

          bbox_pred = fluid.data(name='bbox_pred', shape=[1, 100, 4],
                            dtype='float32')
          cls_logits = fluid.data(name='cls_logits', shape=[1, 100, 10],
                            dtype='float32')
          anchor_box = fluid.data(name='anchor_box', shape=[100, 4],
                            dtype='float32')
          anchor_var = fluid.data(name='anchor_var', shape=[100, 4],
                            dtype='float32')
          gt_boxes = fluid.data(name='gt_boxes', shape=[10, 4],
                            dtype='float32')
          gt_labels = fluid.data(name='gt_labels', shape=[10, 1],
                            dtype='int32')
          is_crowd = fluid.data(name='is_crowd', shape=[1],
                            dtype='int32')
          im_info = fluid.data(name='im_info', shape=[1, 3],
                            dtype='float32')
          score_pred, loc_pred, score_target, loc_target, bbox_inside_weight, fg_num = fluid.layers.retinanet_target_assign(bbox_pred, cls_logits, anchor_box,
                anchor_var, gt_boxes, gt_labels, is_crowd, im_info, 10)

    """

    check_variable_and_dtype(
        bbox_pred,
        'bbox_pred',
        ['float32', 'float64'],
        'retinanet_target_assign',
    )
    check_variable_and_dtype(
        cls_logits,
        'cls_logits',
        ['float32', 'float64'],
        'retinanet_target_assign',
    )
    check_variable_and_dtype(
        anchor_box,
        'anchor_box',
        ['float32', 'float64'],
        'retinanet_target_assign',
    )
    check_variable_and_dtype(
        anchor_var,
        'anchor_var',
        ['float32', 'float64'],
        'retinanet_target_assign',
    )
    check_variable_and_dtype(
        gt_boxes, 'gt_boxes', ['float32', 'float64'], 'retinanet_target_assign'
    )
    check_variable_and_dtype(
        gt_labels, 'gt_labels', ['int32'], 'retinanet_target_assign'
    )
    check_variable_and_dtype(
        is_crowd, 'is_crowd', ['int32'], 'retinanet_target_assign'
    )
    check_variable_and_dtype(
        im_info, 'im_info', ['float32', 'float64'], 'retinanet_target_assign'
    )

    helper = LayerHelper('retinanet_target_assign', **locals())
    # Assign target label to anchors
    loc_index = helper.create_variable_for_type_inference(dtype='int32')
    score_index = helper.create_variable_for_type_inference(dtype='int32')
    target_label = helper.create_variable_for_type_inference(dtype='int32')
    target_bbox = helper.create_variable_for_type_inference(
        dtype=anchor_box.dtype
    )
    bbox_inside_weight = helper.create_variable_for_type_inference(
        dtype=anchor_box.dtype
    )
    fg_num = helper.create_variable_for_type_inference(dtype='int32')
    helper.append_op(
        type="retinanet_target_assign",
        inputs={
            'Anchor': anchor_box,
            'GtBoxes': gt_boxes,
            'GtLabels': gt_labels,
            'IsCrowd': is_crowd,
            'ImInfo': im_info,
        },
        outputs={
            'LocationIndex': loc_index,
            'ScoreIndex': score_index,
            'TargetLabel': target_label,
            'TargetBBox': target_bbox,
            'BBoxInsideWeight': bbox_inside_weight,
            'ForegroundNumber': fg_num,
        },
        attrs={
            'positive_overlap': positive_overlap,
            'negative_overlap': negative_overlap,
        },
    )

    loc_index.stop_gradient = True
    score_index.stop_gradient = True
    target_label.stop_gradient = True
    target_bbox.stop_gradient = True
    bbox_inside_weight.stop_gradient = True
    fg_num.stop_gradient = True

    cls_logits = paddle.reshape(x=cls_logits, shape=(-1, num_classes))
    bbox_pred = paddle.reshape(x=bbox_pred, shape=(-1, 4))
    predicted_cls_logits = paddle.gather(cls_logits, score_index)
    predicted_bbox_pred = paddle.gather(bbox_pred, loc_index)

    return (
        predicted_cls_logits,
        predicted_bbox_pred,
        target_label,
        target_bbox,
        bbox_inside_weight,
        fg_num,
    )


def rpn_target_assign(
    bbox_pred,
    cls_logits,
    anchor_box,
    anchor_var,
    gt_boxes,
    is_crowd,
    im_info,
    rpn_batch_size_per_im=256,
    rpn_straddle_thresh=0.0,
    rpn_fg_fraction=0.5,
    rpn_positive_overlap=0.7,
    rpn_negative_overlap=0.3,
    use_random=True,
):
    """
    **Target Assign Layer for region proposal network (RPN) in Faster-RCNN detection.**

    This layer can be, for given the  Intersection-over-Union (IoU) overlap
    between anchors and ground truth boxes, to assign classification and
    regression targets to each each anchor, these target labels are used for
    train RPN. The classification targets is a binary class label (of being
    an object or not). Following the paper of Faster-RCNN, the positive labels
    are two kinds of anchors: (i) the anchor/anchors with the highest IoU
    overlap with a ground-truth box, or (ii) an anchor that has an IoU overlap
    higher than rpn_positive_overlap(0.7) with any ground-truth box. Note
    that a single ground-truth box may assign positive labels to multiple
    anchors. A non-positive anchor is when its IoU ratio is lower than
    rpn_negative_overlap (0.3) for all ground-truth boxes. Anchors that are
    neither positive nor negative do not contribute to the training objective.
    The regression targets are the encoded ground-truth boxes associated with
    the positive anchors.

    Args:
        bbox_pred(Variable): A 3-D Tensor with shape [N, M, 4] represents the
            predicted locations of M bounding bboxes. N is the batch size,
            and each bounding box has four coordinate values and the layout
            is [xmin, ymin, xmax, ymax]. The data type can be float32 or float64.
        cls_logits(Variable): A 3-D Tensor with shape [N, M, 1] represents the
            predicted confidence predictions. N is the batch size, 1 is the
            frontground and background sigmoid, M is number of bounding boxes.
            The data type can be float32 or float64.
        anchor_box(Variable): A 2-D Tensor with shape [M, 4] holds M boxes,
            each box is represented as [xmin, ymin, xmax, ymax],
            [xmin, ymin] is the left top coordinate of the anchor box,
            if the input is image feature map, they are close to the origin
            of the coordinate system. [xmax, ymax] is the right bottom
            coordinate of the anchor box. The data type can be float32 or float64.
        anchor_var(Variable): A 2-D Tensor with shape [M,4] holds expanded
            variances of anchors. The data type can be float32 or float64.
        gt_boxes (Variable): The ground-truth bounding boxes (bboxes) are a 2D
            LoDTensor with shape [Ng, 4], Ng is the total number of ground-truth
            bboxes of mini-batch input. The data type can be float32 or float64.
        is_crowd (Variable): A 1-D LoDTensor which indicates groud-truth is crowd.
                             The data type must be int32.
        im_info (Variable): A 2-D LoDTensor with shape [N, 3]. N is the batch size,
        3 is the height, width and scale.
        rpn_batch_size_per_im(int): Total number of RPN examples per image.
                                    The data type must be int32.
        rpn_straddle_thresh(float): Remove RPN anchors that go outside the image
            by straddle_thresh pixels. The data type must be float32.
        rpn_fg_fraction(float): Target fraction of RoI minibatch that is labeled
            foreground (i.e. class > 0), 0-th class is background. The data type must be float32.
        rpn_positive_overlap(float): Minimum overlap required between an anchor
            and ground-truth box for the (anchor, gt box) pair to be a positive
            example. The data type must be float32.
        rpn_negative_overlap(float): Maximum overlap allowed between an anchor
            and ground-truth box for the (anchor, gt box) pair to be a negative
            examples. The data type must be float32.

    Returns:
        tuple:
        A tuple(predicted_scores, predicted_location, target_label,
        target_bbox, bbox_inside_weight) is returned. The predicted_scores
        and predicted_location is the predicted result of the RPN.
        The target_label and target_bbox is the ground truth,
        respectively. The predicted_location is a 2D Tensor with shape
        [F, 4], and the shape of target_bbox is same as the shape of
        the predicted_location, F is the number of the foreground
        anchors. The predicted_scores is a 2D Tensor with shape
        [F + B, 1], and the shape of target_label is same as the shape
        of the predicted_scores, B is the number of the background
        anchors, the F and B is depends on the input of this operator.
        Bbox_inside_weight represents whether the predicted loc is fake_fg
        or not and the shape is [F, 4].

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            bbox_pred = fluid.data(name='bbox_pred', shape=[None, 4], dtype='float32')
            cls_logits = fluid.data(name='cls_logits', shape=[None, 1], dtype='float32')
            anchor_box = fluid.data(name='anchor_box', shape=[None, 4], dtype='float32')
            anchor_var = fluid.data(name='anchor_var', shape=[None, 4], dtype='float32')
            gt_boxes = fluid.data(name='gt_boxes', shape=[None, 4], dtype='float32')
            is_crowd = fluid.data(name='is_crowd', shape=[None], dtype='float32')
            im_info = fluid.data(name='im_infoss', shape=[None, 3], dtype='float32')
            loc, score, loc_target, score_target, inside_weight = fluid.layers.rpn_target_assign(
                bbox_pred, cls_logits, anchor_box, anchor_var, gt_boxes, is_crowd, im_info)

    """

    helper = LayerHelper('rpn_target_assign', **locals())

    check_variable_and_dtype(
        bbox_pred, 'bbox_pred', ['float32', 'float64'], 'rpn_target_assign'
    )
    check_variable_and_dtype(
        cls_logits, 'cls_logits', ['float32', 'float64'], 'rpn_target_assign'
    )
    check_variable_and_dtype(
        anchor_box, 'anchor_box', ['float32', 'float64'], 'rpn_target_assign'
    )
    check_variable_and_dtype(
        anchor_var, 'anchor_var', ['float32', 'float64'], 'rpn_target_assign'
    )
    check_variable_and_dtype(
        gt_boxes, 'gt_boxes', ['float32', 'float64'], 'rpn_target_assign'
    )
    check_variable_and_dtype(
        is_crowd, 'is_crowd', ['int32'], 'rpn_target_assign'
    )
    check_variable_and_dtype(
        im_info, 'im_info', ['float32', 'float64'], 'rpn_target_assign'
    )

    # Assign target label to anchors
    loc_index = helper.create_variable_for_type_inference(dtype='int32')
    score_index = helper.create_variable_for_type_inference(dtype='int32')
    target_label = helper.create_variable_for_type_inference(dtype='int32')
    target_bbox = helper.create_variable_for_type_inference(
        dtype=anchor_box.dtype
    )
    bbox_inside_weight = helper.create_variable_for_type_inference(
        dtype=anchor_box.dtype
    )
    helper.append_op(
        type="rpn_target_assign",
        inputs={
            'Anchor': anchor_box,
            'GtBoxes': gt_boxes,
            'IsCrowd': is_crowd,
            'ImInfo': im_info,
        },
        outputs={
            'LocationIndex': loc_index,
            'ScoreIndex': score_index,
            'TargetLabel': target_label,
            'TargetBBox': target_bbox,
            'BBoxInsideWeight': bbox_inside_weight,
        },
        attrs={
            'rpn_batch_size_per_im': rpn_batch_size_per_im,
            'rpn_straddle_thresh': rpn_straddle_thresh,
            'rpn_positive_overlap': rpn_positive_overlap,
            'rpn_negative_overlap': rpn_negative_overlap,
            'rpn_fg_fraction': rpn_fg_fraction,
            'use_random': use_random,
        },
    )

    loc_index.stop_gradient = True
    score_index.stop_gradient = True
    target_label.stop_gradient = True
    target_bbox.stop_gradient = True
    bbox_inside_weight.stop_gradient = True

    cls_logits = paddle.reshape(x=cls_logits, shape=(-1, 1))
    bbox_pred = paddle.reshape(x=bbox_pred, shape=(-1, 4))
    predicted_cls_logits = paddle.gather(cls_logits, score_index)
    predicted_bbox_pred = paddle.gather(bbox_pred, loc_index)

    return (
        predicted_cls_logits,
        predicted_bbox_pred,
        target_label,
        target_bbox,
        bbox_inside_weight,
    )


def detection_output(
    loc,
    scores,
    prior_box,
    prior_box_var,
    background_label=0,
    nms_threshold=0.3,
    nms_top_k=400,
    keep_top_k=200,
    score_threshold=0.01,
    nms_eta=1.0,
    return_index=False,
):
    """

    Given the regression locations, classification confidences and prior boxes,
    calculate the detection outputs by performing following steps:

    1. Decode input bounding box predictions according to the prior boxes and
       regression locations.
    2. Get the final detection results by applying multi-class non maximum
       suppression (NMS).

    Please note, this operation doesn't clip the final output bounding boxes
    to the image window.

    Args:
        loc(Variable): A 3-D Tensor with shape [N, M, 4] represents the
            predicted locations of M bounding bboxes. Data type should be
            float32 or float64. N is the batch size,
            and each bounding box has four coordinate values and the layout
            is [xmin, ymin, xmax, ymax].
        scores(Variable): A 3-D Tensor with shape [N, M, C] represents the
            predicted confidence predictions. Data type should be float32
            or float64. N is the batch size, C is the
            class number, M is number of bounding boxes.
        prior_box(Variable): A 2-D Tensor with shape [M, 4] holds M boxes,
            each box is represented as [xmin, ymin, xmax, ymax]. Data type
            should be float32 or float64.
        prior_box_var(Variable): A 2-D Tensor with shape [M, 4] holds M group
            of variance. Data type should be float32 or float64.
        background_label(int): The index of background label,
            the background label will be ignored. If set to -1, then all
            categories will be considered. Default: 0.
        nms_threshold(float): The threshold to be used in NMS. Default: 0.3.
        nms_top_k(int): Maximum number of detections to be kept according
            to the confidences after filtering detections based on
            score_threshold and before NMS. Default: 400.
        keep_top_k(int): Number of total bboxes to be kept per image after
            NMS step. -1 means keeping all bboxes after NMS step. Default: 200.
        score_threshold(float): Threshold to filter out bounding boxes with
            low confidence score. If not provided, consider all boxes.
            Default: 0.01.
        nms_eta(float): The parameter for adaptive NMS. It works only when the
            value is less than 1.0. Default: 1.0.
        return_index(bool): Whether return selected index. Default: False

    Returns:

        A tuple with two Variables: (Out, Index) if return_index is True,
        otherwise, a tuple with one Variable(Out) is returned.

        Out (Variable): The detection outputs is a LoDTensor with shape [No, 6].
        Data type is the same as input (loc). Each row has six values:
        [label, confidence, xmin, ymin, xmax, ymax]. `No` is
        the total number of detections in this mini-batch. For each instance,
        the offsets in first dimension are called LoD, the offset number is
        N + 1, N is the batch size. The i-th image has `LoD[i + 1] - LoD[i]`
        detected results, if it is 0, the i-th image has no detected results.

        Index (Variable): Only return when return_index is True. A 2-D LoDTensor
        with shape [No, 1] represents the selected index which type is Integer.
        The index is the absolute value cross batches. No is the same number
        as Out. If the index is used to gather other attribute such as age,
        one needs to reshape the input(N, M, 1) to (N * M, 1) as first, where
        N is the batch size and M is the number of boxes.


    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle

            paddle.enable_static()

            pb = fluid.data(name='prior_box', shape=[10, 4], dtype='float32')
            pbv = fluid.data(name='prior_box_var', shape=[10, 4], dtype='float32')
            loc = fluid.data(name='target_box', shape=[2, 21, 4], dtype='float32')
            scores = fluid.data(name='scores', shape=[2, 21, 10], dtype='float32')
            nmsed_outs, index = fluid.layers.detection_output(scores=scores,
                                       loc=loc,
                                       prior_box=pb,
                                       prior_box_var=pbv,
                                       return_index=True)
    """
    helper = LayerHelper("detection_output", **locals())
    decoded_box = box_coder(
        prior_box=prior_box,
        prior_box_var=prior_box_var,
        target_box=loc,
        code_type='decode_center_size',
    )
    scores = paddle.nn.functional.softmax(scores)
    scores = paddle.transpose(scores, perm=[0, 2, 1])
    scores.stop_gradient = True
    nmsed_outs = helper.create_variable_for_type_inference(
        dtype=decoded_box.dtype
    )
    if return_index:
        index = helper.create_variable_for_type_inference(dtype='int')
        helper.append_op(
            type="multiclass_nms2",
            inputs={'Scores': scores, 'BBoxes': decoded_box},
            outputs={'Out': nmsed_outs, 'Index': index},
            attrs={
                'background_label': 0,
                'nms_threshold': nms_threshold,
                'nms_top_k': nms_top_k,
                'keep_top_k': keep_top_k,
                'score_threshold': score_threshold,
                'nms_eta': 1.0,
            },
        )
        index.stop_gradient = True
    else:
        helper.append_op(
            type="multiclass_nms",
            inputs={'Scores': scores, 'BBoxes': decoded_box},
            outputs={'Out': nmsed_outs},
            attrs={
                'background_label': 0,
                'nms_threshold': nms_threshold,
                'nms_top_k': nms_top_k,
                'keep_top_k': keep_top_k,
                'score_threshold': score_threshold,
                'nms_eta': 1.0,
            },
        )
    nmsed_outs.stop_gradient = True
    if return_index:
        return nmsed_outs, index
    return nmsed_outs


@templatedoc()
def iou_similarity(x, y, box_normalized=True, name=None):
    """
        :alias_main: paddle.nn.functional.iou_similarity
        :alias: paddle.nn.functional.iou_similarity,paddle.nn.functional.loss.iou_similarity
        :old_api: paddle.fluid.layers.iou_similarity

    ${comment}

    Args:
        x (Variable): ${x_comment}.The data type is float32 or float64.
        y (Variable): ${y_comment}.The data type is float32 or float64.
        box_normalized(bool): Whether treat the priorbox as a normalized box.
            Set true by default.
    Returns:
        Variable: ${out_comment}.The data type is same with x.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle.fluid as fluid

            use_gpu = False
            place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
            exe = fluid.Executor(place)

            x = fluid.data(name='x', shape=[None, 4], dtype='float32')
            y = fluid.data(name='y', shape=[None, 4], dtype='float32')
            iou = fluid.layers.iou_similarity(x=x, y=y)

            exe.run(fluid.default_startup_program())
            test_program = fluid.default_main_program().clone(for_test=True)

            [out_iou] = exe.run(test_program,
                    fetch_list=iou,
                    feed={'x': np.array([[0.5, 0.5, 2.0, 2.0],
                                         [0., 0., 1.0, 1.0]]).astype('float32'),
                          'y': np.array([[1.0, 1.0, 2.5, 2.5]]).astype('float32')})
            # out_iou is [[0.2857143],
            #             [0.       ]] with shape: [2, 1]
    """
    helper = LayerHelper("iou_similarity", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type="iou_similarity",
        inputs={"X": x, "Y": y},
        attrs={"box_normalized": box_normalized},
        outputs={"Out": out},
    )
    return out


@templatedoc()
def box_coder(
    prior_box,
    prior_box_var,
    target_box,
    code_type="encode_center_size",
    box_normalized=True,
    name=None,
    axis=0,
):
    r"""

    **Box Coder Layer**

    Encode/Decode the target bounding box with the priorbox information.

    The Encoding schema described below:

    .. math::

        ox = (tx - px) / pw / pxv

        oy = (ty - py) / ph / pyv

        ow = \log(\abs(tw / pw)) / pwv

        oh = \log(\abs(th / ph)) / phv

    The Decoding schema described below:

    .. math::

        ox = (pw * pxv * tx * + px) - tw / 2

        oy = (ph * pyv * ty * + py) - th / 2

        ow = \exp(pwv * tw) * pw + tw / 2

        oh = \exp(phv * th) * ph + th / 2

    where `tx`, `ty`, `tw`, `th` denote the target box's center coordinates,
    width and height respectively. Similarly, `px`, `py`, `pw`, `ph` denote
    the priorbox's (anchor) center coordinates, width and height. `pxv`,
    `pyv`, `pwv`, `phv` denote the variance of the priorbox and `ox`, `oy`,
    `ow`, `oh` denote the encoded/decoded coordinates, width and height.

    During Box Decoding, two modes for broadcast are supported. Say target
    box has shape [N, M, 4], and the shape of prior box can be [N, 4] or
    [M, 4]. Then prior box will broadcast to target box along the
    assigned axis.

    Args:
        prior_box(Variable): Box list prior_box is a 2-D Tensor with shape
            [M, 4] holds M boxes and data type is float32 or float64. Each box
            is represented as [xmin, ymin, xmax, ymax], [xmin, ymin] is the
            left top coordinate of the anchor box, if the input is image feature
            map, they are close to the origin of the coordinate system.
            [xmax, ymax] is the right bottom coordinate of the anchor box.
        prior_box_var(List|Variable|None): prior_box_var supports three types
            of input. One is variable with shape [M, 4] which holds M group and
            data type is float32 or float64. The second is list consist of
            4 elements shared by all boxes and data type is float32 or float64.
            Other is None and not involved in calculation.
        target_box(Variable): This input can be a 2-D LoDTensor with shape
            [N, 4] when code_type is 'encode_center_size'. This input also can
            be a 3-D Tensor with shape [N, M, 4] when code_type is
            'decode_center_size'. Each box is represented as
            [xmin, ymin, xmax, ymax]. The data type is float32 or float64.
            This tensor can contain LoD information to represent a batch of inputs.
        code_type(str): The code type used with the target box. It can be
            `encode_center_size` or `decode_center_size`. `encode_center_size`
            by default.
        box_normalized(bool): Whether treat the priorbox as a normalized box.
            Set true by default.
        name(str, optional): For detailed information, please refer
            to :ref:`api_guide_Name`. Usually name is no need to set and
            None by default.
        axis(int): Which axis in PriorBox to broadcast for box decode,
            for example, if axis is 0 and TargetBox has shape [N, M, 4] and
            PriorBox has shape [M, 4], then PriorBox will broadcast to [N, M, 4]
            for decoding. It is only valid when code type is
            `decode_center_size`. Set 0 by default.

    Returns:
        Variable:

        output_box(Variable): When code_type is 'encode_center_size', the
        output tensor of box_coder_op with shape [N, M, 4] representing the
        result of N target boxes encoded with M Prior boxes and variances.
        When code_type is 'decode_center_size', N represents the batch size
        and M represents the number of decoded boxes.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            # For encode
            prior_box_encode = fluid.data(name='prior_box_encode',
                                  shape=[512, 4],
                                  dtype='float32')
            target_box_encode = fluid.data(name='target_box_encode',
                                   shape=[81, 4],
                                   dtype='float32')
            output_encode = fluid.layers.box_coder(prior_box=prior_box_encode,
                                    prior_box_var=[0.1,0.1,0.2,0.2],
                                    target_box=target_box_encode,
                                    code_type="encode_center_size")
            # For decode
            prior_box_decode = fluid.data(name='prior_box_decode',
                                  shape=[512, 4],
                                  dtype='float32')
            target_box_decode = fluid.data(name='target_box_decode',
                                   shape=[512, 81, 4],
                                   dtype='float32')
            output_decode = fluid.layers.box_coder(prior_box=prior_box_decode,
                                    prior_box_var=[0.1,0.1,0.2,0.2],
                                    target_box=target_box_decode,
                                    code_type="decode_center_size",
                                    box_normalized=False,
                                    axis=1)
    """
    return paddle.vision.ops.box_coder(
        prior_box=prior_box,
        prior_box_var=prior_box_var,
        target_box=target_box,
        code_type=code_type,
        box_normalized=box_normalized,
        axis=axis,
        name=name,
    )


@templatedoc()
def polygon_box_transform(input, name=None):
    """
    ${comment}

    Args:
        input(Variable): The input with shape [batch_size, geometry_channels, height, width].
                         A Tensor with type float32, float64.
        name(str, Optional): For details, please refer to :ref:`api_guide_Name`.
                        Generally, no setting is required. Default: None.

    Returns:
        Variable: The output with the same shape as input. A Tensor with type float32, float64.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            input = fluid.data(name='input', shape=[4, 10, 5, 5], dtype='float32')
            out = fluid.layers.polygon_box_transform(input)
    """
    check_variable_and_dtype(
        input, "input", ['float32', 'float64'], 'polygon_box_transform'
    )
    helper = LayerHelper("polygon_box_transform", **locals())
    output = helper.create_variable_for_type_inference(dtype=input.dtype)

    helper.append_op(
        type="polygon_box_transform",
        inputs={"Input": input},
        attrs={},
        outputs={"Output": output},
    )
    return output


@deprecated(since="2.0.0", update_to="paddle.vision.ops.yolo_loss")
@templatedoc(op_type="yolov3_loss")
def yolov3_loss(
    x,
    gt_box,
    gt_label,
    anchors,
    anchor_mask,
    class_num,
    ignore_thresh,
    downsample_ratio,
    gt_score=None,
    use_label_smooth=True,
    name=None,
    scale_x_y=1.0,
):
    """

    ${comment}

    Args:
        x (Variable): ${x_comment}The data type is float32 or float64.
        gt_box (Variable): groud truth boxes, should be in shape of [N, B, 4],
                          in the third dimension, x, y, w, h should be stored.
                          x,y is the center coordinate of boxes, w, h are the
                          width and height, x, y, w, h should be divided by
                          input image height to scale to [0, 1].
                          N is the batch number and B is the max box number in
                          an image.The data type is float32 or float64.
        gt_label (Variable): class id of ground truth boxes, should be in shape
                            of [N, B].The data type is int32.
        anchors (list|tuple): ${anchors_comment}
        anchor_mask (list|tuple): ${anchor_mask_comment}
        class_num (int): ${class_num_comment}
        ignore_thresh (float): ${ignore_thresh_comment}
        downsample_ratio (int): ${downsample_ratio_comment}
        name (string): The default value is None.  Normally there is no need
                       for user to set this property.  For more information,
                       please refer to :ref:`api_guide_Name`
        gt_score (Variable): mixup score of ground truth boxes, should be in shape
                            of [N, B]. Default None.
        use_label_smooth (bool): ${use_label_smooth_comment}
        scale_x_y (float): ${scale_x_y_comment}

    Returns:
        Variable: A 1-D tensor with shape [N], the value of yolov3 loss

    Raises:
        TypeError: Input x of yolov3_loss must be Variable
        TypeError: Input gtbox of yolov3_loss must be Variable
        TypeError: Input gtlabel of yolov3_loss must be Variable
        TypeError: Input gtscore of yolov3_loss must be None or Variable
        TypeError: Attr anchors of yolov3_loss must be list or tuple
        TypeError: Attr class_num of yolov3_loss must be an integer
        TypeError: Attr ignore_thresh of yolov3_loss must be a float number
        TypeError: Attr use_label_smooth of yolov3_loss must be a bool value

    Examples:
      .. code-block:: python

          import paddle.fluid as fluid
          import paddle
          paddle.enable_static()
          x = fluid.data(name='x', shape=[None, 255, 13, 13], dtype='float32')
          gt_box = fluid.data(name='gt_box', shape=[None, 6, 4], dtype='float32')
          gt_label = fluid.data(name='gt_label', shape=[None, 6], dtype='int32')
          gt_score = fluid.data(name='gt_score', shape=[None, 6], dtype='float32')
          anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
          anchor_mask = [0, 1, 2]
          loss = fluid.layers.yolov3_loss(x=x, gt_box=gt_box, gt_label=gt_label,
                                          gt_score=gt_score, anchors=anchors,
                                          anchor_mask=anchor_mask, class_num=80,
                                          ignore_thresh=0.7, downsample_ratio=32)
    """

    if not isinstance(x, Variable):
        raise TypeError("Input x of yolov3_loss must be Variable")
    if not isinstance(gt_box, Variable):
        raise TypeError("Input gtbox of yolov3_loss must be Variable")
    if not isinstance(gt_label, Variable):
        raise TypeError("Input gtlabel of yolov3_loss must be Variable")
    if gt_score is not None and not isinstance(gt_score, Variable):
        raise TypeError("Input gtscore of yolov3_loss must be Variable")
    if not isinstance(anchors, list) and not isinstance(anchors, tuple):
        raise TypeError("Attr anchors of yolov3_loss must be list or tuple")
    if not isinstance(anchor_mask, list) and not isinstance(anchor_mask, tuple):
        raise TypeError("Attr anchor_mask of yolov3_loss must be list or tuple")
    if not isinstance(class_num, int):
        raise TypeError("Attr class_num of yolov3_loss must be an integer")
    if not isinstance(ignore_thresh, float):
        raise TypeError(
            "Attr ignore_thresh of yolov3_loss must be a float number"
        )
    if not isinstance(use_label_smooth, bool):
        raise TypeError(
            "Attr use_label_smooth of yolov3_loss must be a bool value"
        )

    if _non_static_mode():
        attrs = (
            "anchors",
            anchors,
            "anchor_mask",
            anchor_mask,
            "class_num",
            class_num,
            "ignore_thresh",
            ignore_thresh,
            "downsample_ratio",
            downsample_ratio,
            "use_label_smooth",
            use_label_smooth,
            "scale_x_y",
            scale_x_y,
        )
        loss, _, _ = _legacy_C_ops.yolov3_loss(
            x, gt_box, gt_label, gt_score, *attrs
        )
        return loss

    helper = LayerHelper('yolov3_loss', **locals())
    loss = helper.create_variable_for_type_inference(dtype=x.dtype)
    objectness_mask = helper.create_variable_for_type_inference(dtype='int32')
    gt_match_mask = helper.create_variable_for_type_inference(dtype='int32')

    inputs = {
        "X": x,
        "GTBox": gt_box,
        "GTLabel": gt_label,
    }
    if gt_score is not None:
        inputs["GTScore"] = gt_score

    attrs = {
        "anchors": anchors,
        "anchor_mask": anchor_mask,
        "class_num": class_num,
        "ignore_thresh": ignore_thresh,
        "downsample_ratio": downsample_ratio,
        "use_label_smooth": use_label_smooth,
        "scale_x_y": scale_x_y,
    }

    helper.append_op(
        type='yolov3_loss',
        inputs=inputs,
        outputs={
            'Loss': loss,
            'ObjectnessMask': objectness_mask,
            'GTMatchMask': gt_match_mask,
        },
        attrs=attrs,
    )
    return loss


@deprecated(since="2.0.0", update_to="paddle.vision.ops.yolo_box")
@templatedoc(op_type="yolo_box")
def yolo_box(
    x,
    img_size,
    anchors,
    class_num,
    conf_thresh,
    downsample_ratio,
    clip_bbox=True,
    name=None,
    scale_x_y=1.0,
    iou_aware=False,
    iou_aware_factor=0.5,
):
    """

    ${comment}

    Args:
        x (Variable): ${x_comment} The data type is float32 or float64.
        img_size (Variable): ${img_size_comment} The data type is int32.
        anchors (list|tuple): ${anchors_comment}
        class_num (int): ${class_num_comment}
        conf_thresh (float): ${conf_thresh_comment}
        downsample_ratio (int): ${downsample_ratio_comment}
        clip_bbox (bool): ${clip_bbox_comment}
        scale_x_y (float): ${scale_x_y_comment}
        name (string): The default value is None.  Normally there is no need
                       for user to set this property.  For more information,
                       please refer to :ref:`api_guide_Name`
        iou_aware (bool): ${iou_aware_comment}
        iou_aware_factor (float): ${iou_aware_factor_comment}

    Returns:
        Variable: A 3-D tensor with shape [N, M, 4], the coordinates of boxes,
        and a 3-D tensor with shape [N, M, :attr:`class_num`], the classification
        scores of boxes.

    Raises:
        TypeError: Input x of yolov_box must be Variable
        TypeError: Attr anchors of yolo box must be list or tuple
        TypeError: Attr class_num of yolo box must be an integer
        TypeError: Attr conf_thresh of yolo box must be a float number

    Examples:

    .. code-block:: python

        import paddle.fluid as fluid
        import paddle
        paddle.enable_static()
        x = fluid.data(name='x', shape=[None, 255, 13, 13], dtype='float32')
        img_size = fluid.data(name='img_size',shape=[None, 2],dtype='int64')
        anchors = [10, 13, 16, 30, 33, 23]
        boxes,scores = fluid.layers.yolo_box(x=x, img_size=img_size, class_num=80, anchors=anchors,
                                        conf_thresh=0.01, downsample_ratio=32)
    """
    helper = LayerHelper('yolo_box', **locals())

    if not isinstance(x, Variable):
        raise TypeError("Input x of yolo_box must be Variable")
    if not isinstance(img_size, Variable):
        raise TypeError("Input img_size of yolo_box must be Variable")
    if not isinstance(anchors, list) and not isinstance(anchors, tuple):
        raise TypeError("Attr anchors of yolo_box must be list or tuple")
    if not isinstance(class_num, int):
        raise TypeError("Attr class_num of yolo_box must be an integer")
    if not isinstance(conf_thresh, float):
        raise TypeError("Attr ignore_thresh of yolo_box must be a float number")

    boxes = helper.create_variable_for_type_inference(dtype=x.dtype)
    scores = helper.create_variable_for_type_inference(dtype=x.dtype)

    attrs = {
        "anchors": anchors,
        "class_num": class_num,
        "conf_thresh": conf_thresh,
        "downsample_ratio": downsample_ratio,
        "clip_bbox": clip_bbox,
        "scale_x_y": scale_x_y,
        "iou_aware": iou_aware,
        "iou_aware_factor": iou_aware_factor,
    }

    helper.append_op(
        type='yolo_box',
        inputs={
            "X": x,
            "ImgSize": img_size,
        },
        outputs={
            'Boxes': boxes,
            'Scores': scores,
        },
        attrs=attrs,
    )
    return boxes, scores


@templatedoc()
def detection_map(
    detect_res,
    label,
    class_num,
    background_label=0,
    overlap_threshold=0.3,
    evaluate_difficult=True,
    has_state=None,
    input_states=None,
    out_states=None,
    ap_version='integral',
):
    """
    ${comment}

    Args:
        detect_res: ${detect_res_comment}
        label:  ${label_comment}
        class_num: ${class_num_comment}
        background_label: ${background_label_comment}
        overlap_threshold: ${overlap_threshold_comment}
        evaluate_difficult: ${evaluate_difficult_comment}
        has_state: ${has_state_comment}
        input_states: (tuple|None) If not None, It contains 3 elements:
            (1) pos_count ${pos_count_comment}.
            (2) true_pos ${true_pos_comment}.
            (3) false_pos ${false_pos_comment}.
        out_states: (tuple|None) If not None, it contains 3 elements.
            (1) accum_pos_count ${accum_pos_count_comment}.
            (2) accum_true_pos ${accum_true_pos_comment}.
            (3) accum_false_pos ${accum_false_pos_comment}.
        ap_version: ${ap_type_comment}

    Returns:
        ${map_comment}


    Examples:
          .. code-block:: python

            import paddle.fluid as fluid
            from fluid.layers import detection
            detect_res = fluid.data(
                name='detect_res',
                shape=[10, 6],
                dtype='float32')
            label = fluid.data(
                name='label',
                shape=[10, 6],
                dtype='float32')

            map_out = detection.detection_map(detect_res, label, 21)
    """
    helper = LayerHelper("detection_map", **locals())

    def __create_var(type):
        return helper.create_variable_for_type_inference(dtype=type)

    map_out = __create_var('float32')
    accum_pos_count_out = (
        out_states[0] if out_states is not None else __create_var('int32')
    )
    accum_true_pos_out = (
        out_states[1] if out_states is not None else __create_var('float32')
    )
    accum_false_pos_out = (
        out_states[2] if out_states is not None else __create_var('float32')
    )

    pos_count = input_states[0] if input_states is not None else None
    true_pos = input_states[1] if input_states is not None else None
    false_pos = input_states[2] if input_states is not None else None

    helper.append_op(
        type="detection_map",
        inputs={
            'Label': label,
            'DetectRes': detect_res,
            'HasState': has_state,
            'PosCount': pos_count,
            'TruePos': true_pos,
            'FalsePos': false_pos,
        },
        outputs={
            'MAP': map_out,
            'AccumPosCount': accum_pos_count_out,
            'AccumTruePos': accum_true_pos_out,
            'AccumFalsePos': accum_false_pos_out,
        },
        attrs={
            'overlap_threshold': overlap_threshold,
            'evaluate_difficult': evaluate_difficult,
            'ap_type': ap_version,
            'class_num': class_num,
        },
    )
    return map_out


def bipartite_match(
    dist_matrix, match_type=None, dist_threshold=None, name=None
):
    """

    This operator implements a greedy bipartite matching algorithm, which is
    used to obtain the matching with the maximum distance based on the input
    distance matrix. For input 2D matrix, the bipartite matching algorithm can
    find the matched column for each row (matched means the largest distance),
    also can find the matched row for each column. And this operator only
    calculate matched indices from column to row. For each instance,
    the number of matched indices is the column number of the input distance
    matrix. **The OP only supports CPU**.

    There are two outputs, matched indices and distance.
    A simple description, this algorithm matched the best (maximum distance)
    row entity to the column entity and the matched indices are not duplicated
    in each row of ColToRowMatchIndices. If the column entity is not matched
    any row entity, set -1 in ColToRowMatchIndices.

    NOTE: the input DistMat can be LoDTensor (with LoD) or Tensor.
    If LoDTensor with LoD, the height of ColToRowMatchIndices is batch size.
    If Tensor, the height of ColToRowMatchIndices is 1.

    NOTE: This API is a very low level API. It is used by :code:`ssd_loss`
    layer. Please consider to use :code:`ssd_loss` instead.

    Args:
        dist_matrix(Variable): This input is a 2-D LoDTensor with shape
            [K, M]. The data type is float32 or float64. It is pair-wise
            distance matrix between the entities represented by each row and
            each column. For example, assumed one entity is A with shape [K],
            another entity is B with shape [M]. The dist_matrix[i][j] is the
            distance between A[i] and B[j]. The bigger the distance is, the
            better matching the pairs are. NOTE: This tensor can contain LoD
            information to represent a batch of inputs. One instance of this
            batch can contain different numbers of entities.
        match_type(str, optional): The type of matching method, should be
           'bipartite' or 'per_prediction'. None ('bipartite') by default.
        dist_threshold(float32, optional): If `match_type` is 'per_prediction',
            this threshold is to determine the extra matching bboxes based
            on the maximum distance, 0.5 by default.
        name(str, optional): For detailed information, please refer
            to :ref:`api_guide_Name`. Usually name is no need to set and
            None by default.

    Returns:
        Tuple:

        matched_indices(Variable): A 2-D Tensor with shape [N, M]. The data
        type is int32. N is the batch size. If match_indices[i][j] is -1, it
        means B[j] does not match any entity in i-th instance.
        Otherwise, it means B[j] is matched to row
        match_indices[i][j] in i-th instance. The row number of
        i-th instance is saved in match_indices[i][j].

        matched_distance(Variable): A 2-D Tensor with shape [N, M]. The data
        type is float32. N is batch size. If match_indices[i][j] is -1,
        match_distance[i][j] is also -1.0. Otherwise, assumed
        match_distance[i][j] = d, and the row offsets of each instance
        are called LoD. Then match_distance[i][j] =
        dist_matrix[d+LoD[i]][j].

    Examples:

        >>> import paddle.fluid as fluid
        >>> x = fluid.data(name='x', shape=[None, 4], dtype='float32')
        >>> y = fluid.data(name='y', shape=[None, 4], dtype='float32')
        >>> iou = fluid.layers.iou_similarity(x=x, y=y)
        >>> matched_indices, matched_dist = fluid.layers.bipartite_match(iou)
    """
    helper = LayerHelper('bipartite_match', **locals())
    match_indices = helper.create_variable_for_type_inference(dtype='int32')
    match_distance = helper.create_variable_for_type_inference(
        dtype=dist_matrix.dtype
    )
    helper.append_op(
        type='bipartite_match',
        inputs={'DistMat': dist_matrix},
        attrs={
            'match_type': match_type,
            'dist_threshold': dist_threshold,
        },
        outputs={
            'ColToRowMatchIndices': match_indices,
            'ColToRowMatchDist': match_distance,
        },
    )
    return match_indices, match_distance


def target_assign(
    input,
    matched_indices,
    negative_indices=None,
    mismatch_value=None,
    name=None,
):
    """

    This operator can be, for given the target bounding boxes or labels,
    to assign classification and regression targets to each prediction as well as
    weights to prediction. The weights is used to specify which prediction would
    not contribute to training loss.

    For each instance, the output `out` and`out_weight` are assigned based on
    `match_indices` and `negative_indices`.
    Assumed that the row offset for each instance in `input` is called lod,
    this operator assigns classification/regression targets by performing the
    following steps:

    1. Assigning all outputs based on `match_indices`:

    .. code-block:: text

        If id = match_indices[i][j] > 0,

            out[i][j][0 : K] = X[lod[i] + id][j % P][0 : K]
            out_weight[i][j] = 1.

        Otherwise,

            out[j][j][0 : K] = {mismatch_value, mismatch_value, ...}
            out_weight[i][j] = 0.

    2. Assigning outputs based on `neg_indices` if `neg_indices` is provided:

    Assumed that i-th instance in `neg_indices` is called `neg_indice`,
    for i-th instance:

    .. code-block:: text

        for id in neg_indice:
            out[i][id][0 : K] = {mismatch_value, mismatch_value, ...}
            out_weight[i][id] = 1.0

    Args:
       input (Variable): This input is a 3D LoDTensor with shape [M, P, K].
           Data type should be int32 or float32.
       matched_indices (Variable): The input matched indices
           is 2D Tenosr<int32> with shape [N, P], If MatchIndices[i][j] is -1,
           the j-th entity of column is not matched to any entity of row in
           i-th instance.
       negative_indices (Variable, optional): The input negative example indices
           are an optional input with shape [Neg, 1] and int32 type, where Neg is
           the total number of negative example indices.
       mismatch_value (float32, optional): Fill this value to the mismatched
           location.
       name (string): The default value is None.  Normally there is no need for
           user to set this property.  For more information, please refer
           to :ref:`api_guide_Name`.

    Returns:
        tuple: A tuple(out, out_weight) is returned.

        out (Variable): a 3D Tensor with shape [N, P, K] and same data type
        with `input`, N and P is the same as they are in `matched_indices`,
        K is the same as it in input of X.

        out_weight (Variable): the weight for output with the shape of [N, P, 1].
        Data type is float32.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            x = fluid.data(
                name='x',
                shape=[4, 20, 4],
                dtype='float',
                lod_level=1)
            matched_id = fluid.data(
                name='indices',
                shape=[8, 20],
                dtype='int32')
            trg, trg_weight = fluid.layers.target_assign(
                x,
                matched_id,
                mismatch_value=0)
    """
    helper = LayerHelper('target_assign', **locals())
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    out_weight = helper.create_variable_for_type_inference(dtype='float32')
    helper.append_op(
        type='target_assign',
        inputs={
            'X': input,
            'MatchIndices': matched_indices,
            'NegIndices': negative_indices,
        },
        outputs={'Out': out, 'OutWeight': out_weight},
        attrs={'mismatch_value': mismatch_value},
    )
    return out, out_weight


def ssd_loss(
    location,
    confidence,
    gt_box,
    gt_label,
    prior_box,
    prior_box_var=None,
    background_label=0,
    overlap_threshold=0.5,
    neg_pos_ratio=3.0,
    neg_overlap=0.5,
    loc_loss_weight=1.0,
    conf_loss_weight=1.0,
    match_type='per_prediction',
    mining_type='max_negative',
    normalize=True,
    sample_size=None,
):
    r"""
	:alias_main: paddle.nn.functional.ssd_loss
	:alias: paddle.nn.functional.ssd_loss,paddle.nn.functional.loss.ssd_loss
	:old_api: paddle.fluid.layers.ssd_loss

    **Multi-box loss layer for object detection algorithm of SSD**

    This layer is to compute detection loss for SSD given the location offset
    predictions, confidence predictions, prior boxes and ground-truth bounding
    boxes and labels, and the type of hard example mining. The returned loss
    is a weighted sum of the localization loss (or regression loss) and
    confidence loss (or classification loss) by performing the following steps:

    1. Find matched bounding box by bipartite matching algorithm.

      1.1 Compute IOU similarity between ground-truth boxes and prior boxes.

      1.2 Compute matched bounding box by bipartite matching algorithm.

    2. Compute confidence for mining hard examples

      2.1. Get the target label based on matched indices.

      2.2. Compute confidence loss.

    3. Apply hard example mining to get the negative example indices and update
       the matched indices.

    4. Assign classification and regression targets

      4.1. Encoded bbox according to the prior boxes.

      4.2. Assign regression targets.

      4.3. Assign classification targets.

    5. Compute the overall objective loss.

      5.1 Compute confidence loss.

      5.2 Compute localization loss.

      5.3 Compute the overall weighted loss.

    Args:
        location (Variable): The location predictions are a 3D Tensor with
            shape [N, Np, 4], N is the batch size, Np is total number of
            predictions for each instance. 4 is the number of coordinate values,
            the layout is [xmin, ymin, xmax, ymax].The data type is float32 or
            float64.
        confidence (Variable): The confidence predictions are a 3D Tensor
            with shape [N, Np, C], N and Np are the same as they are in
            `location`, C is the class number.The data type is float32 or
            float64.
        gt_box (Variable): The ground-truth bounding boxes (bboxes) are a 2D
            LoDTensor with shape [Ng, 4], Ng is the total number of ground-truth
            bboxes of mini-batch input.The data type is float32 or float64.
        gt_label (Variable): The ground-truth labels are a 2D LoDTensor
            with shape [Ng, 1].Ng is the total number of ground-truth bboxes of
            mini-batch input, 1 is the number of class. The data type is float32
            or float64.
        prior_box (Variable): The prior boxes are a 2D Tensor with shape [Np, 4].
            Np and 4 are the same as they are in `location`. The data type is
            float32 or float64.
        prior_box_var (Variable): The variance of prior boxes are a 2D Tensor
            with shape [Np, 4]. Np and 4 are the same as they are in `prior_box`
        background_label (int): The index of background label, 0 by default.
        overlap_threshold (float): If match_type is 'per_prediction', use
            'overlap_threshold' to determine the extra matching bboxes when finding \
            matched boxes. 0.5 by default.
        neg_pos_ratio (float): The ratio of the negative boxes to the positive
            boxes, used only when mining_type is 'max_negative', 3.0 by default.
        neg_overlap (float): The negative overlap upper bound for the unmatched
            predictions. Use only when mining_type is 'max_negative',
            0.5 by default.
        loc_loss_weight (float): Weight for localization loss, 1.0 by default.
        conf_loss_weight (float): Weight for confidence loss, 1.0 by default.
        match_type (str): The type of matching method during training, should
            be 'bipartite' or 'per_prediction', 'per_prediction' by default.
        mining_type (str): The hard example mining type, should be 'hard_example'
            or 'max_negative', now only support `max_negative`.
        normalize (bool): Whether to normalize the SSD loss by the total number
            of output locations, True by default.
        sample_size (int): The max sample size of negative box, used only when
            mining_type is 'hard_example'.

    Returns:
        Variable(Tensor):  The weighted sum of the localization loss and confidence loss, \
        with shape [N * Np, 1], N and Np are the same as they are in
        `location`.The data type is float32 or float64.

    Raises:
        ValueError: If mining_type is 'hard_example', now only support mining \
        type of `max_negative`.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            pb = fluid.data(
                           name='prior_box',
                           shape=[10, 4],
                           dtype='float32')
            pbv = fluid.data(
                           name='prior_box_var',
                           shape=[10, 4],
                           dtype='float32')
            loc = fluid.data(name='target_box', shape=[10, 4], dtype='float32')
            scores = fluid.data(name='scores', shape=[10, 21], dtype='float32')
            gt_box = fluid.data(
                 name='gt_box', shape=[4], lod_level=1, dtype='float32')
            gt_label = fluid.data(
                 name='gt_label', shape=[1], lod_level=1, dtype='float32')
            loss = fluid.layers.ssd_loss(loc, scores, gt_box, gt_label, pb, pbv)
    """

    helper = LayerHelper('ssd_loss', **locals())
    if mining_type != 'max_negative':
        raise ValueError("Only support mining_type == max_negative now.")

    num, num_prior, num_class = confidence.shape
    conf_shape = paddle.shape(confidence)

    def __reshape_to_2d(var):
        out = paddle.flatten(var, 2, -1)
        out = paddle.flatten(out, 0, 1)
        return out

    # 1. Find matched bounding box by prior box.
    #   1.1 Compute IOU similarity between ground-truth boxes and prior boxes.
    iou = iou_similarity(x=gt_box, y=prior_box)
    #   1.2 Compute matched bounding box by bipartite matching algorithm.
    matched_indices, matched_dist = bipartite_match(
        iou, match_type, overlap_threshold
    )

    # 2. Compute confidence for mining hard examples
    # 2.1. Get the target label based on matched indices
    gt_label = paddle.reshape(
        x=gt_label, shape=(len(gt_label.shape) - 1) * (0,) + (-1, 1)
    )
    gt_label.stop_gradient = True
    target_label, _ = target_assign(
        gt_label, matched_indices, mismatch_value=background_label
    )
    # 2.2. Compute confidence loss.
    # Reshape confidence to 2D tensor.
    confidence = __reshape_to_2d(confidence)
    target_label = tensor.cast(x=target_label, dtype='int64')
    target_label = __reshape_to_2d(target_label)
    target_label.stop_gradient = True
    conf_loss = softmax_with_cross_entropy(confidence, target_label)
    # 3. Mining hard examples
    actual_shape = paddle.slice(conf_shape, axes=[0], starts=[0], ends=[2])
    actual_shape.stop_gradient = True
    # shape=(-1, 0) is set for compile-time, the correct shape is set by
    # actual_shape in runtime.
    conf_loss = paddle.reshape(x=conf_loss, shape=actual_shape)
    conf_loss.stop_gradient = True
    neg_indices = helper.create_variable_for_type_inference(dtype='int32')
    dtype = matched_indices.dtype
    updated_matched_indices = helper.create_variable_for_type_inference(
        dtype=dtype
    )
    helper.append_op(
        type='mine_hard_examples',
        inputs={
            'ClsLoss': conf_loss,
            'LocLoss': None,
            'MatchIndices': matched_indices,
            'MatchDist': matched_dist,
        },
        outputs={
            'NegIndices': neg_indices,
            'UpdatedMatchIndices': updated_matched_indices,
        },
        attrs={
            'neg_pos_ratio': neg_pos_ratio,
            'neg_dist_threshold': neg_overlap,
            'mining_type': mining_type,
            'sample_size': sample_size,
        },
    )

    # 4. Assign classification and regression targets
    # 4.1. Encoded bbox according to the prior boxes.
    encoded_bbox = box_coder(
        prior_box=prior_box,
        prior_box_var=prior_box_var,
        target_box=gt_box,
        code_type='encode_center_size',
    )
    # 4.2. Assign regression targets
    target_bbox, target_loc_weight = target_assign(
        encoded_bbox, updated_matched_indices, mismatch_value=background_label
    )
    # 4.3. Assign classification targets
    target_label, target_conf_weight = target_assign(
        gt_label,
        updated_matched_indices,
        negative_indices=neg_indices,
        mismatch_value=background_label,
    )

    # 5. Compute loss.
    # 5.1 Compute confidence loss.
    target_label = __reshape_to_2d(target_label)
    target_label = tensor.cast(x=target_label, dtype='int64')

    conf_loss = softmax_with_cross_entropy(confidence, target_label)
    target_conf_weight = __reshape_to_2d(target_conf_weight)
    conf_loss = conf_loss * target_conf_weight

    # the target_label and target_conf_weight do not have gradient.
    target_label.stop_gradient = True
    target_conf_weight.stop_gradient = True

    # 5.2 Compute regression loss.
    location = __reshape_to_2d(location)
    target_bbox = __reshape_to_2d(target_bbox)

    smooth_l1_loss = paddle.nn.loss.SmoothL1Loss()
    loc_loss = smooth_l1_loss(location, target_bbox)
    target_loc_weight = __reshape_to_2d(target_loc_weight)
    loc_loss = loc_loss * target_loc_weight

    # the target_bbox and target_loc_weight do not have gradient.
    target_bbox.stop_gradient = True
    target_loc_weight.stop_gradient = True

    # 5.3 Compute overall weighted loss.
    loss = conf_loss_weight * conf_loss + loc_loss_weight * loc_loss
    # reshape to [N, Np], N is the batch size and Np is the prior box number.
    # shape=(-1, 0) is set for compile-time, the correct shape is set by
    # actual_shape in runtime.
    loss = paddle.reshape(x=loss, shape=actual_shape)
    loss = paddle.sum(loss, axis=1, keepdim=True)
    if normalize:
        normalizer = paddle.sum(target_loc_weight)
        loss = loss / normalizer

    return loss
