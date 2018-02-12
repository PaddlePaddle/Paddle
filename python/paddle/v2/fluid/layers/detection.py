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
from ..framework import Variable
from tensor import concat
from ops import reshape
import math

__all__ = [
    'detection_output',
    'prior_box',
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


def prior_box(inputs,
              image,
              min_ratio,
              max_ratio,
              aspect_ratios,
              base_size,
              steps=None,
              step_w=None,
              step_h=None,
              offset=0.5,
              variance=[0.1, 0.1, 0.1, 0.1],
              flip=False,
              clip=False,
              min_sizes=None,
              max_sizes=None,
              name=None):
    """
    **Prior_boxes**

    Generate prior boxes for SSD(Single Shot MultiBox Detector)
    algorithm. The details of this algorithm, please refer the
    section 2.2 of SSD paper (SSD: Single Shot MultiBox Detector)
    <https://arxiv.org/abs/1512.02325>`_ .
    
    Args:
       inputs(list): The list of input Variables, the format
            of all Variables is NCHW.
       image(Variable): The input image data of PriorBoxOp,
            the layout is NCHW.
       min_ratio(int): the min ratio of generated prior boxes.
       max_ratio(int): the max ratio of generated prior boxes.
       aspect_ratios(list): the aspect ratios of generated prior
            boxes. The length of input and aspect_ratios must be equal.
       base_size(int): the base_size is used to get min_size
            and max_size according to min_ratio and max_ratio.
       step_w(list, optional, default=None): Prior boxes step
            across width. If step_w[i] == 0.0, the prior boxes step
            across width of the inputs[i] will be automatically calculated.
       step_h(list, optional, default=None): Prior boxes step
            across height, If step_h[i] == 0.0, the prior boxes
            step across height of the inputs[i] will be automatically calculated.
       offset(float, optional, default=0.5): Prior boxes center offset.
       variance(list, optional, default=[0.1, 0.1, 0.1, 0.1]): the variances
            to be encoded in prior boxes.
       flip(bool, optional, default=False): Whether to flip
            aspect ratios.
       clip(bool, optional, default=False): Whether to clip
            out-of-boundary boxes.
       min_sizes(list, optional, default=None): If `len(inputs) <=2`,
            min_sizes must be set up, and the length of min_sizes
            should equal to the length of inputs.
       max_sizes(list, optional, default=None): If `len(inputs) <=2`,
            max_sizes must be set up, and the length of min_sizes
            should equal to the length of inputs.
       name(str, optional, None): Name of the prior box layer.
    
    Returns:
        boxes(Variable): the output prior boxes of PriorBoxOp.
             The layout is [num_priors, 4]. num_priors is the total
             box count of each position of inputs.
        Variances(Variable): the expanded variances of PriorBoxOp.
             The layout is [num_priors, 4]. num_priors is the total
             box count of each position of inputs
    
    Examples:
        .. code-block:: python
    
          prior_box(
             inputs = [conv1, conv2, conv3, conv4, conv5, conv6],
             image = data,
             min_ratio = 20, # 0.20
             max_ratio = 90, # 0.90
             offset = 0.5,
             base_size = 300,
             variance = [0.1,0.1,0.1,0.1],
             aspect_ratios = [[2.], [2., 3.], [2., 3.], [2., 3.], [2.], [2.]],
             flip=True,
             clip=True)
    """

    def _prior_box_(input,
                    image,
                    min_sizes,
                    max_sizes,
                    aspect_ratios,
                    variance,
                    flip=False,
                    clip=False,
                    step_w=0.0,
                    step_h=0.0,
                    offset=0.5,
                    name=None):
        helper = LayerHelper("prior_box", **locals())
        dtype = helper.input_dtype()

        box = helper.create_tmp_variable(dtype)
        var = helper.create_tmp_variable(dtype)
        helper.append_op(
            type="prior_box",
            inputs={"Input": input,
                    "Image": image},
            outputs={"Boxes": box,
                     "Variances": var},
            attrs={
                'min_sizes': min_sizes,
                'max_sizes': max_sizes,
                'aspect_ratios': aspect_ratios,
                'variances': variance,
                'flip': flip,
                'clip': clip,
                'step_w': step_w,
                'step_h': step_h,
                'offset': offset
            })
        return box, var

    def _reshape_with_axis_(input, axis=1):
        if not (axis > 0 and axis < len(input.shape)):
            raise ValueError("The axis should be smaller than "
                             "the arity of input and bigger than 0.")
        new_shape = [
            -1, reduce(lambda x, y: x * y, input.shape[axis:len(input.shape)])
        ]
        out = reshape(x=input, shape=new_shape)
        return out

    assert isinstance(inputs, list), 'inputs should be a list.'
    num_layer = len(inputs)

    if num_layer <= 2:
        assert min_sizes is not None and max_sizes is not None
        assert len(min_sizes) == num_layer and len(max_sizes) == num_layer
    else:
        min_sizes = []
        max_sizes = []
        step = int(math.floor(((max_ratio - min_ratio)) / (num_layer - 2)))
        for ratio in xrange(min_ratio, max_ratio + 1, step):
            min_sizes.append(base_size * ratio / 100.)
            max_sizes.append(base_size * (ratio + step) / 100.)
        min_sizes = [base_size * .10] + min_sizes
        max_sizes = [base_size * .20] + max_sizes

    if aspect_ratios:
        if not (isinstance(aspect_ratios, list) and
                len(aspect_ratios) == num_layer):
            raise ValueError(
                'aspect_ratios should be list and the length of inputs '
                'and aspect_ratios should be the same.')
    if step_h:
        if not (isinstance(step_h, list) and len(step_h) == num_layer):
            raise ValueError(
                'step_h should be list and the length of inputs and '
                'step_h should be the same.')
    if step_w:
        if not (isinstance(step_w, list) and len(step_w) == num_layer):
            raise ValueError(
                'step_w should be list and the length of inputs and '
                'step_w should be the same.')
    if steps:
        if not (isinstance(steps, list) and len(steps) == num_layer):
            raise ValueError(
                'steps should be list and the length of inputs and '
                'step_w should be the same.')
        step_w = steps
        step_h = steps

    box_results = []
    var_results = []
    for i, input in enumerate(inputs):
        min_size = min_sizes[i]
        max_size = max_sizes[i]
        aspect_ratio = []
        if not isinstance(min_size, list):
            min_size = [min_size]
        if not isinstance(max_size, list):
            max_size = [max_size]
        if aspect_ratios:
            aspect_ratio = aspect_ratios[i]
            if not isinstance(aspect_ratio, list):
                aspect_ratio = [aspect_ratio]

        box, var = _prior_box_(input, image, min_size, max_size, aspect_ratio,
                               variance, flip, clip, step_w[i]
                               if step_w else 0.0, step_h[i]
                               if step_w else 0.0, offset)

        box_results.append(box)
        var_results.append(var)

    if len(box_results) == 1:
        box = box_results[0]
        var = var_results[0]
    else:
        reshaped_boxes = []
        reshaped_vars = []
        for i in range(len(box_results)):
            reshaped_boxes.append(_reshape_with_axis_(box_results[i], axis=3))
            reshaped_vars.append(_reshape_with_axis_(var_results[i], axis=3))

        box = concat(reshaped_boxes)
        var = concat(reshaped_vars)

    return box, var
