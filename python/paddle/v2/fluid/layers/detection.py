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
from operator import mul
import math

__all__ = ['prior_box', ]


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

    Generate prior boxes for SSD(Single Shot MultiBox Detector) algorithm.
    The details of this algorithm, please refer the section 2.2 of SSD paper
    (SSD: Single Shot MultiBox Detector)<https://arxiv.org/abs/1512.02325>`_ .

    Args:
       inputs(list): The list of input Variables, the format of all Variables is NCHW.
       image(Variable): The input image data of PriorBoxOp, the layout is NCHW.
       min_ratio(int): the min ratio of generated prior boxes.
       max_ratio(int): the max ratio of generated prior boxes.
       aspect_ratios(list): the aspect ratios of generated prior boxes.
            The length of input and aspect_ratios must be equal.
       base_size(int): the base_size is used to get min_size and max_size
            according to min_ratio and max_ratio.
       step_w(list, optional, default=None): Prior boxes step across width.
            If step_w[i] == 0.0, the prior boxes step across width of the inputs[i]
            will be automatically calculated.
       step_h(list, optional, default=None): Prior boxes step across height,
            If step_h[i] == 0.0, the prior boxes step across height of the inputs[i]
            will be automatically calculated.
       offset(float, optional, default=0.5): Prior boxes center offset.
       variance(list, optional, default=[0.1, 0.1, 0.1, 0.1]): the variances
            to be encoded in prior boxes.
       flip(bool, optional, default=False): Whether to flip aspect ratios.
       clip(bool, optional, default=False): Whether to clip out-of-boundary boxes.
       min_sizes(list, optional, default=None): If `len(inputs) <=2`, min_sizes must
            be set up, and the length of min_sizes should equal to the length of inputs.
       max_sizes(list, optional, default=None): If `len(inputs) <=2`, max_sizes must
            be set up, and the length of min_sizes should equal to the length of inputs.
       name(str, optional, None): Name of the prior box layer.

    Returns:
        boxes(Variable): the output prior boxes of PriorBoxOp. The layout is
             [num_priors, 4]. num_priors is the total box count of each
              position of inputs.
        Variances(Variable): the expanded variances of PriorBoxOp. The layout
             is [num_priors, 4]. num_priors is the total box count of each
             position of inputs

    Examples:
        .. code-block:: python

          prior_boxes(
             inputs = [conv1, conv2, conv3, conv4, conv5, conv6],
             image = data,
             min_ratio = 20, # 0.20
             max_ratio = 90, # 0.90
             steps = [8., 16., 32., 64., 100., 300.],
             aspect_ratios = [[2.], [2., 3.], [2., 3.], [2., 3.], [2.], [2.]],
             base_size = 300,
             offset = 0.5,
             variance = [0.1,0.1,0.1,0.1],
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
            raise ValueError(
                "The axis should be smaller than the arity of input and bigger than 0."
            )
        new_shape = [-1, reduce(mul, input.shape[axis:len(input.shape)], 1)]
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
