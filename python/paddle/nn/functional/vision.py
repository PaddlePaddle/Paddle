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

from ...fluid.data_feeder import check_variable_and_dtype
from ...fluid.layer_helper import LayerHelper

__all__ = [
    #'affine_channel',
    #'affine_grid',
    #'anchor_generator',
    #'bipartite_match',
    #'box_clip',
    #'box_coder',
    #'box_decoder_and_assign',
    #'collect_fpn_proposals',
    #       'deformable_conv',
    #'deformable_roi_pooling',
    #'density_prior_box',
    #'detection_output',
    #'distribute_fpn_proposals',
    #'fsp_matrix',
    #'generate_mask_labels',
    #'generate_proposal_labels',
    #'generate_proposals',
    #'grid_sampler',
    #'image_resize',
    #'image_resize_short',
    #       'multi_box_head',
    'pixel_shuffle',
    #'prior_box',
    #'prroi_pool',
    #'psroi_pool',
    #'resize_bilinear',
    #'resize_nearest',
    #'resize_trilinear',
    #'retinanet_detection_output',
    #'retinanet_target_assign',
    #'roi_align',
    #'roi_perspective_transform',
    #'roi_pool',
    #'shuffle_channel',
    #'space_to_depth',
    #'yolo_box',
    #'yolov3_loss'
]


def pixel_shuffle(x, upscale_factor, name=None):
    """
        :alias_main: paddle.nn.functional.pixel_shuffle
        :alias: paddle.nn.functional.pixel_shuffle,paddle.nn.functional.vision.pixel_shuffle
    
    This operator rearranges elements in a tensor of shape [N, C, H, W]
    to a tensor of shape [N, C/upscale_factor**2, H*upscale_factor, W*upscale_factor].
    This is useful for implementing efficient sub-pixel convolution
    with a stride of 1/upscale_factor.
    Please refer to the paper: `Real-Time Single Image and Video Super-Resolution
    Using an Efficient Sub-Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158v2>`_ .
    by Shi et. al (2016) for more details.

    Parameters:

        x(Variable): 4-D tensor, the data type should be float32 or float64.
        upscale_factor(int): factor to increase spatial resolution.
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.

    Returns:
        Out(Variable): Reshaped tensor according to the new dimension.

    Raises:
        ValueError: If the square of upscale_factor cannot divide the channels of input.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F
            import numpy as np

            x = np.random.randn(2, 9, 4, 4).astype(np.float32)
            place = fluid.CPUPlace()
            paddle.enable_imperative()

            x_var = paddle.imperative.to_variable(x)
            y_var = F.pixel_shuffle(x_var, 3)
            y_np = y_var.numpy()
            print(y_np.shape) # (2, 1, 12, 12)

    """

    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'pixel_shuffle')
    helper = LayerHelper("pixel_shuffle", **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    if not isinstance(upscale_factor, int):
        raise TypeError("upscale factor must be int type")

    helper.append_op(
        type="pixel_shuffle",
        inputs={"X": x},
        outputs={"Out": out},
        attrs={"upscale_factor": upscale_factor})
    return out
