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
from ...fluid.framework import core, in_dygraph_mode

from ...fluid.layers import affine_channel  #DEFINE_ALIAS
from ...fluid.layers import affine_grid  #DEFINE_ALIAS
from ...fluid.layers import anchor_generator  #DEFINE_ALIAS
from ...fluid.layers import bipartite_match  #DEFINE_ALIAS
from ...fluid.layers import box_clip  #DEFINE_ALIAS
from ...fluid.layers import box_coder  #DEFINE_ALIAS
from ...fluid.layers import box_decoder_and_assign  #DEFINE_ALIAS
from ...fluid.layers import collect_fpn_proposals  #DEFINE_ALIAS
from ...fluid.layers import deformable_roi_pooling  #DEFINE_ALIAS
from ...fluid.layers import density_prior_box  #DEFINE_ALIAS
from ...fluid.layers import detection_output  #DEFINE_ALIAS
from ...fluid.layers import distribute_fpn_proposals  #DEFINE_ALIAS
from ...fluid.layers import generate_mask_labels  #DEFINE_ALIAS
from ...fluid.layers import generate_proposal_labels  #DEFINE_ALIAS
from ...fluid.layers import generate_proposals  #DEFINE_ALIAS
from ...fluid.layers import grid_sampler  #DEFINE_ALIAS
from ...fluid.layers import image_resize  #DEFINE_ALIAS
from ...fluid.layers import prior_box  #DEFINE_ALIAS
from ...fluid.layers import prroi_pool  #DEFINE_ALIAS
from ...fluid.layers import psroi_pool  #DEFINE_ALIAS
from ...fluid.layers import resize_bilinear  #DEFINE_ALIAS
from ...fluid.layers import resize_nearest  #DEFINE_ALIAS
from ...fluid.layers import resize_trilinear  #DEFINE_ALIAS
from ...fluid.layers import roi_align  #DEFINE_ALIAS
from ...fluid.layers import roi_pool  #DEFINE_ALIAS
from ...fluid.layers import space_to_depth  #DEFINE_ALIAS
from ...fluid.layers import yolo_box  #DEFINE_ALIAS
from ...fluid.layers import yolov3_loss  #DEFINE_ALIAS

from ...fluid.layers import fsp_matrix  #DEFINE_ALIAS
from ...fluid.layers import image_resize_short  #DEFINE_ALIAS
# from ...fluid.layers import pixel_shuffle  #DEFINE_ALIAS
from ...fluid.layers import retinanet_detection_output  #DEFINE_ALIAS
from ...fluid.layers import retinanet_target_assign  #DEFINE_ALIAS
from ...fluid.layers import roi_perspective_transform  #DEFINE_ALIAS
from ...fluid.layers import shuffle_channel  #DEFINE_ALIAS

__all__ = [
    'affine_channel',
    'affine_grid',
    'anchor_generator',
    'bipartite_match',
    'box_clip',
    'box_coder',
    'box_decoder_and_assign',
    'collect_fpn_proposals',
    #       'deformable_conv',
    'deformable_roi_pooling',
    'density_prior_box',
    'detection_output',
    'distribute_fpn_proposals',
    'fsp_matrix',
    'generate_mask_labels',
    'generate_proposal_labels',
    'generate_proposals',
    'grid_sampler',
    'image_resize',
    'image_resize_short',
    #       'multi_box_head',
    'pixel_shuffle',
    'prior_box',
    'prroi_pool',
    'psroi_pool',
    'resize_bilinear',
    'resize_nearest',
    'resize_trilinear',
    'retinanet_detection_output',
    'retinanet_target_assign',
    'roi_align',
    'roi_perspective_transform',
    'roi_pool',
    'shuffle_channel',
    'space_to_depth',
    'yolo_box',
    'yolov3_loss'
]


def pixel_shuffle(x, upscale_factor, data_format="NCHW", name=None):
    """
    This API implements pixel shuffle operation.
    See more details in :ref:`api_nn_vision_PixelShuffle` .

    Parameters:

        x(Tensor): 4-D tensor, the data type should be float32 or float64.
        upscale_factor(int): factor to increase spatial resolution.
        data_format (str): The data format of the input and output data. An optional string from: "NCHW", "NHWC". The default is "NCHW". When it is "NCHW", the data is stored in the order of: [batch_size, input_channels, input_height, input_width].
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.

    Returns:
        Out(tensor): Reshaped tensor according to the new dimension.

    Raises:
        ValueError: If the square of upscale_factor cannot divide the channels of input.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F
            import numpy as np

            x = np.random.randn(2, 9, 4, 4).astype(np.float32)
            paddle.disable_static()

            x_var = paddle.to_tensor(x)
            out_var = F.pixel_shuffle(x_var, 3)
            out = out_var.numpy()
            print(out.shape) 
            # (2, 1, 12, 12)

    """
    if in_dygraph_mode():
        return core.ops.pixel_shuffle(x, "upscale_factor", upscale_factor,
                                      "data_format", data_format)

    helper = LayerHelper("pixel_shuffle", **locals())
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'pixel_shuffle')

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    if not isinstance(upscale_factor, int):
        raise TypeError("upscale factor must be int type")

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError("Attr(data_format) should be 'NCHW' or 'NHWC'."
                         "But recevie Attr(data_format): {} ".format(
                             data_format))
    helper.append_op(
        type="pixel_shuffle",
        inputs={"X": x},
        outputs={"Out": out},
        attrs={"upscale_factor": upscale_factor,
               "data_format": data_format})
    return out
