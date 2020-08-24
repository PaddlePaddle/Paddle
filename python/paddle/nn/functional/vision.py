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

from ...device import get_cudnn_version
from ...fluid.framework import core, in_dygraph_mode, Variable
from ...fluid.layer_helper import LayerHelper
from ...fluid.data_feeder import check_variable_and_dtype

# TODO: define specitial functions used in computer vision task  
from ...fluid.layers import affine_channel  #DEFINE_ALIAS
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
from ...fluid.layers import pixel_shuffle  #DEFINE_ALIAS
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


def affine_grid(theta, out_shape, align_corners=True, name=None):
    """
    It generates a grid of (x,y) coordinates using the parameters of
    the affine transformation that correspond to a set of points where
    the input feature map should be sampled to produce the transformed
    output feature map.

    Args:
        theta (Tensor) - A tensor with shape [N, 2, 3]. It contains a batch of affine transform parameters.
                           The data type can be float32 or float64.
        out_shape (Tensor | list | tuple): The shape of target output with format [batch_size, channel, height, width].
                                             ``out_shape`` can be a Tensor or a list or tuple. The data
                                             type must be int32.
        align_corners(bool): Whether to align corners of target feature map and source feature map. Default: True.
        name(str|None): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, A Tensor with shape [batch_size, H, W, 2] while 'H' and 'W' are the height and width of feature map in affine transformation. The data type is the same as `theta`.

    Raises:
        ValueError: If the type of arguments is not supported.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.nn.functional as F
            import numpy as np
            
            paddle.disable_static()
            # theta shape = [1, 2, 3]
            theta = np.array([[[-0.7, -0.4, 0.3],
                               [ 0.6,  0.5, 1.5]]]).astype("float32")
            theta_t = paddle.to_tensor(theta)
            y_t = F.affine_grid(
                    theta_t,
                    [1, 2, 3, 3],
                    align_corners=False)
            print(y_t.numpy())
            
            #[[[[ 1.0333333   0.76666665]
            #   [ 0.76666665  1.0999999 ]
            #   [ 0.5         1.4333333 ]]
            #
            #  [[ 0.5666667   1.1666666 ]
            #   [ 0.3         1.5       ]
            #   [ 0.03333333  1.8333334 ]]
            #
            #  [[ 0.10000002  1.5666667 ]
            #   [-0.16666666  1.9000001 ]
            #   [-0.43333334  2.2333333 ]]]]
    """
    helper = LayerHelper('affine_grid')

    check_variable_and_dtype(theta, 'theta', ['float32', 'float64'],
                             'affine_grid')

    cudnn_version = get_cudnn_version()
    if cudnn_version is not None and cudnn_version >= 6000 and align_corners:
        use_cudnn = True
    else:
        use_cudnn = False

    if not (isinstance(out_shape, list) or isinstance(out_shape, tuple) or \
            isinstance(out_shape, Variable)):
        raise ValueError("The out_shape should be a list, tuple or Tensor.")

    if in_dygraph_mode():
        _out_shape = out_shape.numpy().tolist() if isinstance(
            out_shape, Variable) else out_shape
        return core.ops.affine_grid(theta, "output_shape", _out_shape,
                                    "align_corners", align_corners, "use_cudnn",
                                    use_cudnn)

    if not isinstance(theta, Variable):
        raise ValueError("The theta should be a Tensor.")

    out = helper.create_variable_for_type_inference(theta.dtype)
    ipts = {'Theta': theta}
    attrs = {"align_corners": align_corners, "use_cudnn": use_cudnn}
    if isinstance(out_shape, Variable):
        ipts['OutputShape'] = out_shape
        check_variable_and_dtype(out_shape, 'out_shape', ['int32'],
                                 'affine_grid')
    else:
        attrs['output_shape'] = out_shape

    helper.append_op(
        type='affine_grid',
        inputs=ipts,
        outputs={'Output': out},
        attrs=None if len(attrs) == 0 else attrs)
    return out
