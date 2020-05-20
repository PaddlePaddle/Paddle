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

# TODO: define specitial functions used in computer vision task  
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
