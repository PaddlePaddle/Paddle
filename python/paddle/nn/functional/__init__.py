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

# TODO: import all neural network related api under this directory,
# including layers, linear, conv, rnn etc.

# TODO: define alias in functional directory
from . import conv
from .activation import elu  # noqa: F401
from .activation import elu_  # noqa: F401
# from .activation import erf  # noqa: F401
from .activation import gelu  # noqa: F401
from .activation import hardshrink  # noqa: F401
from .activation import hardtanh  # noqa: F401
from .activation import hardsigmoid  # noqa: F401
from .activation import hardswish  # noqa: F401
from .activation import leaky_relu  # noqa: F401
from .activation import log_sigmoid  # noqa: F401
from .activation import maxout  # noqa: F401
from .activation import prelu  # noqa: F401
from .activation import relu  # noqa: F401
from .activation import relu_  # noqa: F401
from .activation import relu6  # noqa: F401
from .activation import selu  # noqa: F401
from .activation import sigmoid  # noqa: F401
# from .activation import soft_relu  # noqa: F401
from .activation import softmax  # noqa: F401
from .activation import softmax_  # noqa: F401
from .activation import softplus  # noqa: F401
from .activation import softshrink  # noqa: F401
from .activation import softsign  # noqa: F401
from .activation import swish  # noqa: F401
from .activation import tanh  # noqa: F401
from .activation import tanh_  # noqa: F401
from .activation import tanhshrink  # noqa: F401
from .activation import thresholded_relu  # noqa: F401
from .activation import log_softmax  # noqa: F401
from .common import dropout  # noqa: F401
from .common import dropout2d  # noqa: F401
from .common import dropout3d  # noqa: F401
from .common import alpha_dropout  # noqa: F401
# from .common import embedding        # noqa: F401
# from .common import fc  # noqa: F401
from .common import label_smooth
# from .common import one_hot  # noqa: F401
from .common import pad  # noqa: F401
# from .common import pad_constant_like  # noqa: F401
# from .common import pad2d  # noqa: F401
from .common import cosine_similarity  # noqa: F401
from .common import unfold  # noqa: F401
# from .common import bilinear_tensor_product        # noqa: F401
from .common import interpolate  # noqa: F401
from .common import upsample  # noqa: F401
from .common import bilinear  # noqa: F401
from .conv import conv1d  # noqa: F401
from .conv import conv1d_transpose  # noqa: F401
from .common import linear  # noqa: F401
from .conv import conv2d  # noqa: F401
from .conv import conv2d_transpose  # noqa: F401
from .conv import conv3d  # noqa: F401
from .conv import conv3d_transpose  # noqa: F401
# from .extension import add_position_encoding  # noqa: F401
# from .extension import autoincreased_step_counter        # noqa: F401
# from .extension import continuous_value_model  # noqa: F401
# from .extension import filter_by_instag  # noqa: F401
# from .extension import linear_chain_crf        # noqa: F401
# from .extension import merge_selected_rows        # noqa: F401
# from .extension import multiclass_nms  # noqa: F401
# from .extension import polygon_box_transform  # noqa: F401
# from .extension import random_crop  # noqa: F401
# from .extension import rpn_target_assign  # noqa: F401
# from .extension import similarity_focus  # noqa: F401
# from .extension import target_assign  # noqa: F401
# from .extension import temporal_shift  # noqa: F401
# from .extension import warpctc  # noqa: F401
from .extension import diag_embed  # noqa: F401
# from .lod import sequence_concat        # noqa: F401
# from .lod import sequence_conv        # noqa: F401
# from .lod import sequence_enumerate        # noqa: F401
# from .lod import sequence_expand_as        # noqa: F401
# from .lod import sequence_expand        # noqa: F401
# from .lod import sequence_first_step        # noqa: F401
# from .lod import sequence_last_step        # noqa: F401
# from .lod import sequence_mask        # noqa: F401
# from .lod import sequence_pad        # noqa: F401
# from .lod import sequence_pool        # noqa: F401
# from .lod import sequence_reshape        # noqa: F401
# from .lod import sequence_reverse        # noqa: F401
# from .lod import sequence_scatter        # noqa: F401
# from .lod import sequence_slice        # noqa: F401
# from .lod import sequence_softmax        # noqa: F401
# from .lod import sequence_unpad        # noqa: F401
# from .lod import array_length        # noqa: F401
# from .lod import array_read        # noqa: F401
# from .lod import array_write        # noqa: F401
# from .lod import create_array        # noqa: F401
# from .lod import hash  # noqa: F401
# from .lod import im2sequence        # noqa: F401
# from .lod import lod_append        # noqa: F401
# from .lod import lod_reset        # noqa: F401
# from .lod import reorder_lod_tensor_by_rank        # noqa: F401
# from .lod import tensor_array_to_tensor        # noqa: F401
# from .lod import dynamic_gru        # noqa: F401
# from .lod import dynamic_lstm        # noqa: F401
# from .lod import dynamic_lstmp        # noqa: F401
from .loss import binary_cross_entropy  # noqa: F401
from .loss import binary_cross_entropy_with_logits  # noqa: F401
# from .loss import bpr_loss  # noqa: F401
# from .loss import center_loss  # noqa: F401
#from .loss import cross_entropy  # noqa: F401
from .loss import cross_entropy  # noqa: F401
from .loss import dice_loss  # noqa: F401
from .loss import hsigmoid_loss  # noqa: F401
from .loss import kl_div  # noqa: F401
from .loss import l1_loss  # noqa: F401
from .loss import log_loss  # noqa: F401
from .loss import margin_ranking_loss  # noqa: F401
from .loss import mse_loss  # noqa: F401
from .loss import nll_loss  # noqa: F401
# from .loss import nce        # noqa: F401
from .loss import npair_loss  # noqa: F401
from .loss import sigmoid_focal_loss  # noqa: F401
# from .loss import smooth_l1  # noqa: F401
from .loss import smooth_l1_loss  # noqa: F401
from .loss import softmax_with_cross_entropy  # noqa: F401
from .loss import square_error_cost  # noqa: F401
# from .loss import teacher_student_sigmoid_loss  # noqa: F401
from .loss import ctc_loss  # noqa: F401
# from .norm import data_norm        # noqa: F401
# from .norm import group_norm        # noqa: F401
from .norm import batch_norm  # noqa: F401
from .norm import instance_norm  # noqa: F401
from .norm import layer_norm  # noqa: F401
from .norm import local_response_norm  # noqa: F401
from .norm import normalize  # noqa: F401
# from .norm import spectral_norm        # noqa: F401
# from .pooling import pool2d  # noqa: F401
# from .pooling import pool3d  # noqa: F401
from .pooling import avg_pool1d  # noqa: F401
from .pooling import avg_pool2d  # noqa: F401
from .pooling import avg_pool3d  # noqa: F401
from .pooling import max_pool1d  # noqa: F401
from .pooling import max_pool2d  # noqa: F401
from .pooling import max_pool3d  # noqa: F401

from .pooling import adaptive_max_pool1d  # noqa: F401
from .pooling import adaptive_max_pool2d  # noqa: F401
from .pooling import adaptive_max_pool3d  # noqa: F401
from .pooling import adaptive_avg_pool1d  # noqa: F401
from .pooling import adaptive_avg_pool2d  # noqa: F401
from .pooling import adaptive_avg_pool3d  # noqa: F401

# from .rnn import rnn  # noqa: F401
# from .rnn import birnn  # noqa: F401
# from .rnn import gru_unit        # noqa: F401
# from .rnn import lstm        # noqa: F401
# from .rnn import lstm_unit        # noqa: F401
# from .vision import affine_channel  # noqa: F401
from .vision import affine_grid  # noqa: F401
# from .vision import anchor_generator  # noqa: F401
# from .vision import bipartite_match  # noqa: F401
# from .vision import box_clip  # noqa: F401
# from .vision import box_coder  # noqa: F401
# from .vision import box_decoder_and_assign  # noqa: F401
# from .vision import collect_fpn_proposals  # noqa: F401
# from .vision import deformable_conv  # noqa: F401
# from .vision import deformable_roi_pooling  # noqa: F401
# from .vision import density_prior_box  # noqa: F401
# from .vision import detection_output  # noqa: F401
# from .vision import distribute_fpn_proposals  # noqa: F401
# from .vision import fsp_matrix  # noqa: F401
# from .vision import generate_mask_labels  # noqa: F401
# from .vision import generate_proposal_labels  # noqa: F401
# from .vision import generate_proposals  # noqa: F401
from .vision import grid_sample  # noqa: F401
# from .vision import image_resize  # noqa: F401
# from .vision import image_resize_short  # noqa: F401
# from .vision import multi_box_head  # noqa: F401
from .vision import pixel_shuffle  # noqa: F401
# from .vision import prior_box  # noqa: F401
# from .vision import prroi_pool  # noqa: F401
# from .vision import psroi_pool  # noqa: F401
# from .vision import resize_bilinear  # noqa: F401
# from .vision import resize_nearest  # noqa: F401
# from .vision import resize_trilinear  # noqa: F401
# from .vision import retinanet_detection_output  # noqa: F401
# from .vision import retinanet_target_assign  # noqa: F401
# from .vision import roi_align  # noqa: F401
# from .vision import roi_perspective_transform  # noqa: F401
# from .vision import roi_pool  # noqa: F401
# from .vision import shuffle_channel  # noqa: F401
# from .vision import space_to_depth  # noqa: F401
# from .vision import yolo_box  # noqa: F401
# from .vision import yolov3_loss  # noqa: F401
from .input import one_hot  # noqa: F401
from .input import embedding  # noqa: F401
from ...fluid.layers import gather_tree
from ...fluid.layers import temporal_shift

__all__ = [     #noqa
           'conv1d',
           'conv1d_transpose',
           'conv2d',
           'conv2d_transpose',
           'conv3d',
           'conv3d_transpose',
           'elu',
           'gelu',
           'hardshrink',
           'hardtanh',
           'hardsigmoid',
           'hardswish',
           'leaky_relu',
           'log_sigmoid',
           'maxout',
           'prelu',
           'relu',
           'relu6',
           'selu',
           'softmax',
           'softplus',
           'softshrink',
           'softsign',
           'sigmoid',
           'swish',
           'tanhshrink',
           'thresholded_relu',
           'log_softmax',
           'diag_embed',
           'dropout',
           'dropout2d',
           'dropout3d',
           'alpha_dropout',
           'label_smooth',
           'linear',
           'pad',
           'unfold',
           'interpolate',
           'upsample',
           'bilinear',
           'cosine_similarity',
           'avg_pool1d',
           'avg_pool2d',
           'avg_pool3d',
           'max_pool1d',
           'max_pool2d',
           'max_pool3d',
           'adaptive_avg_pool1d',
           'adaptive_avg_pool2d',
           'adaptive_avg_pool3d',
           'adaptive_max_pool1d',
           'adaptive_max_pool2d',
           'adaptive_max_pool3d',
           'binary_cross_entropy',
           'binary_cross_entropy_with_logits',
           'cross_entropy',
           'dice_loss',
           'hsigmoid_loss',
           'kl_div',
           'l1_loss',
           'log_loss',
           'mse_loss',
           'margin_ranking_loss',
           'nll_loss',
           'npair_loss',
           'sigmoid_focal_loss',
           'smooth_l1_loss',
           'softmax_with_cross_entropy',
           'square_error_cost',
           'ctc_loss',
           'affine_grid',
           'grid_sample',
           'local_response_norm',
           'pixel_shuffle',
           'embedding',
           'gather_tree',
           'one_hot',
           'normalize'
]
