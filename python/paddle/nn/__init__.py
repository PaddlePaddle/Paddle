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

from .layer import norm
from .functional import extension

__all__ = []
__all__ += norm.__all__
__all__ += extension.__all__

# TODO: define alias in nn directory
# from .clip import ErrorClipByValue   #DEFINE_ALIAS
# from .clip import GradientClipByGlobalNorm   #DEFINE_ALIAS
# from .clip import GradientClipByNorm   #DEFINE_ALIAS
# from .clip import GradientClipByValue   #DEFINE_ALIAS
# from .clip import set_gradient_clip   #DEFINE_ALIAS
# from .clip import clip   #DEFINE_ALIAS
# from .clip import clip_by_norm   #DEFINE_ALIAS
# from .initalizer import Bilinear   #DEFINE_ALIAS
# from .initalizer import Constant   #DEFINE_ALIAS
# from .initalizer import MSRA   #DEFINE_ALIAS
# from .initalizer import Normal   #DEFINE_ALIAS
# from .initalizer import TruncatedNormal   #DEFINE_ALIAS
# from .initalizer import Uniform   #DEFINE_ALIAS
# from .initalizer import Xavier   #DEFINE_ALIAS
# from .decode import BeamSearchDecoder   #DEFINE_ALIAS
# from .decode import Decoder   #DEFINE_ALIAS
# from .decode import beam_search   #DEFINE_ALIAS
# from .decode import beam_search_decode   #DEFINE_ALIAS
# from .decode import crf_decoding   #DEFINE_ALIAS
# from .decode import ctc_greedy_decoder   #DEFINE_ALIAS
# from .decode import dynamic_decode   #DEFINE_ALIAS
# from .decode import gather_tree   #DEFINE_ALIAS
# from .bin.conv import 0   #DEFINE_ALIAS
# from .control_flow import case   #DEFINE_ALIAS
# from .control_flow import cond   #DEFINE_ALIAS
# from .control_flow import DynamicRNN   #DEFINE_ALIAS
# from .control_flow import StaticRNN   #DEFINE_ALIAS
# from .control_flow import switch_case   #DEFINE_ALIAS
# from .control_flow import while_loop   #DEFINE_ALIAS
# from .control_flow import rnn   #DEFINE_ALIAS
# from .layer.conv import Conv2D   #DEFINE_ALIAS
# from .layer.conv import Conv2DTranspose   #DEFINE_ALIAS
# from .layer.conv import Conv3D   #DEFINE_ALIAS
# from .layer.conv import Conv3DTranspose   #DEFINE_ALIAS
# from .layer.conv import TreeConv   #DEFINE_ALIAS
# from .layer.conv import Conv1D   #DEFINE_ALIAS
# from .layer.loss import NCELoss   #DEFINE_ALIAS
from .layer.loss import CrossEntropyLoss  #DEFINE_ALIAS
# from .layer.loss import MSELoss   #DEFINE_ALIAS
from .layer.loss import L1Loss  #DEFINE_ALIAS
from .layer import loss  #DEFINE_ALIAS
from .layer import conv  #DEFINE_ALIAS
from .layer.conv import Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose  #DEFINE_ALIAS
from .layer.loss import NLLLoss  #DEFINE_ALIAS
from .layer.loss import BCELoss  #DEFINE_ALIAS
# from .layer.learning_rate import CosineDecay   #DEFINE_ALIAS
# from .layer.learning_rate import ExponentialDecay   #DEFINE_ALIAS
# from .layer.learning_rate import InverseTimeDecay   #DEFINE_ALIAS
# from .layer.learning_rate import NaturalExpDecay   #DEFINE_ALIAS
# from .layer.learning_rate import NoamDecay   #DEFINE_ALIAS
# from .layer.learning_rate import PiecewiseDecay   #DEFINE_ALIAS
# from .layer.learning_rate import PolynomialDecay   #DEFINE_ALIAS
# from .layer.transformer import    #DEFINE_ALIAS
# from .layer.norm import BatchNorm   #DEFINE_ALIAS
# from .layer.norm import GroupNorm   #DEFINE_ALIAS
# from .layer.norm import LayerNorm   #DEFINE_ALIAS
from .layer.norm import InstanceNorm  #DEFINE_ALIAS
# from .layer.norm import SpectralNorm   #DEFINE_ALIAS
from .layer.activation import HSigmoid  #DEFINE_ALIAS
# from .layer.activation import PReLU   #DEFINE_ALIAS
from .layer.activation import ReLU  #DEFINE_ALIAS
from .layer.activation import Sigmoid  #DEFINE_ALIAS
# from .layer.activation import Softmax   #DEFINE_ALIAS
# from .layer.activation import LogSoftmax   #DEFINE_ALIAS
from .layer.extension import RowConv  #DEFINE_ALIAS
from .layer.activation import LogSoftmax  #DEFINE_ALIAS
# from .layer.rnn import RNNCell   #DEFINE_ALIAS
# from .layer.rnn import GRUCell   #DEFINE_ALIAS
# from .layer.rnn import LSTMCell   #DEFINE_ALIAS
# from .layer.common import BilinearTensorProduct   #DEFINE_ALIAS
# from .layer.common import Pool2D   #DEFINE_ALIAS
# from .layer.common import Embedding   #DEFINE_ALIAS
# from .layer.common import Linear   #DEFINE_ALIAS
# from .layer.common import UpSample   #DEFINE_ALIAS
from .functional.conv import conv2d  #DEFINE_ALIAS
from .functional.conv import conv2d_transpose  #DEFINE_ALIAS
from .functional.conv import conv3d  #DEFINE_ALIAS
from .functional.conv import conv3d_transpose  #DEFINE_ALIAS
# from .functional.loss import bpr_loss   #DEFINE_ALIAS
# from .functional.loss import center_loss   #DEFINE_ALIAS
# from .functional.loss import cross_entropy   #DEFINE_ALIAS
# from .functional.loss import dice_loss   #DEFINE_ALIAS
# from .functional.loss import edit_distance   #DEFINE_ALIAS
# from .functional.loss import huber_loss   #DEFINE_ALIAS
# from .functional.loss import iou_similarity   #DEFINE_ALIAS
# from .functional.loss import kldiv_loss   #DEFINE_ALIAS
# from .functional.loss import log_loss   #DEFINE_ALIAS
# from .functional.loss import margin_rank_loss   #DEFINE_ALIAS
# from .functional.loss import mse_loss   #DEFINE_ALIAS
# from .functional.loss import nce   #DEFINE_ALIAS
# from .functional.loss import npair_loss   #DEFINE_ALIAS
# from .functional.loss import rank_loss   #DEFINE_ALIAS
# from .functional.loss import sampled_softmax_with_cross_entropy   #DEFINE_ALIAS
# from .functional.loss import sigmoid_cross_entropy_with_logits   #DEFINE_ALIAS
# from .functional.loss import sigmoid_focal_loss   #DEFINE_ALIAS
# from .functional.loss import smooth_l1   #DEFINE_ALIAS
# from .functional.loss import softmax_with_cross_entropy   #DEFINE_ALIAS
# from .functional.loss import square_error_cost   #DEFINE_ALIAS
# from .functional.loss import ssd_loss   #DEFINE_ALIAS
# from .functional.loss import teacher_student_sigmoid_loss   #DEFINE_ALIAS
# from .functional.learning_rate import cosine_decay   #DEFINE_ALIAS
# from .functional.learning_rate import exponential_decay   #DEFINE_ALIAS
# from .functional.learning_rate import inverse_time_decay   #DEFINE_ALIAS
# from .functional.learning_rate import natural_exp_decay   #DEFINE_ALIAS
# from .functional.learning_rate import noam_decay   #DEFINE_ALIAS
# from .functional.learning_rate import piecewise_decay   #DEFINE_ALIAS
# from .functional.learning_rate import polynomial_decay   #DEFINE_ALIAS
# from .functional.learning_rate import linear_lr_warmup   #DEFINE_ALIAS
# from .functional.transformer import    #DEFINE_ALIAS
# from .functional.pooling import pool2d   #DEFINE_ALIAS
# from .functional.pooling import pool3d   #DEFINE_ALIAS
# from .functional.pooling import adaptive_pool2d   #DEFINE_ALIAS
# from .functional.pooling import adaptive_pool3d   #DEFINE_ALIAS
# from .functional.norm import batch_norm   #DEFINE_ALIAS
# from .functional.norm import data_norm   #DEFINE_ALIAS
# from .functional.norm import group_norm   #DEFINE_ALIAS
# from .functional.norm import instance_norm   #DEFINE_ALIAS
# from .functional.norm import l2_normalize   #DEFINE_ALIAS
# from .functional.norm import layer_norm   #DEFINE_ALIAS
# from .functional.norm import lrn   #DEFINE_ALIAS
# from .functional.norm import spectral_norm   #DEFINE_ALIAS
# from .functional.vision import affine_channel   #DEFINE_ALIAS
# from .functional.vision import affine_grid   #DEFINE_ALIAS
# from .functional.vision import anchor_generator   #DEFINE_ALIAS
# from .functional.vision import bipartite_match   #DEFINE_ALIAS
# from .functional.vision import box_clip   #DEFINE_ALIAS
# from .functional.vision import box_coder   #DEFINE_ALIAS
# from .functional.vision import box_decoder_and_assign   #DEFINE_ALIAS
# from .functional.vision import collect_fpn_proposals   #DEFINE_ALIAS
# from .functional.vision import deformable_conv   #DEFINE_ALIAS
# from .functional.vision import deformable_roi_pooling   #DEFINE_ALIAS
# from .functional.vision import density_prior_box   #DEFINE_ALIAS
# from .functional.vision import detection_output   #DEFINE_ALIAS
# from .functional.vision import distribute_fpn_proposals   #DEFINE_ALIAS
# from .functional.vision import fsp_matrix   #DEFINE_ALIAS
# from .functional.vision import generate_mask_labels   #DEFINE_ALIAS
# from .functional.vision import generate_proposal_labels   #DEFINE_ALIAS
# from .functional.vision import generate_proposals   #DEFINE_ALIAS
# from .functional.vision import grid_sampler   #DEFINE_ALIAS
# from .functional.vision import image_resize   #DEFINE_ALIAS
# from .functional.vision import image_resize_short   #DEFINE_ALIAS
# from .functional.vision import multi_box_head   #DEFINE_ALIAS
# from .functional.vision import pixel_shuffle   #DEFINE_ALIAS
# from .functional.vision import prior_box   #DEFINE_ALIAS
# from .functional.vision import prroi_pool   #DEFINE_ALIAS
# from .functional.vision import psroi_pool   #DEFINE_ALIAS
# from .functional.vision import resize_bilinear   #DEFINE_ALIAS
# from .functional.vision import resize_nearest   #DEFINE_ALIAS
# from .functional.vision import resize_trilinear   #DEFINE_ALIAS
# from .functional.vision import retinanet_detection_output   #DEFINE_ALIAS
# from .functional.vision import retinanet_target_assign   #DEFINE_ALIAS
# from .functional.vision import roi_align   #DEFINE_ALIAS
# from .functional.vision import roi_perspective_transform   #DEFINE_ALIAS
# from .functional.vision import roi_pool   #DEFINE_ALIAS
# from .functional.vision import shuffle_channel   #DEFINE_ALIAS
# from .functional.vision import space_to_depth   #DEFINE_ALIAS
# from .functional.vision import yolo_box   #DEFINE_ALIAS
# from .functional.vision import yolov3_loss   #DEFINE_ALIAS
# from .functional.activation import brelu   #DEFINE_ALIAS
# from .functional.activation import elu   #DEFINE_ALIAS
# from .functional.activation import erf   #DEFINE_ALIAS
# from .functional.activation import gelu   #DEFINE_ALIAS
# from .functional.activation import hard_shrink   #DEFINE_ALIAS
# from .functional.activation import hard_sigmoid   #DEFINE_ALIAS
# from .functional.activation import hard_swish   #DEFINE_ALIAS
from .functional.activation import hsigmoid  #DEFINE_ALIAS
# from .functional.activation import leaky_relu   #DEFINE_ALIAS
# from .functional.activation import logsigmoid   #DEFINE_ALIAS
# from .functional.activation import maxout   #DEFINE_ALIAS
# from .functional.activation import prelu   #DEFINE_ALIAS
from .functional.activation import relu  #DEFINE_ALIAS
# from .functional.activation import relu6   #DEFINE_ALIAS
# from .functional.activation import selu   #DEFINE_ALIAS
from .functional.activation import sigmoid  #DEFINE_ALIAS
# from .functional.activation import soft_relu   #DEFINE_ALIAS
# from .functional.activation import softmax   #DEFINE_ALIAS
# from .functional.activation import softplus   #DEFINE_ALIAS
# from .functional.activation import softshrink   #DEFINE_ALIAS
# from .functional.activation import softsign   #DEFINE_ALIAS
# from .functional.activation import swish   #DEFINE_ALIAS
# from .functional.activation import tanh_shrink   #DEFINE_ALIAS
# from .functional.activation import thresholded_relu   #DEFINE_ALIAS
from .functional.activation import log_softmax  #DEFINE_ALIAS
# from .functional.extension import add_position_encoding   #DEFINE_ALIAS
# from .functional.extension import autoincreased_step_counter   #DEFINE_ALIAS
# from .functional.extension import continuous_value_model   #DEFINE_ALIAS
# from .functional.extension import filter_by_instag   #DEFINE_ALIAS
# from .functional.extension import linear_chain_crf   #DEFINE_ALIAS
# from .functional.extension import merge_selected_rows   #DEFINE_ALIAS
# from .functional.extension import multiclass_nms   #DEFINE_ALIAS
# from .functional.extension import polygon_box_transform   #DEFINE_ALIAS
# from .functional.extension import random_crop   #DEFINE_ALIAS
from .functional.extension import row_conv  #DEFINE_ALIAS
# from .functional.extension import rpn_target_assign   #DEFINE_ALIAS
# from .functional.extension import similarity_focus   #DEFINE_ALIAS
# from .functional.extension import target_assign   #DEFINE_ALIAS
# from .functional.extension import temporal_shift   #DEFINE_ALIAS
# from .functional.extension import warpctc   #DEFINE_ALIAS
from .functional.extension import diag_embed  #DEFINE_ALIAS
# from .functional.rnn import gru_unit   #DEFINE_ALIAS
# from .functional.rnn import lstm   #DEFINE_ALIAS
# from .functional.rnn import lstm_unit   #DEFINE_ALIAS
# from .functional.lod import sequence_concat   #DEFINE_ALIAS
# from .functional.lod import sequence_conv   #DEFINE_ALIAS
# from .functional.lod import sequence_enumerate   #DEFINE_ALIAS
# from .functional.lod import sequence_expand_as   #DEFINE_ALIAS
# from .functional.lod import sequence_expand   #DEFINE_ALIAS
# from .functional.lod import sequence_first_step   #DEFINE_ALIAS
# from .functional.lod import sequence_last_step   #DEFINE_ALIAS
# from .functional.lod import sequence_mask   #DEFINE_ALIAS
# from .functional.lod import sequence_pad   #DEFINE_ALIAS
# from .functional.lod import sequence_pool   #DEFINE_ALIAS
# from .functional.lod import sequence_reshape   #DEFINE_ALIAS
# from .functional.lod import sequence_reverse   #DEFINE_ALIAS
# from .functional.lod import sequence_scatter   #DEFINE_ALIAS
# from .functional.lod import sequence_slice   #DEFINE_ALIAS
# from .functional.lod import sequence_softmax   #DEFINE_ALIAS
# from .functional.lod import sequence_unpad   #DEFINE_ALIAS
# from .functional.lod import array_length   #DEFINE_ALIAS
# from .functional.lod import array_read   #DEFINE_ALIAS
# from .functional.lod import array_write   #DEFINE_ALIAS
# from .functional.lod import create_array   #DEFINE_ALIAS
# from .functional.lod import hash   #DEFINE_ALIAS
# from .functional.lod import im2sequence   #DEFINE_ALIAS
# from .functional.lod import lod_append   #DEFINE_ALIAS
# from .functional.lod import lod_reset   #DEFINE_ALIAS
# from .functional.lod import reorder_lod_tensor_by_rank   #DEFINE_ALIAS
# from .functional.lod import tensor_array_to_tensor   #DEFINE_ALIAS
# from .functional.lod import dynamic_gru   #DEFINE_ALIAS
# from .functional.lod import dynamic_lstm   #DEFINE_ALIAS
# from .functional.lod import dynamic_lstmp   #DEFINE_ALIAS
# from .functional.common import dropout   #DEFINE_ALIAS
# from .functional.common import embedding   #DEFINE_ALIAS
# from .functional.common import fc   #DEFINE_ALIAS
# from .functional.common import label_smooth   #DEFINE_ALIAS
# from .functional.common import one_hot   #DEFINE_ALIAS
# from .functional.common import pad   #DEFINE_ALIAS
# from .functional.common import pad_constant_like   #DEFINE_ALIAS
# from .functional.common import pad2d   #DEFINE_ALIAS
# from .functional.common import unfold   #DEFINE_ALIAS
# from .functional.common import bilinear_tensor_product   #DEFINE_ALIAS
# from .functional.common import assign   #DEFINE_ALIAS
# from .functional.common import interpolate   #DEFINE_ALIAS
# from .input import data   #DEFINE_ALIAS
# from .input import Input   #DEFINE_ALIAS
