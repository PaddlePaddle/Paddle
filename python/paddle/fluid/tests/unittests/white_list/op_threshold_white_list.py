# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

NEED_FIX_FP64_CHECK_GRAD_THRESHOLD_OP_LIST = [
    'abs', 'acos', 'add_position_encoding', 'affine_channel', 'asin', 'assign',
    'atan', 'bilinear_interp', 'bilinear_tensor_product', 'bpr_loss',
    'center_loss', 'concat', 'cos', 'crop', 'crop_tensor', 'cross_entropy',
    'cross_entropy2', 'cumsum', 'elementwise_add', 'elementwise_div',
    'elementwise_min', 'elementwise_mul', 'elementwise_pow', 'elementwise_sub',
    'elu', 'exp', 'expand', 'expand_as', 'filter_by_instag', 'flatten',
    'flatten2', 'gather', 'gather_nd', 'gelu', 'grid_sampler', 'group_norm',
    'gru', 'gru_unit', 'hard_shrink', 'hard_swish', 'label_smooth',
    'leaky_relu', 'linear_chain_crf', 'lod_reset', 'log', 'logsigmoid',
    'lookup_table', 'lookup_table_v2', 'lstm', 'lstmp', 'mean', 'mul',
    'multiplex', 'nearest_interp', 'norm', 'pixel_shuffle', 'pool2d', 'pool3d',
    'pow', 'psroi_pool', 'reciprocal', 'reduce_mean', 'reduce_prod',
    'reduce_sum', 'relu', 'relu6', 'reverse', 'roi_align', 'roi_pool', 'rsqrt',
    'scatter_nd_add', 'selu', 'sequence_unpad', 'sin', 'soft_relu',
    'softmax_with_cross_entropy', 'softplus', 'softshrink', 'softsign',
    'space_to_depth', 'sqrt', 'square', 'stanh', 'strided_slice', 'sum',
    'swish', 'tanh_shrink', 'thresholded_relu'
]
