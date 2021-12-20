// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// clang-format off

#pragma once

// Ops from AiGraphcoreOpset1
OP_DECL(popart_groupnormalization_v2, aiGraphcoreOpset.groupnormalization, ARG(INT,num_groups) ARG(FLOAT,epsilon) ) // NOLINT
OP_DECL(popart_subsample_v2, aiGraphcoreOpset.subsample, ARG(INT_VEC,strides) ) // NOLINT
OP_DECL(popart_nop_v2, aiGraphcoreOpset.nop, NONE) // NOLINT
OP_DECL(popart_scale_v2, aiGraphcoreOpset.scale, ARG(FLOAT,scale) ) // NOLINT
OP_DECL(popart_scaledadd_v2, aiGraphcoreOpset.scaledadd, ARG(FLOAT,scale0) ARG(FLOAT,scale1) ) // NOLINT
OP_DECL(popart_gelu_v2, aiGraphcoreOpset.gelu, NONE) // NOLINT
OP_DECL(popart_detach_v2, aiGraphcoreOpset.detach, NONE) // NOLINT
OP_DECL(popart_depthtospace_v2, aiGraphcoreOpset.depthtospace, ARG(INT,blocksize) ARG(STRING,mode) ) // NOLINT
OP_DECL(popart_round_v2, aiGraphcoreOpset.round, NONE) // NOLINT
OP_DECL(popart_dynamicslice_v2, aiGraphcoreOpset.dynamicslice, ARG(INT_VEC,axes) ARG(INT_VEC,sizes) ARG(INT,noOverlap) ) // NOLINT
OP_DECL(popart_dynamicupdate_v2, aiGraphcoreOpset.dynamicupdate, ARG(INT_VEC,axes) ARG(INT_VEC,sizes) ARG(INT,noOverlap) ) // NOLINT
OP_DECL(popart_dynamiczero_v2, aiGraphcoreOpset.dynamiczero, ARG(INT_VEC,axes) ARG(INT_VEC,sizes) ) // NOLINT
OP_DECL(popart_dynamicadd_v2, aiGraphcoreOpset.dynamicadd, ARG(INT_VEC,axes) ARG(INT_VEC,sizes) ) // NOLINT
OP_DECL(popart_sequenceslice_v2, aiGraphcoreOpset.sequenceslice, ARG(INT,zeroUnused) ) // NOLINT
OP_DECL(popart_replicatedallreduce_v2, aiGraphcoreOpset.replicatedallreduce, OPT_ARG(INT_VEC,commGroup) ) // NOLINT
OP_DECL(popart_ctcbeamsearchdecoder_v2, aiGraphcoreOpset.ctcbeamsearchdecoder, ARG(INT,blank) ARG(INT,beamWidth) ARG(INT,topPaths) ) // NOLINT
OP_DECL(popart_shapeddropout_v2, aiGraphcoreOpset.shapeddropout, ARG(INT_VEC,shape) ARG(FLOAT,ratio) ) // NOLINT
OP_DECL(popart_atan2_v2, aiGraphcoreOpset.atan2, NONE) // NOLINT
OP_DECL(popart_expm1_v2, aiGraphcoreOpset.expm1, NONE) // NOLINT
OP_DECL(popart_log1p_v2, aiGraphcoreOpset.log1p, NONE) // NOLINT
OP_DECL(popart_fmod_v2, aiGraphcoreOpset.fmod, NONE) // NOLINT
OP_DECL(popart_remainder_v2, aiGraphcoreOpset.remainder, NONE) // NOLINT
OP_DECL(popart_reverse_v2, aiGraphcoreOpset.reverse, ARG(INT_VEC,dimensions) ) // NOLINT
OP_DECL(popart_bitwisenot_v2, aiGraphcoreOpset.bitwisenot, NONE) // NOLINT
OP_DECL(popart_bitwiseand_v2, aiGraphcoreOpset.bitwiseand, NONE) // NOLINT
OP_DECL(popart_bitwiseor_v2, aiGraphcoreOpset.bitwiseor, NONE) // NOLINT
OP_DECL(popart_bitwisexor_v2, aiGraphcoreOpset.bitwisexor, NONE) // NOLINT
OP_DECL(popart_bitwisexnor_v2, aiGraphcoreOpset.bitwisexnor, NONE) // NOLINT
OP_DECL(popart_reducemedian_v2, aiGraphcoreOpset.reducemedian, OPT_ARG(INT_VEC,axes) ARG(INT,keepdims) ) // NOLINT
// Ops from AiOnnxOpset11
OP_DECL(popart_argmax, aiOnnxOpset.argmax, ARG(INT,axis) ARG(INT,keepdims) ) // NOLINT
OP_DECL(popart_argmin, aiOnnxOpset.argmin, ARG(INT,axis) ARG(INT,keepdims) ) // NOLINT
OP_DECL(popart_averagepool, aiOnnxOpset.averagepool, ARG(INT_VEC,kernel_shape) ARG(INT,ceil_mode) ARG(INT,count_include_pad) ARG(INT_VEC,pads) ARG(INT_VEC,strides) ) // NOLINT
OP_DECL(popart_bitshift, aiOnnxOpset.bitshift, ARG(STRING,direction) ) // NOLINT
OP_DECL(popart_clip, aiOnnxOpset.clip, NONE) // NOLINT
OP_DECL(popart_compress, aiOnnxOpset.compress, OPT_ARG(INT,axis) ) // NOLINT
OP_DECL(popart_concat, aiOnnxOpset.concat, ARG(INT,axis) ) // NOLINT
OP_DECL(popart_concatfromsequence, aiOnnxOpset.concatfromsequence, ARG(INT,axis) ARG(INT,new_axis) ) // NOLINT
OP_DECL(popart_conv, aiOnnxOpset.conv, ARG(INT_VEC,dilations) ARG(INT,group) ARG(INT_VEC,kernel_shape) ARG(INT_VEC,pads) ARG(INT_VEC,strides) ) // NOLINT
OP_DECL(popart_convtranspose, aiOnnxOpset.convtranspose, ARG(INT_VEC,dilations) ARG(INT,group) ARG(INT_VEC,kernel_shape) ARG(INT_VEC,output_padding) ARG(INT_VEC,output_shape) ARG(INT_VEC,pads) ARG(INT_VEC,strides) ) // NOLINT
OP_DECL(popart_cumsum, aiOnnxOpset.cumsum, ARG(INT,exclusive) ARG(INT,reverse) ) // NOLINT
OP_DECL(popart_depthtospace, aiOnnxOpset.depthtospace, ARG(INT,blocksize) ARG(STRING,mode) ) // NOLINT
OP_DECL(popart_det, aiOnnxOpset.det, NONE) // NOLINT
OP_DECL(popart_dynamicquantizelinear, aiOnnxOpset.dynamicquantizelinear, NONE) // NOLINT
OP_DECL(popart_equal, aiOnnxOpset.equal, NONE) // NOLINT
OP_DECL(popart_flatten, aiOnnxOpset.flatten, ARG(INT,axis) ) // NOLINT
OP_DECL(popart_gather, aiOnnxOpset.gather, ARG(INT,axis) ) // NOLINT
OP_DECL(popart_gatherelements, aiOnnxOpset.gatherelements, ARG(INT,axis) ) // NOLINT
OP_DECL(popart_gathernd, aiOnnxOpset.gathernd, NONE) // NOLINT
OP_DECL(popart_gemm, aiOnnxOpset.gemm, ARG(FLOAT,alpha) ARG(FLOAT,beta) ARG(INT,transA) ARG(INT,transB) ) // NOLINT
OP_DECL(popart_hardmax, aiOnnxOpset.hardmax, ARG(INT,axis) ) // NOLINT
OP_DECL(popart_logsoftmax, aiOnnxOpset.logsoftmax, ARG(INT,axis) ) // NOLINT
OP_DECL(popart_lppool, aiOnnxOpset.lppool, ARG(INT_VEC,kernel_shape) ARG(INT,p) ARG(INT_VEC,pads) ARG(INT_VEC,strides) ) // NOLINT
OP_DECL(popart_maxpool, aiOnnxOpset.maxpool, ARG(INT,num_outputs) ARG(INT_VEC,kernel_shape) ARG(INT,ceil_mode) ARG(INT_VEC,dilations) ARG(INT_VEC,pads) ARG(INT,storage_order) ARG(INT_VEC,strides) ) // NOLINT
OP_DECL(popart_maxunpool, aiOnnxOpset.maxunpool, ARG(INT_VEC,kernel_shape) ARG(INT_VEC,pads) ARG(INT_VEC,strides) ) // NOLINT
OP_DECL(popart_nonmaxsuppression, aiOnnxOpset.nonmaxsuppression, ARG(INT,center_point_box) ) // NOLINT
OP_DECL(popart_onehot, aiOnnxOpset.onehot, ARG(INT,axis) ) // NOLINT
OP_DECL(popart_pad, aiOnnxOpset.pad, ARG(STRING,mode) ) // NOLINT
OP_DECL(popart_range, aiOnnxOpset.range, NONE) // NOLINT
OP_DECL(popart_reducel1, aiOnnxOpset.reducel1, OPT_ARG(INT_VEC,axes) ARG(INT,keepdims) ) // NOLINT
OP_DECL(popart_reducel2, aiOnnxOpset.reducel2, OPT_ARG(INT_VEC,axes) ARG(INT,keepdims) ) // NOLINT
OP_DECL(popart_reducelogsum, aiOnnxOpset.reducelogsum, OPT_ARG(INT_VEC,axes) ARG(INT,keepdims) ) // NOLINT
OP_DECL(popart_reducelogsumexp, aiOnnxOpset.reducelogsumexp, OPT_ARG(INT_VEC,axes) ARG(INT,keepdims) ) // NOLINT
OP_DECL(popart_reducemax, aiOnnxOpset.reducemax, OPT_ARG(INT_VEC,axes) ARG(INT,keepdims) ) // NOLINT
OP_DECL(popart_reducemean, aiOnnxOpset.reducemean, OPT_ARG(INT_VEC,axes) ARG(INT,keepdims) ) // NOLINT
OP_DECL(popart_reducemin, aiOnnxOpset.reducemin, OPT_ARG(INT_VEC,axes) ARG(INT,keepdims) ) // NOLINT
OP_DECL(popart_reduceprod, aiOnnxOpset.reduceprod, OPT_ARG(INT_VEC,axes) ARG(INT,keepdims) ) // NOLINT
OP_DECL(popart_reducesum, aiOnnxOpset.reducesum, OPT_ARG(INT_VEC,axes) ARG(INT,keepdims) ) // NOLINT
OP_DECL(popart_reducesumsquare, aiOnnxOpset.reducesumsquare, OPT_ARG(INT_VEC,axes) ARG(INT,keepdims) ) // NOLINT
OP_DECL(popart_resize, aiOnnxOpset.resize, ARG(STRING,coordinate_transformation_mode) ARG(FLOAT,cubic_coeff_a) ARG(INT,exclude_outside) ARG(FLOAT,extrapolation_value) ARG(STRING,mode) ARG(STRING,nearest_mode) ) // NOLINT
OP_DECL(popart_round, aiOnnxOpset.round, NONE) // NOLINT
OP_DECL(popart_scatter, aiOnnxOpset.scatter, ARG(INT,axis) ) // NOLINT
OP_DECL(popart_scatterelements, aiOnnxOpset.scatterelements, ARG(INT,axis) ) // NOLINT
OP_DECL(popart_scatternd, aiOnnxOpset.scatternd, NONE) // NOLINT
OP_DECL(popart_sequenceat, aiOnnxOpset.sequenceat, NONE) // NOLINT
OP_DECL(popart_sequenceconstruct, aiOnnxOpset.sequenceconstruct, NONE) // NOLINT
OP_DECL(popart_sequenceerase, aiOnnxOpset.sequenceerase, NONE) // NOLINT
OP_DECL(popart_sequenceinsert, aiOnnxOpset.sequenceinsert, NONE) // NOLINT
OP_DECL(popart_sequencelength, aiOnnxOpset.sequencelength, NONE) // NOLINT
OP_DECL(popart_slice, aiOnnxOpset.slice, NONE) // NOLINT
OP_DECL(popart_softmax, aiOnnxOpset.softmax, ARG(INT,axis) ) // NOLINT
OP_DECL(popart_split, aiOnnxOpset.split, ARG(INT,num_outputs) ARG(INT,axis) ARG(INT_VEC,split) ) // NOLINT
OP_DECL(popart_splittosequence, aiOnnxOpset.splittosequence, ARG(INT,axis) ARG(INT,keepdims) ) // NOLINT
OP_DECL(popart_squeeze, aiOnnxOpset.squeeze, ARG(INT_VEC,axes) ) // NOLINT
OP_DECL(popart_topk, aiOnnxOpset.topk, ARG(INT,axis) ARG(INT,largest) ARG(INT,sorted) ) // NOLINT
OP_DECL(popart_unique, aiOnnxOpset.unique, ARG(INT,num_outputs) OPT_ARG(INT,axis) ARG(INT,sorted) ) // NOLINT
OP_DECL(popart_unsqueeze, aiOnnxOpset.unsqueeze, ARG(INT_VEC,axes) ) // NOLINT
// Ops from AiOnnxOpset10
OP_DECL(popart_convinteger, aiOnnxOpset.convinteger, ARG(INT_VEC,dilations) ARG(INT,group) ARG(INT_VEC,kernel_shape) ARG(INT_VEC,pads) ARG(INT_VEC,strides) ) // NOLINT
OP_DECL(popart_dequantizelinear, aiOnnxOpset.dequantizelinear, NONE) // NOLINT
OP_DECL(popart_dropout, aiOnnxOpset.dropout, ARG(INT,num_outputs) ARG(FLOAT,ratio) ) // NOLINT
OP_DECL(popart_isinf, aiOnnxOpset.isinf, ARG(INT,detect_negative) ARG(INT,detect_positive) ) // NOLINT
OP_DECL(popart_matmulinteger, aiOnnxOpset.matmulinteger, NONE) // NOLINT
OP_DECL(popart_mod, aiOnnxOpset.mod, ARG(INT,fmod) ) // NOLINT
OP_DECL(popart_qlinearconv, aiOnnxOpset.qlinearconv, ARG(INT_VEC,dilations) ARG(INT,group) ARG(INT_VEC,kernel_shape) ARG(INT_VEC,pads) ARG(INT_VEC,strides) ) // NOLINT
OP_DECL(popart_qlinearmatmul, aiOnnxOpset.qlinearmatmul, NONE) // NOLINT
OP_DECL(popart_quantizelinear, aiOnnxOpset.quantizelinear, NONE) // NOLINT
OP_DECL(popart_reversesequence, aiOnnxOpset.reversesequence, ARG(INT,batch_axis) ARG(INT,time_axis) ) // NOLINT
OP_DECL(popart_roialign, aiOnnxOpset.roialign, ARG(STRING,mode) ARG(INT,output_height) ARG(INT,output_width) ARG(INT,sampling_ratio) ARG(FLOAT,spatial_scale) ) // NOLINT
OP_DECL(popart_thresholdedrelu, aiOnnxOpset.thresholdedrelu, ARG(FLOAT,alpha) ) // NOLINT
OP_DECL(popart_upsample, aiOnnxOpset.upsample, ARG(STRING,mode) ) // NOLINT
// Ops from AiOnnxOpset9
OP_DECL(popart_acosh, aiOnnxOpset.acosh, NONE) // NOLINT
OP_DECL(popart_asinh, aiOnnxOpset.asinh, NONE) // NOLINT
OP_DECL(popart_atanh, aiOnnxOpset.atanh, NONE) // NOLINT
OP_DECL(popart_batchnormalization, aiOnnxOpset.batchnormalization, ARG(INT,num_outputs) ARG(FLOAT,epsilon) ARG(FLOAT,momentum) ) // NOLINT
OP_DECL(popart_cast, aiOnnxOpset.cast, ARG(STRING,to) ) // NOLINT
OP_DECL(popart_cosh, aiOnnxOpset.cosh, NONE) // NOLINT
OP_DECL(popart_erf, aiOnnxOpset.erf, NONE) // NOLINT
OP_DECL(popart_eyelike, aiOnnxOpset.eyelike, OPT_ARG(INT,dtype) ARG(INT,k) ) // NOLINT
OP_DECL(popart_greater, aiOnnxOpset.greater, NONE) // NOLINT
OP_DECL(popart_isnan, aiOnnxOpset.isnan, NONE) // NOLINT
OP_DECL(popart_less, aiOnnxOpset.less, NONE) // NOLINT
OP_DECL(popart_matmul, aiOnnxOpset.matmul, NONE) // NOLINT
OP_DECL(popart_meanvariancenormalization, aiOnnxOpset.meanvariancenormalization, ARG(INT_VEC,axes) ) // NOLINT
OP_DECL(popart_nonzero, aiOnnxOpset.nonzero, NONE) // NOLINT
OP_DECL(popart_prelu, aiOnnxOpset.prelu, NONE) // NOLINT
OP_DECL(popart_shrink, aiOnnxOpset.shrink, ARG(FLOAT,bias) ARG(FLOAT,lambd) ) // NOLINT
OP_DECL(popart_sign, aiOnnxOpset.sign, NONE) // NOLINT
OP_DECL(popart_sinh, aiOnnxOpset.sinh, NONE) // NOLINT
OP_DECL(popart_where, aiOnnxOpset.where, NONE) // NOLINT
// Ops from AiOnnxOpset8
OP_DECL(popart_expand, aiOnnxOpset.expand, NONE) // NOLINT
OP_DECL(popart_max, aiOnnxOpset.max, NONE) // NOLINT
OP_DECL(popart_mean, aiOnnxOpset.mean, NONE) // NOLINT
OP_DECL(popart_min, aiOnnxOpset.min, NONE) // NOLINT
OP_DECL(popart_sum, aiOnnxOpset.sum, NONE) // NOLINT
// Ops from AiOnnxOpset7
OP_DECL(popart_acos, aiOnnxOpset.acos, NONE) // NOLINT
OP_DECL(popart_add, aiOnnxOpset.add, NONE) // NOLINT
OP_DECL(popart_logical_and, aiOnnxOpset.logical_and, NONE) // NOLINT
OP_DECL(popart_asin, aiOnnxOpset.asin, NONE) // NOLINT
OP_DECL(popart_atan, aiOnnxOpset.atan, NONE) // NOLINT
OP_DECL(popart_cos, aiOnnxOpset.cos, NONE) // NOLINT
OP_DECL(popart_div, aiOnnxOpset.div, NONE) // NOLINT
OP_DECL(popart_mul, aiOnnxOpset.mul, NONE) // NOLINT
OP_DECL(popart_multinomial, aiOnnxOpset.multinomial, ARG(INT,dtype) ARG(INT,sample_size) OPT_ARG(FLOAT,seed) ) // NOLINT
OP_DECL(popart_logical_or, aiOnnxOpset.logical_or, NONE) // NOLINT
OP_DECL(popart_pow, aiOnnxOpset.pow, NONE) // NOLINT
OP_DECL(popart_sin, aiOnnxOpset.sin, NONE) // NOLINT
OP_DECL(popart_sub, aiOnnxOpset.sub, NONE) // NOLINT
OP_DECL(popart_tan, aiOnnxOpset.tan, NONE) // NOLINT
OP_DECL(popart_logical_xor, aiOnnxOpset.logical_xor, NONE) // NOLINT
// Ops from AiOnnxOpset6
OP_DECL(popart_abs, aiOnnxOpset.abs, NONE) // NOLINT
OP_DECL(popart_ceil, aiOnnxOpset.ceil, NONE) // NOLINT
OP_DECL(popart_elu, aiOnnxOpset.elu, ARG(FLOAT,alpha) ) // NOLINT
OP_DECL(popart_exp, aiOnnxOpset.exp, NONE) // NOLINT
OP_DECL(popart_floor, aiOnnxOpset.floor, NONE) // NOLINT
OP_DECL(popart_globalaveragepool, aiOnnxOpset.globalaveragepool, NONE) // NOLINT
OP_DECL(popart_globallppool, aiOnnxOpset.globallppool, ARG(INT,p) ) // NOLINT
OP_DECL(popart_globalmaxpool, aiOnnxOpset.globalmaxpool, NONE) // NOLINT
OP_DECL(popart_hardsigmoid, aiOnnxOpset.hardsigmoid, ARG(FLOAT,alpha) ARG(FLOAT,beta) ) // NOLINT
OP_DECL(popart_identity, aiOnnxOpset.identity, NONE) // NOLINT
OP_DECL(popart_instancenormalization, aiOnnxOpset.instancenormalization, ARG(FLOAT,epsilon) ) // NOLINT
OP_DECL(popart_lrn, aiOnnxOpset.lrn, ARG(INT,size) ARG(FLOAT,alpha) ARG(FLOAT,beta) ARG(FLOAT,bias) ) // NOLINT
OP_DECL(popart_leakyrelu, aiOnnxOpset.leakyrelu, ARG(FLOAT,alpha) ) // NOLINT
OP_DECL(popart_log, aiOnnxOpset.log, NONE) // NOLINT
OP_DECL(popart_lpnormalization, aiOnnxOpset.lpnormalization, ARG(INT,axis) ARG(INT,p) ) // NOLINT
OP_DECL(popart_maxroipool, aiOnnxOpset.maxroipool, ARG(INT_VEC,pooled_shape) ARG(FLOAT,spatial_scale) ) // NOLINT
OP_DECL(popart_neg, aiOnnxOpset.neg, NONE) // NOLINT
OP_DECL(popart_logical_not, aiOnnxOpset.logical_not, NONE) // NOLINT
OP_DECL(popart_randomnormallike, aiOnnxOpset.randomnormallike, OPT_ARG(INT,dtype) ARG(FLOAT,mean) ARG(FLOAT,scale) OPT_ARG(FLOAT,seed) ) // NOLINT
OP_DECL(popart_randomuniformlike, aiOnnxOpset.randomuniformlike, OPT_ARG(INT,dtype) ARG(FLOAT,high) ARG(FLOAT,low) OPT_ARG(FLOAT,seed) ) // NOLINT
OP_DECL(popart_reciprocal, aiOnnxOpset.reciprocal, NONE) // NOLINT
OP_DECL(popart_relu, aiOnnxOpset.relu, NONE) // NOLINT
OP_DECL(popart_reshape, aiOnnxOpset.reshape, NONE) // NOLINT
OP_DECL(popart_selu, aiOnnxOpset.selu, ARG(FLOAT,alpha) ARG(FLOAT,gamma) ) // NOLINT
OP_DECL(popart_shape, aiOnnxOpset.shape, NONE) // NOLINT
OP_DECL(popart_sigmoid, aiOnnxOpset.sigmoid, NONE) // NOLINT
OP_DECL(popart_size, aiOnnxOpset.size, NONE) // NOLINT
OP_DECL(popart_softplus, aiOnnxOpset.softplus, NONE) // NOLINT
OP_DECL(popart_softsign, aiOnnxOpset.softsign, NONE) // NOLINT
OP_DECL(popart_spacetodepth, aiOnnxOpset.spacetodepth, ARG(INT,blocksize) ) // NOLINT
OP_DECL(popart_sqrt, aiOnnxOpset.sqrt, NONE) // NOLINT
OP_DECL(popart_tanh, aiOnnxOpset.tanh, NONE) // NOLINT
OP_DECL(popart_tile, aiOnnxOpset.tile, NONE) // NOLINT
OP_DECL(popart_transpose, aiOnnxOpset.transpose, ARG(INT_VEC,perm) ) // NOLINT

// clang-format on
