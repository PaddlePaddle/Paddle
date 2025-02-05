/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/spmd_rules/rules.h"

/**
 * Design Notes:
 *
 * 1. SPMD info is the special meta info of DistTensor, so we put Spmd infer
 * functions in `infermeta` directory.
 *
 * 2. Since the infer functions of Spmd forward and backward are closely related
 * and need to be registered together, we manage them together in one file.
 *
 * 3. SPMD rules are much smaller than infermeta function, and we manage files
 * in operator units.
 *
 * 4. The previous registration used some compile-time regular matching methods,
 * which was less flexible, and the registration of SPMD rules here is declare
 * directly in the header file
 */

namespace phi::distributed {

// matmul rule
PD_REGISTER_SPMD_RULE(matmul,
                      PD_INFER_SPMD(phi::distributed::MatmulInferSpmd),
                      PD_INFER_SPMD(phi::distributed::MatmulInferSpmdReverse));
PD_REGISTER_SPMD_RULE(matmul_v2,  // static mode
                      PD_INFER_SPMD(phi::distributed::MatmulInferSpmd),
                      PD_INFER_SPMD(phi::distributed::MatmulInferSpmdReverse));

PD_REGISTER_SPMD_RULE(
    elementwise_unary,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));

PD_REGISTER_SPMD_RULE(
    elementwise_binary,
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmdReverse));

// default data parallel rule
PD_REGISTER_SPMD_RULE(
    default_data_parallel,
    PD_INFER_SPMD(phi::distributed::DefaultDataParallelInferSpmd),
    PD_INFER_SPMD(phi::distributed::DefaultDataParallelInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    default_,
    PD_INFER_SPMD(phi::distributed::DefaultDataParallelInferSpmd),
    PD_INFER_SPMD(phi::distributed::DefaultDataParallelInferSpmdReverse));

// fused rope
PD_REGISTER_SPMD_RULE(
    fused_rotary_position_embedding,
    PD_INFER_SPMD(phi::distributed::FusedRopeInferSpmd),
    PD_INFER_SPMD(phi::distributed::FusedRopeInferSpmdReverse));

// replicated rule /* for unittest */
PD_REGISTER_SPMD_RULE(
    replicated,
    PD_INFER_SPMD(phi::distributed::ReplicatedInferSpmd),
    PD_INFER_SPMD(phi::distributed::ReplicatedInferSpmdReverse));

// unsqueeze rule
PD_REGISTER_SPMD_RULE(
    unsqueeze,
    PD_INFER_SPMD(phi::distributed::UnsqueezeInferSpmd),
    PD_INFER_SPMD(phi::distributed::UnsqueezeInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    unsqueeze2,
    PD_INFER_SPMD(phi::distributed::UnsqueezeInferSpmd),
    PD_INFER_SPMD(phi::distributed::UnsqueezeInferSpmdReverse));

// elementwise unary rule
PD_REGISTER_SPMD_RULE(
    assign,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    hardswish,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    mish,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    relu6,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    swish,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    acos,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    acosh,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    asin,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    asinh,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    atan,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    atanh,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    bernoulli,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    bitwise_not,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    ceil,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    celu,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    clip,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    conj,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    conv2d,
    PD_INFER_SPMD(phi::distributed::Conv2dInferSpmdBase),
    PD_INFER_SPMD(phi::distributed::Conv2dInferSpmdReverseBase));
PD_REGISTER_SPMD_RULE(
    cos,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    cosh,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    digamma,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    elu,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    erf,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    erfinv,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    exp,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    expm1,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    fill,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    floor,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    gelu,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    hardshrink,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    hardsigmoid,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    hardtanh,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    label_smooth,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    leaky_relu,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    lgamma,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    log,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    log10,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    log1p,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    log2,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    logical_not,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    logit,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    logsigmoid,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    poisson,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    pow,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    reciprocal,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    relu,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    round,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    rsqrt,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    scale,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    selu,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    sigmoid,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    sign,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    silu,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    sin,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    sinh,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    softplus,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    softshrink,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    softsign,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    sqrt,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    square,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    stanh,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    tan,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    tanh,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    tanh_shrink,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    thresholded_relu,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    trunc,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    dropout,
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseUnaryInferSpmdReverse));

PD_REGISTER_SPMD_RULE(
    fused_dropout_add,
    PD_INFER_SPMD(phi::distributed::FusedDropoutAddSpmdBase),
    PD_INFER_SPMD(phi::distributed::FusedDropoutAddSpmdReverseBase));

// elementwise binary rule
PD_REGISTER_SPMD_RULE(
    add,
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    elementwise_add,
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    divide,
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    elementwise_div,
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    elementwise_pow,
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    floor_divide,
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    fmin,
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    heaviside,
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    maximum,
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    minimum,
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    multiply,
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    elementwise_mul,
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    remainder,
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    subtract,
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    bitwise_and,
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    bitwise_or,
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    bitwise_xor,
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    fmax,
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    logical_and,
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    logical_or,
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    logical_xor,
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    not_equal,
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    greater_than,
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    less_than,
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmd),
    PD_INFER_SPMD(phi::distributed::ElementwiseBinaryInferSpmdReverse));
PD_REGISTER_SPMD_RULE(swiglu,
                      PD_INFER_SPMD(phi::distributed::SwiGLUInferSpmd),
                      PD_INFER_SPMD(phi::distributed::SwiGLUInferSpmdReverse));
// TODO(pkuzyc): add multiary elementwise rule

// reduction rule
PD_REGISTER_SPMD_RULE(
    reduce_base,
    PD_INFER_SPMD(phi::distributed::ReductionInferSpmdBase),
    PD_INFER_SPMD(phi::distributed::ReductionInferSpmdReverse));

PD_REGISTER_SPMD_RULE(
    all,
    PD_INFER_SPMD(phi::distributed::ReductionInferSpmd),
    PD_INFER_SPMD(phi::distributed::ReductionInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    amax,
    PD_INFER_SPMD(phi::distributed::ReductionInferSpmd),
    PD_INFER_SPMD(phi::distributed::ReductionInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    amin,
    PD_INFER_SPMD(phi::distributed::ReductionInferSpmd),
    PD_INFER_SPMD(phi::distributed::ReductionInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    any,
    PD_INFER_SPMD(phi::distributed::ReductionInferSpmd),
    PD_INFER_SPMD(phi::distributed::ReductionInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    frobenius_norm,
    PD_INFER_SPMD(phi::distributed::ReductionInferSpmd),
    PD_INFER_SPMD(phi::distributed::ReductionInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    max,
    PD_INFER_SPMD(phi::distributed::ReductionInferSpmd),
    PD_INFER_SPMD(phi::distributed::ReductionInferSpmdReverse));

PD_REGISTER_SPMD_RULE(
    reduce_max,
    PD_INFER_SPMD(phi::distributed::ReductionInferSpmd),
    PD_INFER_SPMD(phi::distributed::ReductionInferSpmdReverse));

PD_REGISTER_SPMD_RULE(
    min,
    PD_INFER_SPMD(phi::distributed::ReductionInferSpmd),
    PD_INFER_SPMD(phi::distributed::ReductionInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    prod,
    PD_INFER_SPMD(phi::distributed::ReductionInferSpmd),
    PD_INFER_SPMD(phi::distributed::ReductionInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    sum,
    PD_INFER_SPMD(phi::distributed::ReductionInferSpmd),
    PD_INFER_SPMD(phi::distributed::ReductionInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    reduce_sum,  // static
    PD_INFER_SPMD(phi::distributed::ReductionInferSpmd),
    PD_INFER_SPMD(phi::distributed::ReductionInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    squared_l2_norm,
    PD_INFER_SPMD(phi::distributed::ReductionInferSpmd),
    PD_INFER_SPMD(phi::distributed::ReductionInferSpmdReverse));

// layer_norm
PD_REGISTER_SPMD_RULE(
    layer_norm,
    PD_INFER_SPMD(phi::distributed::LayerNormInferSpmd),
    PD_INFER_SPMD(phi::distributed::LayerNormInferSpmdReverse));

// fused_rms_norm
// NOTE(ZHIQIU): Temporally register fused_rms_norm rule,
// this is not for rms_norm kernel, but for the custom kernel
// 'fused_rms_norm' in PaddleNLP.
// It will be no longer needed when the PIR-AutoParallel project
// is finished.
PD_REGISTER_SPMD_RULE(fused_rms_norm,
                      PD_INFER_SPMD(phi::distributed::RmsNormInferSpmd),
                      PD_INFER_SPMD(phi::distributed::RmsNormInferSpmdReverse));

PD_REGISTER_SPMD_RULE(
    flash_attention,
    PD_INFER_SPMD(phi::distributed::FlashAttInferSpmdStatic),
    PD_INFER_SPMD(phi::distributed::FlashAttInferSpmdReverse));

// reshape rule
PD_REGISTER_SPMD_RULE(reshape,
                      PD_INFER_SPMD(phi::distributed::ReshapeInferSpmd),
                      PD_INFER_SPMD(phi::distributed::ReshapeInferSpmdReverse));
PD_REGISTER_SPMD_RULE(reshape2,
                      PD_INFER_SPMD(phi::distributed::ReshapeInferSpmd),
                      PD_INFER_SPMD(phi::distributed::ReshapeInferSpmdReverse));

// squeeze rule
PD_REGISTER_SPMD_RULE(squeeze,
                      PD_INFER_SPMD(phi::distributed::SqueezeInferSpmd),
                      PD_INFER_SPMD(phi::distributed::SqueezeInferSpmdReverse));
// flatten rule
PD_REGISTER_SPMD_RULE(flatten,
                      PD_INFER_SPMD(phi::distributed::FlattenInferSpmd),
                      PD_INFER_SPMD(phi::distributed::FlattenInferSpmdReverse));

// embedding rule
PD_REGISTER_SPMD_RULE(
    embedding,
    PD_INFER_SPMD(phi::distributed::EmbeddingInferSpmd),
    PD_INFER_SPMD(phi::distributed::EmbeddingInferSpmdReverse));
PD_REGISTER_SPMD_RULE(c_embedding,
                      PD_INFER_SPMD(phi::distributed::CEmbeddingInferSpmd),
                      PD_INFER_SPMD(phi::distributed::CEmbeddingGradInferSpmd));
PD_REGISTER_SPMD_RULE(
    lookup_table_v2,
    PD_INFER_SPMD(phi::distributed::EmbeddingInferSpmd),
    PD_INFER_SPMD(phi::distributed::EmbeddingInferSpmdReverse));

// split rule
PD_REGISTER_SPMD_RULE(split,
                      PD_INFER_SPMD(phi::distributed::SplitInferSpmd),
                      PD_INFER_SPMD(phi::distributed::SplitInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    split_with_num,
    PD_INFER_SPMD(phi::distributed::SplitWithNumInferSpmd),
    PD_INFER_SPMD(phi::distributed::SplitWithNumInferSpmdReverse));

// slice rule
PD_REGISTER_SPMD_RULE(slice,
                      PD_INFER_SPMD(phi::distributed::SliceInferSpmd),
                      PD_INFER_SPMD(phi::distributed::SliceInferSpmdReverse));

PD_REGISTER_SPMD_RULE(
    strided_slice,
    PD_INFER_SPMD(phi::distributed::StridedSliceInferSpmd),
    PD_INFER_SPMD(phi::distributed::StridedSliceGradInferSpmd));

PD_REGISTER_SPMD_RULE(concat,
                      PD_INFER_SPMD(phi::distributed::ConcatInferSpmd),
                      PD_INFER_SPMD(phi::distributed::ConcatInferSpmdReverse));

PD_REGISTER_SPMD_RULE(stack,
                      PD_INFER_SPMD(phi::distributed::StackInferSpmd),
                      PD_INFER_SPMD(phi::distributed::StackInferSpmdReverse));

// transpose rule
PD_REGISTER_SPMD_RULE(
    transpose,
    PD_INFER_SPMD(phi::distributed::TransposeInferSpmd),
    PD_INFER_SPMD(phi::distributed::TransposeInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    transpose2,
    PD_INFER_SPMD(phi::distributed::TransposeInferSpmd),
    PD_INFER_SPMD(phi::distributed::TransposeInferSpmdReverse));

// softmax rule
PD_REGISTER_SPMD_RULE(softmax,
                      PD_INFER_SPMD(phi::distributed::SoftmaxInferSpmd),
                      PD_INFER_SPMD(phi::distributed::SoftmaxInferSpmdReverse));

PD_REGISTER_SPMD_RULE(log_softmax,
                      PD_INFER_SPMD(phi::distributed::SoftmaxInferSpmd),
                      PD_INFER_SPMD(phi::distributed::SoftmaxInferSpmdReverse));

PD_REGISTER_SPMD_RULE(where,
                      PD_INFER_SPMD(phi::distributed::WhereInferSpmd),
                      PD_INFER_SPMD(phi::distributed::WhereInferSpmdReverse));

PD_REGISTER_SPMD_RULE(triu,
                      PD_INFER_SPMD(phi::distributed::TriuInferSpmd),
                      PD_INFER_SPMD(phi::distributed::TriuInferSpmdReverse));

PD_REGISTER_SPMD_RULE(
    tril_triu,
    PD_INFER_SPMD(phi::distributed::TrilTriuInferSpmd),
    PD_INFER_SPMD(phi::distributed::TrilTriuInferSpmdReverse));

PD_REGISTER_SPMD_RULE(tile,
                      PD_INFER_SPMD(phi::distributed::TileInferSpmd),
                      PD_INFER_SPMD(phi::distributed::TileInferSpmdReverse));

// cross_entropy_with_softmax
PD_REGISTER_SPMD_RULE(
    cross_entropy_with_softmax,
    PD_INFER_SPMD(phi::distributed::CrossEntropyWithSoftmaxInferSpmdStatic),
    PD_INFER_SPMD(phi::distributed::CrossEntropyWithSoftmaxInferSpmdReverse));

PD_REGISTER_SPMD_RULE(
    softmax_with_cross_entropy,
    PD_INFER_SPMD(phi::distributed::CrossEntropyWithSoftmaxInferSpmdStatic),
    PD_INFER_SPMD(phi::distributed::CrossEntropyWithSoftmaxInferSpmdReverse));

PD_REGISTER_SPMD_RULE(
    c_softmax_with_cross_entropy,
    PD_INFER_SPMD(phi::distributed::CSoftmaxWithCrossEntropyInferSpmd));

PD_REGISTER_SPMD_RULE(
    c_softmax_with_multi_label_cross_entropy,
    PD_INFER_SPMD(
        phi::distributed::CSoftmaxWithMultiLabelCrossEntropyInferSpmd));

// fused_linear_param_grad_add got no reverse infer spmd rule
PD_REGISTER_SPMD_RULE(
    fused_linear_param_grad_add,
    PD_INFER_SPMD(phi::distributed::FusedLinearParamGradAddInferSpmd),
    PD_INFER_SPMD(
        phi::distributed::FusedLinearParamGradAddInferSpmdFakeReverse));

PD_REGISTER_SPMD_RULE(
    expand_as,
    PD_INFER_SPMD(phi::distributed::ExpandAsInferSpmd),
    PD_INFER_SPMD(phi::distributed::ExpandAsInferSpmdReverse));

PD_REGISTER_SPMD_RULE(
    expand_as_v2,
    PD_INFER_SPMD(phi::distributed::ExpandAsInferSpmd),
    PD_INFER_SPMD(phi::distributed::ExpandAsInferSpmdReverse));

// scatter
PD_REGISTER_SPMD_RULE(scatter,
                      PD_INFER_SPMD(phi::distributed::ScatterInferSpmd),
                      PD_INFER_SPMD(phi::distributed::ScatterInferSpmdReverse));

// scatter_nd_add
PD_REGISTER_SPMD_RULE(
    scatter_nd_add,
    PD_INFER_SPMD(phi::distributed::ScatterNdAddInferSpmd),
    PD_INFER_SPMD(phi::distributed::ScatterNdAddInferSpmdReverse));

// gather
PD_REGISTER_SPMD_RULE(
    gather,
    PD_INFER_SPMD(phi::distributed::GatherInferSpmdBase),
    PD_INFER_SPMD(phi::distributed::GatherInferSpmdReverseBase));

PD_REGISTER_SPMD_RULE(
    gather_nd,
    PD_INFER_SPMD(phi::distributed::GatherNdInferSpmd),
    PD_INFER_SPMD(phi::distributed::GatherNdInferSpmdReverse));

// one_hot
PD_REGISTER_SPMD_RULE(one_hot,
                      PD_INFER_SPMD(phi::distributed::OneHotInferSpmd),
                      PD_INFER_SPMD(phi::distributed::OneHotInferSpmdReverse));

PD_REGISTER_SPMD_RULE(cumsum,
                      PD_INFER_SPMD(phi::distributed::CumSumInferSpmd),
                      PD_INFER_SPMD(phi::distributed::CumSumInferSpmdReverse));

// argmax
PD_REGISTER_SPMD_RULE(
    argmax,
    PD_INFER_SPMD(phi::distributed::ArgMaxInferSpmdBase),
    PD_INFER_SPMD(phi::distributed::ArgMaxInferSpmdReverseBase));

// unbind
PD_REGISTER_SPMD_RULE(unbind,
                      PD_INFER_SPMD(phi::distributed::UnbindInferSpmd),
                      PD_INFER_SPMD(phi::distributed::UnbindInferSpmdReverse));

// logsumexp
PD_REGISTER_SPMD_RULE(
    logsumexp,
    PD_INFER_SPMD(phi::distributed::LogSumExpInferSpmd),
    PD_INFER_SPMD(phi::distributed::LogSumExpInferSpmdReverse));

// p_norm
PD_REGISTER_SPMD_RULE(p_norm,
                      PD_INFER_SPMD(phi::distributed::PNormInferSpmd),
                      PD_INFER_SPMD(phi::distributed::PNormInferSpmdReverse));

// pad
PD_REGISTER_SPMD_RULE(pad,
                      PD_INFER_SPMD(phi::distributed::PadInferSpmd),
                      PD_INFER_SPMD(phi::distributed::PadGradInferSpmd));

// nonzero
PD_REGISTER_SPMD_RULE(nonzero,
                      PD_INFER_SPMD(phi::distributed::NonZeroInferSpmd),
                      PD_INFER_SPMD(phi::distributed::NonZeroInferSpmdReverse));

// add_n
PD_REGISTER_SPMD_RULE(add_n, PD_INFER_SPMD(phi::distributed::AddNInferSpmd));
}  // namespace phi::distributed
