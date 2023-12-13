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

#pragma once
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/infermeta/spmd_rules/cast.h"
#include "paddle/phi/infermeta/spmd_rules/concat.h"
#include "paddle/phi/infermeta/spmd_rules/cross_entropy_with_softmax.h"
#include "paddle/phi/infermeta/spmd_rules/default_data_parallel.h"
#include "paddle/phi/infermeta/spmd_rules/elementwise.h"
#include "paddle/phi/infermeta/spmd_rules/embedding.h"
#include "paddle/phi/infermeta/spmd_rules/flash_attention.h"
#include "paddle/phi/infermeta/spmd_rules/flatten.h"
#include "paddle/phi/infermeta/spmd_rules/full_like.h"
#include "paddle/phi/infermeta/spmd_rules/fused_rope.h"
#include "paddle/phi/infermeta/spmd_rules/layer_norm.h"
#include "paddle/phi/infermeta/spmd_rules/matmul.h"
#include "paddle/phi/infermeta/spmd_rules/numel.h"
#include "paddle/phi/infermeta/spmd_rules/optimizer.h"
#include "paddle/phi/infermeta/spmd_rules/pow.h"
#include "paddle/phi/infermeta/spmd_rules/reduction.h"
#include "paddle/phi/infermeta/spmd_rules/replicated.h"
#include "paddle/phi/infermeta/spmd_rules/reshape.h"
#include "paddle/phi/infermeta/spmd_rules/scale.h"
#include "paddle/phi/infermeta/spmd_rules/slice.h"
#include "paddle/phi/infermeta/spmd_rules/softmax.h"
#include "paddle/phi/infermeta/spmd_rules/split.h"
#include "paddle/phi/infermeta/spmd_rules/squeeze.h"
#include "paddle/phi/infermeta/spmd_rules/stack.h"
#include "paddle/phi/infermeta/spmd_rules/transpose.h"
#include "paddle/phi/infermeta/spmd_rules/triu.h"
#include "paddle/phi/infermeta/spmd_rules/unsqueeze.h"
#include "paddle/phi/infermeta/spmd_rules/where.h"

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

namespace phi {
namespace distributed {

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

// TODO(pkuzyc): add multiary elementwise rule

// reduction rule
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

// layer_norm
PD_REGISTER_SPMD_RULE(
    layer_norm,
    PD_INFER_SPMD(phi::distributed::LayerNormInferSpmd),
    PD_INFER_SPMD(phi::distributed::LayerNormInferSpmdReverse));

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

PD_REGISTER_SPMD_RULE(concat,
                      PD_INFER_SPMD(phi::distributed::ConcatInferSpmd),
                      PD_INFER_SPMD(phi::distributed::ConcatInferSpmdReverse));

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

// cross_entropy_with_softmax
PD_REGISTER_SPMD_RULE(
    cross_entropy_with_softmax,
    PD_INFER_SPMD(phi::distributed::CrossEntropyWithSoftmaxInferSpmd),
    PD_INFER_SPMD(phi::distributed::CrossEntropyWithSoftmaxInferSpmdReverse));
PD_REGISTER_SPMD_RULE(
    softmax_with_cross_entropy,
    PD_INFER_SPMD(phi::distributed::CrossEntropyWithSoftmaxInferSpmd),
    PD_INFER_SPMD(phi::distributed::CrossEntropyWithSoftmaxInferSpmdReverse));

}  // namespace distributed
}  // namespace phi
