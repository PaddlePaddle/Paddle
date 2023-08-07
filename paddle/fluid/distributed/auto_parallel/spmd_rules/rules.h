// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/common.h"
#include "paddle/fluid/distributed/auto_parallel/spmd_rules/cross_entropy_with_softmax_spmd_rule.h"
#include "paddle/fluid/distributed/auto_parallel/spmd_rules/elementwise_spmd_rule.h"
#include "paddle/fluid/distributed/auto_parallel/spmd_rules/embedding_spmd_rule.h"
#include "paddle/fluid/distributed/auto_parallel/spmd_rules/layer_norm_spmd_rule.h"
#include "paddle/fluid/distributed/auto_parallel/spmd_rules/matmul_spmd_rule.h"
#include "paddle/fluid/distributed/auto_parallel/spmd_rules/reduction_spmd_rule.h"
#include "paddle/fluid/distributed/auto_parallel/spmd_rules/replicated_spmd_rule.h"
#include "paddle/fluid/distributed/auto_parallel/spmd_rules/softmax_spmd_rule.h"
#include "paddle/fluid/distributed/auto_parallel/spmd_rules/split_spmd_rule.h"
#include "paddle/fluid/distributed/auto_parallel/spmd_rules/transpose_spmd_rule.h"

// TODO(ljz) Automatic this process in cmake file.
namespace paddle {
namespace distributed {
namespace auto_parallel {

// matmul rule
REGISTER_SPMD_RULE(matmul, MatmulSPMDRule);

// reduction rules
REGISTER_SPMD_RULE(all, ReductionSPMDRule);
REGISTER_SPMD_RULE(amax, ReductionSPMDRule);
REGISTER_SPMD_RULE(amin, ReductionSPMDRule);
REGISTER_SPMD_RULE(any, ReductionSPMDRule);
REGISTER_SPMD_RULE(frobenius_norm, ReductionSPMDRule);
REGISTER_SPMD_RULE(max, ReductionSPMDRule);
REGISTER_SPMD_RULE(mean, ReductionSPMDRule);
REGISTER_SPMD_RULE(min, ReductionSPMDRule);
REGISTER_SPMD_RULE(prod, ReductionSPMDRule);
REGISTER_SPMD_RULE(sum, ReductionSPMDRule);

// elementwise rule
REGISTER_SPMD_RULE(add, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(assign, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(assign_out_, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(divide, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(elementwise_pow, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(exponential_, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(floor_divide, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(fmin, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(hardswish, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(heaviside, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(maximum, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(minimum, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(mish, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(multiply, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(relu6, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(remainder, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(subtract, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(swish, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(acos, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(acosh, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(asin, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(asinh, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(atan, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(atanh, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(bernoulli, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(bitwise_and, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(bitwise_not, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(bitwise_or, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(bitwise_xor, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(ceil, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(celu, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(clip, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(conj, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(cos, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(cosh, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(det, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(digamma, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(elu, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(erf, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(erfinv, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(exp, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(expm1, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(fill, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(floor, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(fmax, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(gelu, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(hardshrink, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(hardsigmoid, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(hardtanh, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(label_smooth, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(leaky_relu, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(lgamma, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(log, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(log10, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(log1p, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(log2, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(logical_and, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(logical_not, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(logical_or, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(logical_xor, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(logit, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(logsigmoid, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(poisson, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(pow, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(reciprocal, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(relu, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(round, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(rsqrt, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(scale, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(selu, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(sigmoid, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(sign, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(silu, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(sin, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(sinh, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(softplus, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(softshrink, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(softsign, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(sqrt, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(square, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(stanh, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(tan, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(tanh, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(tanh_shrink, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(thresholded_relu, ElementwiseSPMDRule);
REGISTER_SPMD_RULE(trunc, ElementwiseSPMDRule);

// layer_norm rule
REGISTER_SPMD_RULE(layer_norm, LayerNormSPMDRule);

// replicated rule
REGISTER_SPMD_RULE(replicated, ReplicatedSPMDRule);

// embedding rule
REGISTER_SPMD_RULE(embedding, EmbeddingSPMDRule);
REGISTER_SPMD_RULE(lookup_table_v2, EmbeddingSPMDRule);

// softmax rule
REGISTER_SPMD_RULE(softmax, SoftmaxSPMDRule);
REGISTER_SPMD_RULE(log_softmax, SoftmaxSPMDRule);

// cross_entropy_with_softmax
REGISTER_SPMD_RULE(cross_entropy_with_softmax, CrossEntropyWithSoftmaxSPMDRule);
REGISTER_SPMD_RULE(softmax_with_cross_entropy, CrossEntropyWithSoftmaxSPMDRule);

// split rule
REGISTER_SPMD_RULE(split, SplitSPMDRule);
REGISTER_SPMD_RULE(split_with_num, SplitSPMDRule);

// transpose rule
REGISTER_SPMD_RULE(transpose, TransposeSPMDRule);

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
