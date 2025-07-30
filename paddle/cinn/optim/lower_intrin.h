// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <set>
#include <string>

#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace optim {

static const std::set<std::string> kIntrinsicCalls{
    {"exp",         "exp2",        "sqrt",       "log",         "log2",
     "log10",       "floor",       "ceil",       "round",       "trunc",
     "cos",         "cosh",        "tan",        "tanh",        "sin",
     "sinh",        "fabs",        "isnan",      "isfinite",    "isinf",
     "left_shift",  "right_shift", "bitwise_or", "bitwise_and", "bitwise_xor",
     "bitwise_not", "fma",         "rsqrt",      "mod",         "pow"}};

/**
 * Map the Call nodes to llvm intrinsic.
 *
 * This will rename the external call with the function in different backends.
 *
 * Notes: only support cpu currently.
 */
void LowerIntrin(ir::Expr *expr, Target target);

}  // namespace optim
}  // namespace cinn
