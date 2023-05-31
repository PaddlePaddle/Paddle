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

#include "paddle/cinn/ir/ir_mutator.h"

namespace cinn {
namespace optim {

/**
 * Vectorize the forloops(For) if its for_type is marked as kVectorize.
 * @param expr
 * @param target
 */
void VectorizeLoops(Expr* expr, const Target& target);

namespace detail {

//! Vecorize the \p expr by making the \p var has \p lanes lanes.
void Vectorize(Var var, int lanes, Expr* expr);

}  // namespace detail

}  // namespace optim
}  // namespace cinn
