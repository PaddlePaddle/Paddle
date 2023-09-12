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
#include <map>
#include <string>

#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace optim {

/**
 * Replace a Call node with a Expr (inline).
 * @param e The expression to modify.
 * @param statement The map from tuple_name to the expression candidate.
 * @param candidate Var of each axis in the expression candidate.
 */
void ReplaceCallWithExpr(Expr *e,
                         const std::string &statement,
                         const Expr &candidate);

/**
 * Replace a Call node with a Expr (inline).
 * @param e The expression to modify.
 * @param statement The map from tuple_name to the expression candidate.
 * @param candidate Var of each axis in the expression candidate.
 * @param axis_map The map from a variable to expression.
 */
void ReplaceIslCallWithExpr(Expr *e,
                            const std::string &statement,
                            const Expr &candidate,
                            const std::map<std::string, Expr> &axis_map);

}  // namespace optim
}  // namespace cinn
