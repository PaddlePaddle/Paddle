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
#include <unordered_set>
#include <vector>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/stmt.h"

namespace cinn {
namespace optim {

/**
 * Replace the variable with a expression.
 * @param var The variable to replace.
 * @param expr The candidate expression.
 * @param tensor_name Name of the tensor whose indices will be edited. If it is
 * empty, means we will do the replace in all Expr instead of only in specific
 * tensor's indices.
 */
/**
 * Example 1: ReplaceVarWithExpr(source, Var("i"), Expr(0), "A")
 * for(i, 0, 10)
 *   for(j, 0, 10)
 *      B[i,j] = A[i,j]
 *
 * =>
 *
 * for(i, 0, 10)
 *   for(j, 0, 10)
 *      B[i,j] = A[0,j]
 *
 * Example 2: ReplaceVarWithExpr(source, Var("i"), Expr(Var("k")))
 * for(i, 0, 10)
 *   for(j, 0, 10)
 *      B[i,j] = A[i,j]
 *
 * =>
 *
 * for(k, 0, 10)
 *   for(j, 0, 10)
 *      B[k,j] = A[k,j]
 */
template <typename SourceType>
void ReplaceVarWithExpr(SourceType source,
                        const Var &var,
                        const Expr &expr,
                        const std::string &tensor_name = "");

/**
 * Collect the specific tensor's indices.
 * @param tensor_name The specific tensor's name.
 * @return Return a vector containing all the indices of the specific tensor
 * appeared in source.
 */
/**
 * Example: CollectTensorIndex(source, "A")
 * for(i, 0, 10)
 *   for(j, 0, 10)
 *      C[i,j] = A[i,j] + A[0,j] + B[j,i] + B[i,0]
 *
 * =>
 *
 * Return value:
 * {{i,j},{0,j}}
 */
std::vector<std::vector<Expr>> CollectTensorIndex(
    Expr *source, const std::string &tensor_name);

}  // namespace optim
}  // namespace cinn
