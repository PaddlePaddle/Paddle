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

/**
 * This file implements the isl AST build interface, it helps to generate isl
 * AST given the polyhedral domain and schedule.
 */
#pragma once
#include <isl/cpp.h>

#include <map>
#include <string>
#include <vector>

#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/poly/isl_utils.h"
#include "paddle/cinn/poly/poly_scheduler.h"
#include "paddle/cinn/poly/schedule.h"
#include "paddle/cinn/poly/stage.h"
#include "paddle/cinn/utils/functional.h"

namespace cinn {
namespace poly {

static const char* kIslParamConstPrefix = "_const_";

/**
 * Generate IR from polyhedral schedule.
 */
class AstGen {
 public:
  AstGen(const isl::set& context,
         const std::vector<Stage*>& stages,
         const poly::ScheduleGroup& group);
  ~AstGen();

  /**
   * Set for-loop iterator names.
   * @param names
   * @return AstGen itself.
   */
  AstGen& SetIteratorNames(const std::vector<std::string>& names);

  isl::ctx ctx() const;

  isl::ast_node Build();

  //! Get the map from original CINN iterators to the transformed actual ISL ast
  //! nodes.
  const std::map<std::string, isl::ast_expr>& axis2ast(
      const std::string& tuple_name) const;

  const std::map<std::string, Expr> axis2expr(
      const std::string& tuple_name) const;

  bool ContainsStatement(const std::string& name) const;

  void SetBuildOptions(const isl::union_map& options);

  isl::union_set domain() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

void AddUnitLoopOfDomain(const isl::ast_node& node,
                         const isl::set& domain,
                         ir::Expr* expr);

/**
 * Transform the isl ast to Expr.
 */
void IslAstNodeToCinnExpr(const isl::ast_node& node, ir::Expr* expr);
void IslAstNodeToCinnExpr(const isl::ast_node& node,
                          const isl::union_set& domain,
                          ir::Expr* expr);
void IslAstExprToCinnExpr(const isl::ast_expr& node, ir::Expr* expr);

/**
 * Transform the set whose axis has one element like
 *  { s[i=0,j] : ... }
 * to a new set with a parameter to force all the axis has a range:
 *  [_const_0] -> { s[i,j]: 0 <= i <= _const_0 and _const_0 < 0+2 and ... }
 */
isl::union_set TransIdentityExtentToContextId(isl::union_set set);
isl::set TransIdentityExtentToContextId(isl::set set);

namespace detail {

//! Get tuple name of a ast node.
std::string GetTupleName(isl_ast_node* node);

}  // namespace detail

}  // namespace poly
}  // namespace cinn
