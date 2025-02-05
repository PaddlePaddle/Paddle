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

#include "paddle/cinn/optim/replace_call_with_expr.h"

#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"

namespace cinn {
namespace optim {

struct ReplaceCallWithExprModifier : public ir::IRMutator<> {
  ReplaceCallWithExprModifier(const std::string &statement,
                              const Expr &candidate)
      : statement_(statement), candidate_(candidate) {}

  void operator()(Expr *e) { IRMutator<>::Visit(e, e); }

 private:
  void Visit(const ir::Call *expr, Expr *op) override {
    auto *node = op->As<ir::Call>();
    PADDLE_ENFORCE_EQ(
        !node->name.empty(),
        true,
        ::common::errors::InvalidArgument(
            "Call node must have a name, but an empty name was found."));

    VLOG(3) << "Processing Call node " << *op;
    if (statement_ != node->name) return;

    Expr expr_candidate =
        ir::ir_utils::IRCopy(candidate_, /* copy_buffer_node = */ false);
    VLOG(3) << "Original candidate expr: " << candidate_;
    VLOG(3) << "Copied candidate expr: " << expr_candidate;

    // Replace the Call node with the expression candidate.
    *op = expr_candidate;
    VLOG(3) << "expr " << *op;
  }

 private:
  std::string statement_;
  const Expr &candidate_;
};

void ReplaceCallWithExpr(Expr *e,
                         const std::string &statement,
                         const Expr &candidate) {
  ReplaceCallWithExprModifier modifier(statement, candidate);
  modifier(e);
}

void ReplaceIslCallWithExpr(Expr *e,
                            const std::string &statement,
                            const Expr &candidate,
                            const std::map<std::string, Expr> &axis_map) {
  VLOG(3) << "ReplaceCallWithExpr, original expression: " << candidate;
  Expr copied = ir::ir_utils::IRCopy(candidate, /* copy_buffer_node = */ false);
  // update the axis in the copied expression.

  // we treat the Store node as the normal statement, the others like Call node
  // has no axis.
  std::map<std::string, Expr> local_axis;
  std::vector<std::string> origin_axes;
  std::map<std::string, Expr> new_axis_map = axis_map;
  for (auto &item : axis_map) {
    origin_axes.push_back(item.first);
  }
  // Add '_after' to the transformed var's name to avoid duplicating
  // transforming. For example, given indices [i,j], if we want to switch 'i'
  // and 'j'(i->j, j->i) When we don't add '_after', the processing will be :
  // 1. [i,j] to [j,j]
  // 2. [j,j] to [i,i]
  // Then we get result [i,i], which is different form the correct result [j,i]
  // If we add '_after', the processing will be:
  // 1. [i,j] to [j_after,j]
  // 2. [j_after,j] to [j_after,i_after]
  // 3. [j_after,i_after] to [j, i]
  // Mission Complete!
  for (auto &item : new_axis_map) {
    for (auto &axis : origin_axes) {
      ReplaceVarWithExpr(&item.second, Var(axis), Expr(Var(axis + "_after")));
    }
  }
  if (copied.As<ir::Store>()) {
    auto *store = copied.As<ir::Store>();
    for (int i = 0; i < store->indices.size(); i++) {
      auto indice = store->indices[i];
      if (indice.is_var() || indice.is_constant()) {
        if (!new_axis_map.count(std::to_string(i))) continue;
        if (!indice.is_constant()) {
          local_axis[indice.as_var()->name] =
              new_axis_map.at(std::to_string(i));
        }
      }
    }
    // the store indices just contains the ones of transform's domain, not the
    // range. e.g. { s[i,j] -> s[i0,i1,j]: i0=i/4 and i1=i%4 }, the store's
    // indices just contains i,j while in the final code, the axis are from the
    // range, that is, there are some new axis not exists in store->indice, i0
    // and i1.
  }

  for (auto &laxis : local_axis) {
    VLOG(3) << "local_axis Replacing axis: " << laxis.first << " to "
            << laxis.second;
    ReplaceVarWithExpr(&copied, Var(laxis.first), laxis.second);
  }
  // replace the remaining axis(in the transform's range)
  for (auto &item : new_axis_map) {
    if (!local_axis.count(item.first)) {
      VLOG(3) << "new_axis_map Replacing axis: " << item.first << " to "
              << item.second;
      ReplaceVarWithExpr(&copied, Var(item.first), item.second);
    }
  }

  for (auto &axis : origin_axes) {
    ReplaceVarWithExpr(&copied, Var(axis + "_after"), Expr(Var(axis)));
  }

  VLOG(3) << "After replacing, the statement [" << statement
          << "] is : " << copied;
  ReplaceCallWithExpr(e, statement, copied);
}

}  // namespace optim
}  // namespace cinn
