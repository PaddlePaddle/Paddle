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

#include "paddle/cinn/optim/compute_inline_expand.h"

#include <map>
#include <string>

#include "paddle/cinn/common/graph_utils.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"

namespace cinn {
namespace optim {

namespace {

/*
 * Replace a tensor(marked as compute_inline) to the expanded expression.
 */
struct SSANode : public cinn::common::GraphNode {
  std::string id_;

  explicit SSANode(const std::string &id) : id_(id) {}

  std::string id() const override { return id_; }

  const char *type_info() const override { return __type_info__; }

  static constexpr char *__type_info__ = "optim::SSANode";
};

// TODO(Superjomn) the graph here is not a SSA now, it is flatten for the
// ir::CollectIRNodes method collects all the tensors recursively, so it can not
// reserve the level information, fix it.
struct SSABuilder : public ir::IRMutator<> {
  cinn::common::Graph graph;

  SSABuilder &operator()(Expr *expr) {
    ir::IRMutator<>::Visit(expr, expr);
    return *this;
  }

  void Visit(const ir::Store *op, Expr *expr) override {
    auto *node = expr->As<ir::Store>();

    auto *cur_graph_node = graph.RetrieveNode(node->tensor.as_tensor()->name);
    if (!cur_graph_node) {
      cur_graph_node =
          graph.RegisterNode(node->tensor.as_tensor()->name,
                             new SSANode(node->tensor.as_tensor()->name));
    }

    auto deps_tensor_names = node->tensor.as_tensor()->GetDependTensorNames();
    for (auto &t : deps_tensor_names) {
      auto *n = graph.RetrieveNode(t);
      if (!n) {
        n = graph.RegisterNode(t, new SSANode(t));
      }
      n->Controls(cur_graph_node);
    }
  }
};

}  // namespace

}  // namespace optim
}  // namespace cinn
