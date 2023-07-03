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

#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/hlir/pass/use_pass.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace hlir {
namespace pass {

using common::Type;
using framework::Graph;
using framework::Node;
using framework::NodeData;
using framework::Operator;

void ConstPropagatePass(Graph* graph) {
  auto store_nodes = std::get<0>(graph->topological_order());
  for (auto& n : store_nodes) {
    auto node = n->safe_as<Node>();
    if (node) {
      bool is_all_const = true;
      for (auto& in_edge : node->inlinks_in_order()) {
        auto* source_node = in_edge->source()->safe_as<NodeData>();
        CHECK(source_node);
        if (!source_node->is_const()) {
          is_all_const = false;
          break;
        }
      }
      if (is_all_const) {
        node->attrs.attr_store["pre_run"] = true;
        VLOG(4) << node->id() << " do pre_run";
        for (auto& out_edge : node->outlinks_in_order()) {
          // mark all out nodedatas as const
          auto* sink_node = out_edge->sink()->safe_as<NodeData>();
          CHECK(sink_node);
          sink_node->set_const(true);
        }
      }
    }
  }
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(ConstPropagate) {
  CINN_REGISTER_PASS(ConstPropagate)
      .describe(
          "This pass will propagate const node_datas and mark the op_node with "
          "the attr[\"pre_run\"] if inputs are all "
          "constants;")
      .set_change_structure(false)
      .provide_graph_attr("pre_run")
      .set_body(cinn::hlir::pass::ConstPropagatePass);
  return true;
}
