// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/common/type.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/hlir/op/external_api_registry.h"
#include "paddle/cinn/utils/string.h"

PD_DECLARE_string(cinn_custom_call_deny_ops);

namespace cinn {
namespace hlir {
namespace pass {

using cinn::hlir::op::ExternalApiRegistry;
using framework::Graph;
using framework::Node;
using framework::NodeData;

class GraphAlterHelper {
 public:
  explicit GraphAlterHelper(Graph* graph) : graph_(graph) {
    if (!FLAGS_cinn_custom_call_deny_ops.empty()) {
      auto splited_names =
          cinn::utils::Split(FLAGS_cinn_custom_call_deny_ops, ";");
      deny_ops_ = {splited_names.begin(), splited_names.end()};
    }
  }
  void TransToCustomCall(const common::Target& target) {
    // collect candidate nodes
    auto mark_nodes = graph_->CollectNodes(
        [this, &target](const common::GraphNode* graph_node) -> bool {
          if (graph_node->safe_as<Node>()) {
            auto node = graph_node->safe_as<Node>();
            auto&& op_name = node->op()->name;
            // a op with external_api registered and not excluded explicitly
            // will be selected
            if (!IsExcluded(op_name) &&
                ExternalApiRegistry::Global()->Has(op_name, target)) {
              VLOG(4) << "Op:" << op_name << " will use custom_call";
              return true;
            }
          }

          return false;
        });

    for (auto* graph_node : mark_nodes) {
      auto* node = graph_node->safe_as<Node>();
      // revise the output edges for conv2d because the compute implement of
      // codegen-registered is not consistent with cudnn
      if ((node->op()->name == "conv2d" ||
           node->op()->name == "depthwise_conv2d") &&
          target == common::DefaultNVGPUTarget()) {
        auto out_links = node->outlinks_in_order();
        for (int idx = 1; idx < out_links.size(); ++idx) {
          auto link = out_links[idx];
          CHECK(link->sink()->safe_as<NodeData>());
          node->UnLinkSingleTo(link->sink());
          graph_->DropNode(link->sink());
        }
      }

      node->attrs.attr_store["original_op"] = node->op()->name;
      node->attrs.op = framework::Operator::Get("custom_call");
    }
  }

 private:
  Graph* graph_;
  std::unordered_set<std::string> deny_ops_;

  bool IsExcluded(const std::string& op_name) {
    return deny_ops_.count(op_name);
  }
};

void TransToCustomCallInternal(Graph* graph) {
  VLOG(3) << "TransToCustomCallPass...!";
  GraphAlterHelper(graph).TransToCustomCall(graph->target_);
  VLOG(3) << "TransToCustomCallPass Finish...!";
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(TransToCustomCallPass) {
  CINN_REGISTER_PASS(TransToCustomCallPass)
      .describe(
          "This pass replaces every op with external_api registered on the "
          "specified target to be custom_call op, "
          "except the blacklist specified by FLAGS_cinn_custom_call_deny_ops")
      .set_change_structure(false)
      .set_body(cinn::hlir::pass::TransToCustomCallInternal);
  return true;
}
