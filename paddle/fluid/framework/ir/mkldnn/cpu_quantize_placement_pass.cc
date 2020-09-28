/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/mkldnn/cpu_quantize_placement_pass.h"
#include <unordered_set>

namespace paddle {
namespace framework {
namespace ir {

class Graph;

void CPUQuantizePlacementPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(3) << "Marks operators which are to be quantized.";
  const auto& excluded_ids_list =
      Get<std::unordered_set<int>>("quantize_excluded_op_ids");
  const auto& op_types_list =
      Get<std::unordered_set<std::string>>("quantize_enabled_op_types");
  Init(name_scope_, graph);
  GraphPatternDetector gpd;
  patterns::QuantizePlacement quantize_placement_pattern{gpd.mutable_pattern(),
                                                         "quantize_placement"};
  quantize_placement_pattern(op_types_list);

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(op, op, quantize_placement_pattern);

    if (std::find(excluded_ids_list.begin(), excluded_ids_list.end(),
                  op->id()) != excluded_ids_list.end()) {
      return;
    }

    if (op->Op()->HasAttr("mkldnn_data_type") ||
        op->Op()->HasProtoAttr("mkldnn_data_type")) {
      // use_quantizer is no longer used
      // assign value for compatibility
      if (op->Op()->GetAttrIfExists<bool>("use_quantizer")) {
        op->Op()->SetAttr("mkldnn_data_type", std::string("int8"));
      }
      op->Op()->SetAttr("mkldnn_data_type", std::string("int8"));
      op->Op()->SetAttr("use_quantizer", true);
    }
  };
  gpd(graph, handler);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(cpu_quantize_placement_pass,
              paddle::framework::ir::CPUQuantizePlacementPass)
    // a vector of operator type names to be quantized ("conv2d" etc.)
    // the second param is the default value for this vector
    .DefaultPassAttr("quantize_enabled_op_types",
                     new std::unordered_set<std::string>())
    // a vector of operator ids that are to be excluded from quantization
    // the second param is the default value for this vector
    .DefaultPassAttr("quantize_excluded_op_ids", new std::unordered_set<int>());
