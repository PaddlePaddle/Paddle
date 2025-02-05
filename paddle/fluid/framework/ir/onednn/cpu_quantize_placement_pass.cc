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

#include "paddle/fluid/framework/ir/onednn/cpu_quantize_placement_pass.h"

#include <unordered_set>
#include "paddle/fluid/framework/ir/onednn/onednn_pass_util.h"

namespace paddle::framework::ir {

class Graph;

void CPUQuantizePlacementPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(3) << "Marks operators which are to be quantized.";
  std::unordered_set<std::string> supported_op_types =
      std::unordered_set<std::string>({"concat",
                                       "conv2d",
                                       "depthwise_conv2d",
                                       "fused_conv2d",
                                       "fused_conv3d",
                                       "fused_matmul",
                                       "fused_elementwise_add",
                                       "fused_elementwise_mul",
                                       "fused_elementwise_sub",
                                       "elementwise_add",
                                       "elementwise_mul",
                                       "elementwise_sub",
                                       "fc",
                                       "matmul",
                                       "matmul_v2",
                                       "nearest_interp",
                                       "nearest_interp_v2",
                                       "pool2d",
                                       "prior_box",
                                       "reshape2",
                                       "fused_transpose",
                                       "transpose2",
                                       "fusion_gru",
                                       "fusion_lstm",
                                       "multi_gru",
                                       "slice",
                                       "split"});
  const auto& excluded_ids_list =
      Get<std::unordered_set<int>>("quantize_excluded_op_ids");
  const auto& op_types_list =
      Get<std::unordered_set<std::string>>("quantize_enabled_op_types");

  if (!op_types_list.empty()) {
    // Verify that all user-specified operators can be quantized.
    for (const auto& op : op_types_list) {
      PADDLE_ENFORCE_NE(
          supported_op_types.count(op),
          0,
          common::errors::InvalidArgument(
              "Pass attribute quantize_enabled_op_types contains operator %s "
              "that is not supported by OneDNN quantization.",
              op));
    }
    supported_op_types = op_types_list;
  }
  Init(name_scope_, graph);
  GraphPatternDetector gpd;
  patterns::QuantizePlacement quantize_placement_pattern{gpd.mutable_pattern(),
                                                         "quantize_placement"};
  quantize_placement_pattern(supported_op_types);

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(op, op, quantize_placement_pattern);
    if (std::find(excluded_ids_list.begin(),
                  excluded_ids_list.end(),
                  op->id()) != excluded_ids_list.end()) {
      return;
    }

    if (op->Op()->GetAttrIfExists<int>("skip_quant") == 1) {
      return;
    }

    ConvertToFusedOp(op->Op());
    op->Op()->SetAttr("mkldnn_data_type", std::string("int8"));
  };
  gpd(graph, handler);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(cpu_quantize_placement_pass,
              paddle::framework::ir::CPUQuantizePlacementPass)
    // a vector of operator type names to be quantized ("conv2d" etc.)
    // the second param is the default value for this vector
    .DefaultPassAttr("quantize_enabled_op_types",
                     new std::unordered_set<std::string>())
    // a vector of operator ids that are to be excluded from quantization
    // the second param is the default value for this vector
    .DefaultPassAttr("quantize_excluded_op_ids", new std::unordered_set<int>());
