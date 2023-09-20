// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {

class AutoTransQuantizeOpPrecisionPass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;
  void FirstRound(ir::Graph* graph) const;

  const std::string name_scope_{"auto_trans_quantize_op_precision_pass"};
  const std::unordered_set<std::string> support_fusion_quant_op_type_{
      "conv2d_xpu", "fc_xpu"};
};

static inline Node* GetOpOutVarNodeByArgsName(ir::Graph* graph,
                                              Node* op_node,
                                              const std::string& arg_name) {
  CHECK_EQ(op_node->IsOp(), true);
  auto* op_desc = op_node->Op();
  auto out_var_nodes = op_desc->Output(arg_name);
  CHECK_EQ(out_var_nodes.size(), 1UL);
  auto out_var_name = out_var_nodes[0];
  auto out_var_node = FindNodeWithName(graph, out_var_name);
  return out_var_node;
}

void AutoTransQuantizeOpPrecisionPass::FirstRound(ir::Graph* graph) const {
  auto graph_size = graph->SubGraphsSize();
  VLOG(1) << "There is " << graph_size << " subgraphs need to be handle.";
  for (size_t i = 0; i < graph_size; i++) {
    auto subgraph = graph->GetSubGraph(i);
    VLOG(1) << "Handling the subgraph id: " << i;
    for (auto* op_node : TopologySortOperations(*subgraph)) {
      auto op_type = op_node->Op()->Type();
      if (support_fusion_quant_op_type_.find(op_type) !=
          support_fusion_quant_op_type_.end()) {
        bool enable_int8 = op_node->Op()->GetAttrIfExists<bool>("enable_int8");
        int out_dtype = op_node->Op()->GetAttrIfExists<int>("out_dtype");
        if (enable_int8) {
          auto* out_var_node =
              GetOpOutVarNodeByArgsName(subgraph, op_node, "out");
          PADDLE_ENFORCE_NOT_NULL(
              out_var_node,
              platform::errors::InvalidArgument(
                  "out_var_node in graph cannot be nullptr."));
          bool is_int8_out = true;
          for (auto* next_op_node : out_var_node->outputs) {
            auto next_op_type = next_op_node->Op()->Type();
            bool is_next_op_support_int8 =
                next_op_node->Op()->GetAttrIfExists<bool>("enable_int8") &&
                ((support_fusion_quant_op_type_.find(next_op_type) !=
                  support_fusion_quant_op_type_.end()));
            if (!is_next_op_support_int8) {
              is_int8_out = false;
              break;
            }
          }
          if (is_int8_out) {
            op_node->Op()->SetAttr(
                "out_dtype",
                static_cast<int>(proto::VarType::Type::VarType_Type_INT8));
            out_var_node->Var()->SetDataType(
                proto::VarType::Type::VarType_Type_INT8);
            VLOG(1) << "The out var node " << out_var_node->Name()
                    << " is INT8";
          }
        }
      }
    }
  }
}

void AutoTransQuantizeOpPrecisionPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  VLOG(1) << "AutoTransQuantizeOpPrecisionPass handling start ...";
  FirstRound(graph);
  VLOG(1) << "AutoTransQuantizeOpPrecisionPass handleing end.";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(auto_trans_quantize_op_precision_pass,
              paddle::framework::ir::AutoTransQuantizeOpPrecisionPass);

REGISTER_PASS_CAPABILITY(auto_trans_quantize_op_precision_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("fc_xpu", 0)
            .EQ("conv2d_xpu", 0));
