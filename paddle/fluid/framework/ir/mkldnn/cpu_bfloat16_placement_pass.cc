/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/mkldnn/cpu_bfloat16_placement_pass.h"

#include <string>
#include <unordered_set>

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/utils/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

void CPUBfloat16PlacementPass::ApplyImpl(ir::Graph* graph) const {
  int bfloat16_operators = 0;
  bfloat16_operators += SetMkldnnDataType(graph);
  bfloat16_operators -= RemoveOrphanedOperators(graph);
  bfloat16_operators -= RemoveUnsupportedOperators(graph);
  PrettyLogDetail("---    marked %d operators to bfloat16 ",
                  bfloat16_operators);
}

int CPUBfloat16PlacementPass::SetMkldnnDataType(ir::Graph* graph) const {
  const auto& op_types_list =
      Get<std::unordered_set<std::string>>("bfloat16_enabled_op_types");
  // set mkldnn_data_type to bfloat16 to all operators that are in
  // bfloat16_enabled_op_types vector or they are included to Bfloat16Placement
  // pattern
  GraphPatternDetector gpd;
  patterns::Bfloat16Placement bfloat16_placement_pattern{gpd.mutable_pattern(),
                                                         "bfloat16_placement"};
  bfloat16_placement_pattern(op_types_list);

  int detected_operators = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(op_in, op_in, bfloat16_placement_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(op, op, bfloat16_placement_pattern);

    // Only float input can be converted to bfloat16
    if (op_in->Var()->GetDataType() != proto::VarType::FP32) return;

    if (platform::HasOpINT8DataType(op->Op()) == false) {
      VLOG(4) << "---    marked " << op->Op()->Type()
              << " operator to bfloat16 ";
      op->Op()->SetAttr("mkldnn_data_type", std::string("bfloat16"));
      detected_operators++;
    }
  };
  gpd(graph, handler);
  return detected_operators;
}

int CPUBfloat16PlacementPass::RemoveOrphanedOperators(ir::Graph* graph) const {
  // find orphaned bfloat16 operator that is between two float32 operators
  // revert mkldnn_data_type attr to float32
  GraphPatternDetector gpd;
  patterns::OrphanedBfloat16 orphaned_bfloat16_pattern{gpd.mutable_pattern(),
                                                       "orphaned_bfloat16"};
  orphaned_bfloat16_pattern();
  int detected_operators = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(op, op, orphaned_bfloat16_pattern);

    op->Op()->SetAttr("mkldnn_data_type", std::string("float32"));
    VLOG(4) << "---  demarked " << op->Op()->Type() << " operator to bfloat16 ";
    detected_operators++;
  };
  gpd(graph, handler);
  return detected_operators;
}

int CPUBfloat16PlacementPass::RemoveUnsupportedOperators(
    ir::Graph* graph) const {
  // now quantize is supported FP32 only, so try to find
  // bfloat16 operator that input type is not FP32
  GraphPatternDetector gpd;
  patterns::UnsupportedBfloat16 unsupported_bfloat16_pattern{
      gpd.mutable_pattern(), "unsupported_bfloat16"};
  unsupported_bfloat16_pattern();
  int detected_operators = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(prev_out, prev_out, unsupported_bfloat16_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(op, op, unsupported_bfloat16_pattern);
    if ((prev_out->Var()->GetDataType() != proto::VarType::FP32)) {
      op->Op()->SetAttr("mkldnn_data_type", std::string("float32"));
      VLOG(4) << "---  demarked " << op->Op()->Type()
              << " operator to bfloat16 ";
      detected_operators++;
    }
  };
  gpd(graph, handler);
  return detected_operators;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(cpu_bfloat16_placement_pass,
              paddle::framework::ir::CPUBfloat16PlacementPass)
    // a vector of operator type names with bfloat16 support ("conv2d" etc.)
    // the second param is the default value for this vector
    .DefaultPassAttr("bfloat16_enabled_op_types",
                     new std::unordered_set<std::string>());
