// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/add_support_int8_pass.h"

namespace paddle {
namespace framework {
namespace ir {

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES        \
  GET_IR_NODE(prev_op);  \
  GET_IR_NODE(prev_out); \
  GET_IR_NODE(quant_op); \
  GET_IR_NODE(quant_out);

void AddSupportInt8Pass::ApplyImpl(ir::Graph* graph) const {
  const std::string pattern_name = "add_support_int8";
  FusePassBase::Init(pattern_name, graph);

  GraphPatternDetector gpd;

  patterns::AddSupportInt8 pattern(gpd.mutable_pattern(), pattern_name);
  pattern();
  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_NODES;
    if (prev_op->Op()->HasAttr("out_threshold") &&
        quant_op->Op()->HasAttr("out_threshold")) {
      quant_op->Op()->SetAttr("support_int8", true);
    }
    found_count++;
  };
  gpd(graph, handler);
  AddStatis(found_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(add_support_int8_pass, paddle::framework::ir::AddSupportInt8Pass);
