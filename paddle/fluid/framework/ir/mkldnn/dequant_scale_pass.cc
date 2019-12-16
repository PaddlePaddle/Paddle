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

#include <string>

#include "paddle/fluid/framework/ir/graph_viz_pass.h"
#include "paddle/fluid/framework/ir/mkldnn/dequant_scale_pass.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

void DequantScalePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph, "graph cannot be nullptr.");
  FusePassBase::Init("dequant_scale_pass", graph);
  GraphPatternDetector gpd;
  patterns::DequantScale dequant_scale_pattern{gpd.mutable_pattern(),
                                               "dequant_scale_pass"};
  dequant_scale_pattern();

  int found_dequant_scale_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    PrettyLogDetail("Handle dequant_scale_pass");
    GET_IR_NODE_FROM_SUBGRAPH(dequant_op, dequant_op, dequant_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequant_out, dequant_out, dequant_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_op, scale_op, dequant_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_out, scale_out, dequant_scale_pattern);

    IR_NODE_LINK_TO(dequant_op, scale_out);
    dequant_op->Op()->SetOutput("Output",
                                std::vector<std::string>({scale_out->Name()}));
    GraphSafeRemoveNodes(graph, {scale_op, dequant_out});
    found_dequant_scale_count++;
  };
  gpd(graph, handler);
  AddStatis(found_dequant_scale_count);
  PrettyLogDetail(
      "---   squashed %d dequant-scale pairs in "
      "dequant_scale_pass.cc",
      found_dequant_scale_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(dequant_scale_pass, paddle::framework::ir::DequantScalePass);
