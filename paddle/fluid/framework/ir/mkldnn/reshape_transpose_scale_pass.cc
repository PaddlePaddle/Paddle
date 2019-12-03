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

#include "paddle/fluid/framework/ir/mkldnn/reshape_transpose_scale_pass.h"
#include "paddle/fluid/framework/ir/graph_viz_pass.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

void ReshapeTransposeScalePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph, "graph cannot be nullptr.");
  FusePassBase::Init("reshape_transpose_scale_pass", graph);
  GraphPatternDetector gpd;
  patterns::ReshapeTransposeScale reshape_transpose_scale_pattern{gpd.mutable_pattern(), "reshape_transpose_scale_pass"};  
  reshape_transpose_scale_pattern();

  int found_reshape_transpose_scale_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                    Graph* g){
      PrettyLogDetail("Handle reshape_transpose_scale_pass");                   
      GET_IR_NODE_FROM_SUBGRAPH(reshape_in, reshape_in, reshape_transpose_scale_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(reshape_op, reshape_op, reshape_transpose_scale_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(reshape_out, reshape_out, reshape_transpose_scale_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(transpose_op, transpose_op, reshape_transpose_scale_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(transpose_out, transpose_out, reshape_transpose_scale_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(scale_op, scale_op, reshape_transpose_scale_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(scale_out, scale_out, reshape_transpose_scale_pattern);

      IR_NODE_LINK_TO(transpose_op, scale_out);
      transpose_op->Op()->SetOutput("Out",
                            std::vector<std::string>({scale_out->Name()}));
      GraphSafeRemoveNodes(graph, {scale_op, transpose_out});
      found_reshape_transpose_scale_count++;
  };
  gpd(graph, handler);
  AddStatis(found_reshape_transpose_scale_count);
  PrettyLogDetail("---   squashed %d reshape2-transpose2-scale pairs in reshape_transpose_scale_pass.cc",
    found_reshape_transpose_scale_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(reshape_transpose_scale_pass,
              paddle::framework::ir::ReshapeTransposeScalePass);
