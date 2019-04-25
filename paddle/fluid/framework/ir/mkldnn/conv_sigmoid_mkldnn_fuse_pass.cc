// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/mkldnn/conv_sigmoid_mkldnn_fuse_pass.h"
#include <string>
#include <vector>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void ConvSigmoidFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE(graph);
  FusePassBase::Init("conv_sigmoid_mkldnn_fuse", graph);
  GraphPatternDetector gpd;
  auto* conv_input = gpd.mutable_pattern()
                         ->NewNode("conv_sigmoid_mkldnn_fuse/conv_input")
                         ->AsInput()
                         ->assert_is_op_input("conv2d", "Input");
  patterns::ConvSigmoid conv_sigmoid_pattern(gpd.mutable_pattern(),
                                             "conv_sigmoid_mkldnn_fuse");
  conv_sigmoid_pattern(conv_input);

  int found_conv_sigmoid_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle ConvSigmoid fuse";
    GET_IR_NODE_FROM_SUBGRAPH(conv_weight, conv_weight,
                              conv_sigmoid_pattern);  // Filter
    GET_IR_NODE_FROM_SUBGRAPH(conv_out, conv_out, conv_sigmoid_pattern);  // tmp
    GET_IR_NODE_FROM_SUBGRAPH(conv, conv, conv_sigmoid_pattern);  // CONV op
    GET_IR_NODE_FROM_SUBGRAPH(sigmoid_out, sigmoid_out,
                              conv_sigmoid_pattern);  // Out
    GET_IR_NODE_FROM_SUBGRAPH(sigmoid, sigmoid,
                              conv_sigmoid_pattern);  // sigmoid op

    // Transform Conv node into ConvSigmoid node.
    OpDesc* desc = conv->Op();
    desc->SetOutput("Output", std::vector<std::string>({sigmoid_out->Name()}));
    desc->SetAttr("fuse_sigmoid", true);
    // desc->SetAttr("fuse_sigmoid_threshold",
    // sigmoid->Op()->GetAttr("threshold"));

    GraphSafeRemoveNodes(graph, {sigmoid, conv_out});

    PADDLE_ENFORCE(subgraph.count(conv_input));
    IR_NODE_LINK_TO(conv, sigmoid_out);
    found_conv_sigmoid_count++;
  };

  gpd(graph, handler);

  AddStatis(found_conv_sigmoid_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv_sigmoid_mkldnn_fuse_pass,
              paddle::framework::ir::ConvSigmoidFusePass);
