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

#include "paddle/fluid/framework/ir/mkldnn/conv_relu_mkldnn_fuse_pass.h"
#include <string>
#include <vector>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void ConvReLUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE(graph);
  FusePassBase::Init("conv_relu_mkldnn_fuse", graph);

  GraphPatternDetector gpd;
  auto* conv_input = gpd.mutable_pattern()
                         ->NewNode("conv_relu_mkldnn_fuse/conv_input")
                         ->AsInput()
                         ->assert_is_op_input("conv2d", "Input");
  patterns::ConvReLU conv_relu_pattern(gpd.mutable_pattern(),
                                       "conv_relu_mkldnn_fuse");
  conv_relu_pattern(conv_input);

  int found_conv_relu_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle ConvReLU fuse";
    GET_IR_NODE_FROM_SUBGRAPH(conv_weight, conv_weight,
                              conv_relu_pattern);                      // Filter
    GET_IR_NODE_FROM_SUBGRAPH(conv_out, conv_out, conv_relu_pattern);  // tmp
    GET_IR_NODE_FROM_SUBGRAPH(conv, conv, conv_relu_pattern);  // CONV op
    GET_IR_NODE_FROM_SUBGRAPH(relu_out, relu_out, conv_relu_pattern);  // Out
    GET_IR_NODE_FROM_SUBGRAPH(relu, relu, conv_relu_pattern);  // ReLU op

    FuseOptions fuse_option = FindFuseOption(*conv, *relu);
    if (fuse_option == DO_NOT_FUSE) {
      VLOG(3) << "do not perform conv+relu fuse";
      return;
    }

    // Transform Conv node into ConvReLU node.
    OpDesc* desc = conv->Op();
    desc->SetOutput("Output", std::vector<std::string>({relu_out->Name()}));
    desc->SetAttr("fuse_relu", true);
    GraphSafeRemoveNodes(graph, {relu, conv_out});

    PADDLE_ENFORCE(subgraph.count(conv_input));
    IR_NODE_LINK_TO(conv, relu_out);

    found_conv_relu_count++;
  };

  gpd(graph, handler);

  AddStatis(found_conv_relu_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv_relu_mkldnn_fuse_pass,
              paddle::framework::ir::ConvReLUFusePass);
