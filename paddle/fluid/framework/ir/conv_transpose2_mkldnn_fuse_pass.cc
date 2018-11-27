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

#include "paddle/fluid/framework/ir/conv_transpose2_mkldnn_fuse_pass.h"
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

std::unique_ptr<ir::Graph> ConvTranspose2FusePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init("conv_transpose2_mkldnn_fuse", graph.get());

  GraphPatternDetector gpd;
  auto* conv_input = gpd.mutable_pattern()
                         ->NewNode("conv_transpose2_mkldnn_fuse/conv_input")
                         ->AsInput()
                         ->assert_is_op_input("conv2d", "Input");
  patterns::ConvTranspose2 conv_transpose2_pattern(
      gpd.mutable_pattern(), "conv_transpose2_mkldnn_fuse");
  conv_transpose2_pattern(conv_input);

  int found_conv_transpose2_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle ConvTranspose2 fuse";
    GET_IR_NODE_FROM_SUBGRAPH(conv_weight, conv_weight,
                              conv_transpose2_pattern);  // Filter
    GET_IR_NODE_FROM_SUBGRAPH(conv_out, conv_out,
                              conv_transpose2_pattern);              // tmp
    GET_IR_NODE_FROM_SUBGRAPH(conv, conv, conv_transpose2_pattern);  // CONV op
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_out, transpose2_out,
                              conv_transpose2_pattern);  // Out
    GET_IR_NODE_FROM_SUBGRAPH(transpose2, transpose2,
                              conv_transpose2_pattern);  // Transpose2 op

    FuseOptions fuse_option = FindFuseOption(*conv, *transpose2);
    if (fuse_option == DO_NOT_FUSE) {
      VLOG(3) << "do not perform conv+transpose2 fuse";
      return;
    }

    // Transform Conv node into ConvTranspose2 node.
    OpDesc* desc = conv->Op();
    desc->SetOutput("Output",
                    std::vector<std::string>({transpose2_out->Name()}));
    std::string nhwc = "NHWC";
    if (desc->HasAttr("reorder_output_format"))
      desc->SetAttr("reorder_output_format", nhwc);
    else
      desc->MutableAttrMap()->insert(
          std::pair<std::string, Attribute>("reorder_output_format", nhwc));

    GraphSafeRemoveNodes(graph.get(), {transpose2, conv_out});

    PADDLE_ENFORCE(subgraph.count(conv_input));
    IR_NODE_LINK_TO(conv, transpose2_out);

    found_conv_transpose2_count++;
  };

  gpd(graph.get(), handler);

  AddStatis(found_conv_transpose2_count);
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv_transpose2_mkldnn_fuse_pass,
              paddle::framework::ir::ConvTranspose2FusePass);
