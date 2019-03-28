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

#include <memory>
#include <string>

#include "paddle/fluid/framework/ir/anakin_fillconstant_elementwisemul_fuse.h"
#include "paddle/fluid/framework/ir/graph_viz_pass.h"

namespace paddle {
namespace framework {
namespace ir {

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES                 \
  GET_IR_NODE(fill_constant);     \
  GET_IR_NODE(fill_constant_out); \
  GET_IR_NODE(elementwise_mul);   \
  GET_IR_NODE(elementwise_mul_out);

void AnakinFillconstantElementwisemulFuse::ApplyImpl(ir::Graph* graph) const {
  const std::string pattern_name = "anakin_fillconstant_elementwisemul_fuse";
  FusePassBase::Init(pattern_name, graph);

  GraphPatternDetector gpd;
  auto* x = gpd.mutable_pattern()
                ->NewNode("x")
                ->assert_is_op_input("elementwise_mul", "X")
                ->AsInput();

  patterns::AnakinFillConstantElementWiseMulFuse pattern(gpd.mutable_pattern(),
                                                         pattern_name);
  pattern(x);

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_NODES;

    PADDLE_ENFORCE(subgraph.count(x));
    auto* elementwise_in = subgraph.at(x);
    float constant_value =
        boost::get<float>(fill_constant->Op()->GetAttr("value"));

    framework::OpDesc new_op_desc;
    new_op_desc.SetType("scale");
    new_op_desc.SetInput("X", {elementwise_in->Name()});
    new_op_desc.SetAttr("scale", constant_value);
    new_op_desc.SetAttr("bias", static_cast<float>(0.0));
    new_op_desc.SetAttr("bias_after_scale", true);
    new_op_desc.SetOutput("Out", {elementwise_mul_out->Name()});
    new_op_desc.Flush();

    // Create a new node for the fused op.
    auto* scale_op = graph->CreateOpNode(&new_op_desc);

    IR_NODE_LINK_TO(elementwise_in, scale_op);       // Input
    IR_NODE_LINK_TO(scale_op, elementwise_mul_out);  // Output

    // Delete the unneeded nodes.
    GraphSafeRemoveNodes(graph,
                         {fill_constant, fill_constant_out, elementwise_mul});
  };

  gpd(graph, handler);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(anakin_fillconstant_elementwisemul_fuse,
              paddle::framework::ir::AnakinFillconstantElementwisemulFuse);
