// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/add_bias_layernorm_pass.h"

#include <string>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES                    \
  GET_IR_NODE(elementwise_add_op);   \
  GET_IR_NODE(elementwise_add_in_y); \
  GET_IR_NODE(elementwise_add_out);  \
  GET_IR_NODE(flatten_op);           \
  GET_IR_NODE(flatten_out);          \
  GET_IR_NODE(flatten_out_xshape);   \
  GET_IR_NODE(transpose_op);         \
  GET_IR_NODE(transpose_out);        \
  GET_IR_NODE(transpose_out_xshape); \
  GET_IR_NODE(layer_norm_op);        \
  GET_IR_NODE(layer_norm_in_bais);   \
  GET_IR_NODE(layer_norm_in_scale);  \
  GET_IR_NODE(layer_norm_out_y);     \
  GET_IR_NODE(layer_norm_out_mean);  \
  GET_IR_NODE(layer_norm_out_variance);

namespace paddle {
namespace framework {
namespace ir {

AddBiasLayernormPass::AddBiasLayernormPass() {
  AddOpCompat(OpCompat("elementwise_add"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsNumEQ(1)
      .End();
}

void AddBiasLayernormPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init("skip_layernorm_fuse", graph);
  const std::string pattern_name = "add_bias_layernorm";
  FusePassBase::Init(pattern_name, graph);
  auto* scope = param_scope();

  // pattern
  PDNode* x = gpd.mutable_pattern()
                  ->NewNode("x")
                  ->assert_is_ops_input("elementwise_add", "X")
                  ->AsInput();
  patterns::AddBiasLayernormPattern pattern(gpd.mutable_pattern(),
                                            pattern_name);
  pattern(x);

  int fusion_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    GET_NODES;
    // Link inputs and outputs.
    PADDLE_ENFORCE_NE(
        subgraph.count(x),
        0,
        platform::errors::NotFound("Detector did not find input x of conv2d."));

    IR_NODE_LINK_TO(subgraph.at(x), flatten_op);  // Input
    IR_NODE_LINK_TO(flatten_out, transpose_op);
    IR_NODE_LINK_TO(transpose_out, elementwise_add_op);
    IR_NODE_LINK_TO(elementwise_add_out, layer_norm_op);  // Output

    ++fusion_count;
  };
  gpd(graph, handler);
  AddStatis(fusion_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(add_bias_layernorm_pass,
              paddle::framework::ir::AddBiasLayernormPass);
REGISTER_PASS_CAPABILITY(add_bias_layernorm_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .GE("flatten2", 0)
            .GE("transpose2", 0)
            .LE("elementwise_add", 1)
            .EQ("layer_norm", 0));
