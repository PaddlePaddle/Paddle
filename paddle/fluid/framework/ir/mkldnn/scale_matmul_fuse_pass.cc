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

#include "paddle/fluid/framework/ir/mkldnn/scale_matmul_fuse_pass.h"
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

void ScaleMatmulFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          platform::errors::InvalidArgument(
                              "Pointer to graph argument should not be NULL."));

  FusePassBase::Init("scale_matmul_fuse_pass", graph);
  GraphPatternDetector gpd;
  patterns::ScaleMatmul scale_matmul_pattern{gpd.mutable_pattern(),
                                             "scale_matmul"};
  scale_matmul_pattern();

  int found_scale_matmul_fuse_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(scale_in, scale_in, scale_matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_op, scale_op, scale_matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_out, scale_out, scale_matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_op, matmul_op, scale_matmul_pattern);

    if (scale_op->Op()->GetAttrIfExists<float>("bias") == 0.0) {
      auto matmul_alpha = matmul_op->Op()->GetAttrIfExists<float>("alpha");
      auto scale_scale = scale_op->Op()->GetAttrIfExists<float>("scale");
      PADDLE_ENFORCE_GT(matmul_alpha, 0.0f,
                        platform::errors::InvalidArgument(
                            "Alpha of matmul op should have positive value"));
      PADDLE_ENFORCE_GT(scale_scale, 0.0f,
                        platform::errors::InvalidArgument(
                            "Scale of scale op should have positive value"));

      std::string matmul_op_input_name;
      for (auto name : matmul_op->Op()->InputNames())
        for (auto input_name : matmul_op->Op()->Input(name))
          if (input_name == scale_out->Name()) matmul_op_input_name = name;

      PADDLE_ENFORCE_NE(
          matmul_op_input_name.empty(), true,
          platform::errors::NotFound("Operator after scale operator "
                                     "should have scale output as input"));
      matmul_op->Op()->SetAttr("alpha", matmul_alpha * scale_scale);
      matmul_op->Op()->SetInput(matmul_op_input_name,
                                std::vector<std::string>({scale_in->Name()}));
      IR_NODE_LINK_TO(scale_in, matmul_op);
      GraphSafeRemoveNodes(graph, {scale_op, scale_out});
      found_scale_matmul_fuse_count++;
    }
  };
  gpd(graph, handler);
  AddStatis(found_scale_matmul_fuse_count);
  PrettyLogDetail("---    fused %d scale with matmul",
                  found_scale_matmul_fuse_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(scale_matmul_fuse_pass,
              paddle::framework::ir::ScaleMatmulFusePass);
