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

#include "paddle/fluid/framework/ir/replace_dense_with_sparse_pass.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

ReplaceDenseWithSparsePass::ReplaceDenseWithSparsePass() {
  AddOpCompat(OpCompat("fc"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("W")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();
}

void ReplaceDenseWithSparsePass::ApplyImpl(Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));

  std::string name_scope = "replace_dense_with_sparse_pass";
  FusePassBase::Init(name_scope, graph);
  GraphPatternDetector gpd;

  patterns::DenseFC dense_fc_pattern(gpd.mutable_pattern(),
                                     "dense_replace_pass");
  dense_fc_pattern();
  int found_dense_fc_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "Replace dense fc with sparse_fc.";

    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }

    // FC output
    GET_IR_NODE_FROM_SUBGRAPH(fc_out, fc_out, dense_fc_pattern);
    // ops
    GET_IR_NODE_FROM_SUBGRAPH(fc, fc, dense_fc_pattern);
    auto *fc_op = fc->Op();
    auto w_name = fc_op->Input("W")[0];
    // recognize sparse op by name
    if (w_name.find("sparse") != w_name.npos) {
      // fake op
      fc_op->SetType("sparse_fc");
      fc_op->Flush();
      found_dense_fc_count++;
    }
  };

  gpd(graph, handler);
  AddStatis(found_dense_fc_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(replace_dense_with_sparse_pass,
              paddle::framework::ir::ReplaceDenseWithSparsePass);
