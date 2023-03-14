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
#include "paddle/fluid/framework/ir/mkldnn/squeeze2_transpose2_onednn_fuse_pass.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/utils/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

void FuseSqueeze2Transpose2OneDNNPass::ApplyImpl(Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph,
      platform::errors::InvalidArgument(
          "Input graph pointer argument should not be nullptr."));

  FusePassBase::Init("squeeze2_transpose2_onednn_fuse_pass", graph);

  GraphPatternDetector gpd;
  patterns::Squeeze2Transpose2 squeeze2_transpose2_pattern(
      gpd.mutable_pattern(), "squeeze2_transpose2_onednn_fuse_pass");
  squeeze2_transpose2_pattern();

  int found_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    GET_IR_NODE_FROM_SUBGRAPH(
        squeeze2_op_in, squeeze2_op_in, squeeze2_transpose2_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        squeeze2_op, squeeze2_op, squeeze2_transpose2_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        squeeze2_op_out, squeeze2_op_out, squeeze2_transpose2_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_op, transpose2_op, squeeze2_transpose2_pattern);

    if (!transpose2_op->Op()->HasAttr("use_mkldnn") ||
        (transpose2_op->Op()->HasAttr("use_mkldnn") &&
         !(PADDLE_GET_CONST(bool,
                            transpose2_op->Op()->GetAttr("use_mkldnn"))))) {
      VLOG(4) << "Only oneDNN version of transpose2 can be fused after with "
                 "squeeze2.";
      return;
    }

    std::vector<int> squeeze2_axes =
        PADDLE_GET_CONST(std::vector<int>, squeeze2_op->Op()->GetAttr("axes"));
    transpose2_op->Op()->SetAttr("fused_squeeze2_axes", squeeze2_axes);
    transpose2_op->Op()->SetInput("X", {squeeze2_op_in->Name()});

    IR_VAR_OP_LINK(squeeze2_op_in, transpose2_op);
    GraphSafeRemoveNodes(g, {squeeze2_op, squeeze2_op_out});
    found_count++;
  };

  gpd(graph, handler);
  AddStatis(found_count);
  if ((!Has("disable_logs") || !Get<bool>("disable_logs"))) {
    PrettyLogDetail("--- fused %d squeeze2 with transpose2", found_count);
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(squeeze2_transpose2_onednn_fuse_pass,
              paddle::framework::ir::FuseSqueeze2Transpose2OneDNNPass);
REGISTER_PASS_CAPABILITY(squeeze2_transpose2_onednn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .GE("squeeze2", 0)
            .GE("transpose2", 0));
