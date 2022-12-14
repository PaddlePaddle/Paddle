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
#include "paddle/fluid/framework/ir/mkldnn/quant_transpose2_onednn_fuse_pass.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

void FuseQuantizeTranspose2OneDNNPass::ApplyImpl(Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph,
      platform::errors::InvalidArgument(
          "Input graph pointer argument should not be nullptr."));
  FusePassBase::Init("quant_transpose2_onednn_fuse_pass", graph);

  GraphPatternDetector gpd;
  patterns::QuantTranspose2 quant_transpose2_pattern(
      gpd.mutable_pattern(), "quant_transpose2_onednn_fuse_pass");
  quant_transpose2_pattern();

  int found_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    GET_IR_NODE_FROM_SUBGRAPH(quant_in, quant_in, quant_transpose2_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quant_op, quant_op, quant_transpose2_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_in, transpose2_in, quant_transpose2_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_op, transpose2_op, quant_transpose2_pattern);

    if (!transpose2_op->Op()->HasAttr("use_mkldnn") ||
        (transpose2_op->Op()->HasAttr("use_mkldnn") &&
         !(PADDLE_GET_CONST(bool,
                            transpose2_op->Op()->GetAttr("use_mkldnn"))))) {
      VLOG(4) << "Only oneDNN version of transpose2 can be fused after with "
                 "quantize.";
      return;
    }
    found_count++;
  };

  gpd(graph, handler);
  AddStatis(found_count);
  if ((!Has("disable_logs") || !Get<bool>("disable_logs"))) {
    paddle::string::PrettyLogDetail("--- fused %d quant with transpose2",
                                    found_count);
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(quant_transpose2_onednn_fuse_pass,
              paddle::framework::ir::FuseQuantizeTranspose2OneDNNPass);
REGISTER_PASS_CAPABILITY(quant_transpose2_onednn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .GE("quantize", 0)
            .GE("transpose2", 0));