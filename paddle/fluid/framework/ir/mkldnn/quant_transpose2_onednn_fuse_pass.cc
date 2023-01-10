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
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

void FuseQuantizeTranspose2OneDNNPass::ApplyImpl(Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init("quant_transpose2_onednn_fuse_pass", graph);

  GraphPatternDetector gpd;
  patterns::QuantTranspose2 quant_transpose2_pattern(
      gpd.mutable_pattern(), "quant_transpose2_onednn_fuse_pass");
  quant_transpose2_pattern();

  int found_patterns_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    GET_IR_NODE_FROM_SUBGRAPH(quant_in, quant_in, quant_transpose2_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quant_op, quant_op, quant_transpose2_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quant_out, quant_out, quant_transpose2_pattern);
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
    if (!quant_op->Op()->HasAttr("Scale") &&
        !quant_op->Op()->HasAttr("Shift")) {
      VLOG(4) << "Quantize operator should have scale and shift attributes.";
      return;
    }
    float scale =
        quant_op->Op()->HasAttr("Scale")
            ? PADDLE_GET_CONST(float, quant_op->Op()->GetAttr("Scale"))
            : 1;
    float shift =
        quant_op->Op()->HasAttr("Shift")
            ? PADDLE_GET_CONST(float, quant_op->Op()->GetAttr("Shift"))
            : 0;
    transpose2_op->Op()->SetAttr("scale", scale);
    transpose2_op->Op()->SetAttr("shift", shift);

    bool is_negative_output = quant_op->Op()->HasAttr("is_negative_input")
            ? PADDLE_GET_CONST(bool, quant_op->Op()->GetAttr("is_negative_input"))
            : false;
    bool is_bfloat =
        quant_op->Op()->HasAttr("bfloat16")
            ? PADDLE_GET_CONST(bool, quant_op->Op()->GetAttr("bfloat16"))
            : false;

    std::string output_dtype;
    if (is_bfloat){
      output_dtype = "bf16";
    }
    else if (is_negative_output) {
      output_dtype = "int8";
    }
    else{
      output_dtype = "uint8";
    }
    transpose2_op->Op()->SetAttr("output_data_type", output_dtype);
    transpose2_op->Op()->SetInput("X",
                                  std::vector<std::string>({quant_in->Name()}));

    IR_NODE_LINK_TO(quant_in, transpose2_op);
    GraphSafeRemoveNodes(graph, {quant_op, quant_out});
    found_patterns_count++;
  };
  gpd(graph, handler);
  AddStatis(found_patterns_count);
  if ((!Has("disable_logs") || !Get<bool>("disable_logs"))) {
    paddle::string::PrettyLogDetail("--- fused %d quant with transpose2",
                                    found_patterns_count);
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
