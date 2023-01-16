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

#include "paddle/fluid/framework/ir/mkldnn/quant_transpose2_dequant_onednn_fuse_pass.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

void FuseQuantTranspose2DequantOneDNNPass::FuseQuantizeTranspose2(
    Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init("quant_transpose2_dequant_onednn_fuse_pass", graph);

  GraphPatternDetector gpd;
  patterns::QuantTranspose2 quant_transpose2_pattern(
      gpd.mutable_pattern(), "quant_transpose2_dequant_onednn_fuse_pass");
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

    bool is_negative_output =
        quant_op->Op()->HasAttr("is_negative_input")
            ? PADDLE_GET_CONST(bool,
                               quant_op->Op()->GetAttr("is_negative_input"))
            : false;
    bool is_bfloat =
        quant_op->Op()->HasAttr("bfloat16")
            ? PADDLE_GET_CONST(bool, quant_op->Op()->GetAttr("bfloat16"))
            : false;

    std::string output_dtype;
    if (is_bfloat) {
      output_dtype = "bf16";
    } else if (is_negative_output) {
      output_dtype = "int8";
    } else {
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

void FuseQuantTranspose2DequantOneDNNPass::FuseTranspose2Dequantize(
    Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init("quant_transpose2_dequant_onednn_fuse_pass", graph);

  GraphPatternDetector gpd;
  patterns::Transpose2Dequant transpose2_dequant_pattern(
      gpd.mutable_pattern(), "quant_transpose2_dequant_onednn_fuse_pass");
  transpose2_dequant_pattern();

  int found_patterns_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_op, transpose2_op, transpose2_dequant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        dequant_in, dequant_in, transpose2_dequant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        dequant_op, dequant_op, transpose2_dequant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        dequant_out, dequant_out, transpose2_dequant_pattern);

    if (!transpose2_op->Op()->HasAttr("use_mkldnn") ||
        (transpose2_op->Op()->HasAttr("use_mkldnn") &&
         !(PADDLE_GET_CONST(bool,
                            transpose2_op->Op()->GetAttr("use_mkldnn"))))) {
      VLOG(4) << "Only oneDNN version of transpose2 can be fused before with "
                 "dequantize.";
      return;
    }

    if (!dequant_op->Op()->HasAttr("Scale") &&
        !dequant_op->Op()->HasAttr("Shift")) {
      VLOG(4) << "Dequantize operator should have scale and shift attributes.";
      return;
    }
    float scale =
        dequant_op->Op()->HasAttr("Scale")
            ? PADDLE_GET_CONST(float, dequant_op->Op()->GetAttr("Scale"))
            : 1;
    float reorder_scale = 1.0 / scale;
    float shift =
        dequant_op->Op()->HasAttr("Shift")
            ? PADDLE_GET_CONST(float, dequant_op->Op()->GetAttr("Shift"))
            : 0;

    PADDLE_ENFORCE(scale != 0.0f,
                   phi::errors::InvalidArgument(
                       "Dequantization scale must be different than 0.0f"));

    PADDLE_ENFORCE(shift <= 255 && shift >= 0,
                   phi::errors::InvalidArgument(
                       "Dequantization shift must be lower or equal to ",
                       "255 and greater or equal to 0, but got %f",
                       shift));

    std::string output_dtype = "fp32";
    transpose2_op->Op()->SetAttr("scale", reorder_scale);
    transpose2_op->Op()->SetAttr("shift", shift);
    transpose2_op->Op()->SetAttr("output_data_type", output_dtype);

    transpose2_op->Op()->SetOutput(
        "Out", std::vector<std::string>({dequant_out->Name()}));
    IR_NODE_LINK_TO(transpose2_op, dequant_out);
    GraphSafeRemoveNodes(graph, {dequant_in, dequant_op});
    found_patterns_count++;
  };

  gpd(graph, handler);
  AddStatis(found_patterns_count);
  if ((!Has("disable_logs") || !Get<bool>("disable_logs"))) {
    paddle::string::PrettyLogDetail("--- fused %d transpose2 with dequant",
                                    found_patterns_count);
  }
}

void FuseQuantTranspose2DequantOneDNNPass::ApplyImpl(Graph *graph) const {
  FuseQuantizeTranspose2(graph);
  FuseTranspose2Dequantize(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(quant_transpose2_dequant_onednn_fuse_pass,
              paddle::framework::ir::FuseQuantTranspose2DequantOneDNNPass);
REGISTER_PASS_CAPABILITY(quant_transpose2_dequant_onednn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .GE("quantize", 0)
            .GE("transpose2", 0));
