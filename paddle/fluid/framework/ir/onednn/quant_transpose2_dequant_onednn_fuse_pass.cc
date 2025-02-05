// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/onednn/quant_transpose2_dequant_onednn_fuse_pass.h"
#include "paddle/fluid/framework/ir/onednn/onednn_pass_util.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/utils/string/pretty_log.h"

namespace paddle::framework::ir {

void FuseQuantTranspose2DequantOneDNNPass::FuseQuantizeTranspose2(
    Graph *graph, const std::string &transpose_type) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init(name_scope, graph);

  GraphPatternDetector gpd;
  patterns::QuantTranspose quant_transpose2_pattern(gpd.mutable_pattern(),
                                                    name_scope);
  quant_transpose2_pattern(transpose_type);

  int found_patterns_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    GET_IR_NODE_FROM_SUBGRAPH(quant_in, quant_in, quant_transpose2_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quant_op, quant_op, quant_transpose2_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quant_out, quant_out, quant_transpose2_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose_op, transpose_op, quant_transpose2_pattern);

    if (!transpose_op->Op()->HasAttr("use_mkldnn") ||
        !(PADDLE_GET_CONST(bool, transpose_op->Op()->GetAttr("use_mkldnn")))) {
      VLOG(4)
          << "Only oneDNN version of transpose2 can be fused with quantize.";
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

    ConvertToFusedOp(transpose_op->Op());
    transpose_op->Op()->SetAttr("scale", scale);
    transpose_op->Op()->SetAttr("shift", shift);

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
    transpose_op->Op()->SetAttr("output_data_type", output_dtype);
    transpose_op->Op()->SetInput("X",
                                 std::vector<std::string>({quant_in->Name()}));

    IR_NODE_LINK_TO(quant_in, transpose_op);
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
    Graph *graph, const std::string &transpose_type) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init(name_scope, graph);

  GraphPatternDetector gpd;
  patterns::TransposeDequant transpose2_dequant_pattern(gpd.mutable_pattern(),
                                                        name_scope);
  transpose2_dequant_pattern(transpose_type);

  int found_patterns_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose_op, transpose_op, transpose2_dequant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        dequant_in, dequant_in, transpose2_dequant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        dequant_op, dequant_op, transpose2_dequant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        dequant_out, dequant_out, transpose2_dequant_pattern);

    if (!transpose_op->Op()->HasAttr("use_mkldnn") ||
        !(PADDLE_GET_CONST(bool, transpose_op->Op()->GetAttr("use_mkldnn")))) {
      VLOG(4)
          << "Only oneDNN version of transpose2 can be fused with dequantize.";
      return;
    }

    float scale =
        dequant_op->Op()->HasAttr("Scale")
            ? PADDLE_GET_CONST(float, dequant_op->Op()->GetAttr("Scale"))
            : 1;
    float reorder_scale = static_cast<float>(1.0) / scale;
    float shift =
        dequant_op->Op()->HasAttr("Shift")
            ? PADDLE_GET_CONST(float, dequant_op->Op()->GetAttr("Shift"))
            : 0;

    ConvertToFusedOp(transpose_op->Op());
    transpose_op->Op()->SetAttr("scale", reorder_scale);
    transpose_op->Op()->SetAttr("shift", shift);
    transpose_op->Op()->SetAttr("output_data_type", std::string("fp32"));
    transpose_op->Op()->SetOutput(
        "Out", std::vector<std::string>({dequant_out->Name()}));

    IR_NODE_LINK_TO(transpose_op, dequant_out);
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
  FuseQuantizeTranspose2(graph, "fused_transpose");
  FuseTranspose2Dequantize(graph, "fused_transpose");
  FuseQuantizeTranspose2(graph, "transpose2");
  FuseTranspose2Dequantize(graph, "transpose2");
}

FuseQuantTranspose2DequantOneDNNPass::
    FuseQuantTranspose2DequantOneDNNPass() {  // NOLINT
  AddOpCompat(OpCompat("transpose2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsOptional()
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsType<std::vector<int>>()
      .End();

  AddOpCompat(OpCompat("fused_transpose"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsOptional()
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsType<std::vector<int>>()
      .End();
}

}  // namespace paddle::framework::ir

REGISTER_PASS(quant_transpose2_dequant_onednn_fuse_pass,
              paddle::framework::ir::FuseQuantTranspose2DequantOneDNNPass);
REGISTER_PASS_CAPABILITY(quant_transpose2_dequant_onednn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "transpose2", 0));
