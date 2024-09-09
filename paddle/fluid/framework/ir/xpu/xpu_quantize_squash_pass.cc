// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file eint8_outcept in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either eint8_outpress or
// implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/ir/xpu/xpu_quantize_squash_pass.h"

#include <string>
#include <vector>

#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/ir/xpu/xpu_graph_pattern_detector.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/utils/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

XPUQuantizeSquashPass::XPUQuantizeSquashPass() {}

void XPUQuantizeSquashPass::FindNodesToKeep(
    Graph* graph,
    std::unordered_map<const Node*, int>* nodes_keep_counter) const {
  GraphPatternDetector gpd;
  patterns::DequantXPUAny deq_any_pattern{gpd.mutable_pattern(),
                                          "dequant_xpu_any"};
  deq_any_pattern();

  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(dequant_out, dequant_out, deq_any_pattern);

    if (nodes_keep_counter->find(dequant_out) == nodes_keep_counter->end())
      (*nodes_keep_counter)[dequant_out] = 1;
    else
      (*nodes_keep_counter)[dequant_out] += 1;

    found_count++;
  };
  gpd(graph, handler);
  AddStatis(found_count);
}

void XPUQuantizeSquashPass::DequantQuantSquash(
    Graph* graph,
    std::unordered_map<const Node*, int>* nodes_keep_counter) const {
  GraphPatternDetector gpd;
  patterns::DequantQuantXPUAny squash_pattern{gpd.mutable_pattern(),
                                              "dequant_quant_xpu_any"};
  squash_pattern();

  int found_dequant_quant_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(dequant_in, dequant_in, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequant_op, dequant_op, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequant_out, dequant_out, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quant_op, quant_op, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quant_out, quant_out, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(next_op, next_op, squash_pattern);

    auto* next_op_desc = next_op->Op();
    float dequant_scale =
        PADDLE_GET_CONST(float, dequant_op->Op()->GetAttr("scale"));
    float quant_scale =
        PADDLE_GET_CONST(float, quant_op->Op()->GetAttr("scale"));

    PADDLE_ENFORCE_NE(
        nodes_keep_counter->find(dequant_out),
        nodes_keep_counter->end(),
        common::errors::NotFound("The dequant output node is not found."));

    // check if dequantize op should be kept or removed, decrease the counter
    bool keep_dequant = (*nodes_keep_counter)[dequant_out]-- > 1;

    if (dequant_scale == quant_scale) {
      // squash dequantize-quantize to nothing
      auto quant_out_var_name = quant_out->Name();
      for (auto input_name : next_op_desc->InputNames()) {
        auto& input_names = next_op_desc->MutableInputs()->at(input_name);
        std::replace(input_names.begin(),
                     input_names.end(),
                     quant_out_var_name,
                     dequant_in->Name());
        next_op_desc->SetInput(input_name, input_names);
      }
      if (keep_dequant)
        GraphSafeRemoveNodes(graph, {quant_op, quant_out});
      else
        GraphSafeRemoveNodes(graph,
                             {dequant_op, quant_op, dequant_out, quant_out});

      IR_NODE_LINK_TO(dequant_in, next_op);

      found_dequant_quant_count++;
    }
  };
  gpd(graph, handler);
  AddStatis(found_dequant_quant_count);
  PrettyLogDetail("---    squashed %d dequantize-quantize pairs",
                  found_dequant_quant_count);
}

void XPUQuantizeSquashPass::OpDequantSquash(Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::OpDequantXPU op_dequant_pattern{gpd.mutable_pattern(),
                                            "op_dequant_xpu"};
  op_dequant_pattern();

  int found_op_dequant_squash_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "squash op-dequant ops pair";
    GET_IR_NODE_FROM_SUBGRAPH(any_op, any_op, op_dequant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequant_in, dequant_in, op_dequant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequant_op, dequant_op, op_dequant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequant_out, dequant_out, op_dequant_pattern);

    if (dequant_in->outputs.size() == 1) {
      // Find the name of the output linking any_op to dequant_in
      std::string output_name =
          FindOutputNameByVarName(any_op->Op(), dequant_in->Name());

      if (output_name.empty()) return;
      if (any_op->Op()->Type() == "conv2d_xpu") {
        bool has_branch = any_op->Op()->HasInput("branch");
        if (has_branch) {
          std::string branch_name = any_op->Op()->Input("branch")[0];
          auto* branch_node = FindNodeWithName(graph, branch_name);
          // If branch datatype is not equal to dequant_out datatype, can not
          // squash. Because phase1: dequantize + quantize squash maybe squash
          // branch quantize, if so, We judge the datatype to decide whether to
          // squash. If squash, the result will be wrong.
          if (branch_node->Var()->GetDataType() !=
              dequant_out->Var()->GetDataType()) {
            return;
          }
        }
      }
      any_op->Op()->SetAttr("out_dtype", dequant_out->Var()->GetDataType());
      any_op->Op()->SetOutput(output_name,
                              std::vector<std::string>({dequant_out->Name()}));
      IR_NODE_LINK_TO(any_op, dequant_out);
      GraphSafeRemoveNodes(graph, {dequant_in, dequant_op});
      found_op_dequant_squash_count++;
    }
  };
  gpd(graph, handler);
  AddStatis(found_op_dequant_squash_count);
  PrettyLogDetail("---    squashed %d dequant with ops",
                  found_op_dequant_squash_count);
}

// conv2d_xpu, fc_xpu
void XPUQuantizeSquashPass::QuantOpSquash(Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::QuantXPUAny quant_any_pattern{gpd.mutable_pattern(),
                                          "quant_xpu_any"};
  quant_any_pattern();

  int found_quant_op_squash_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "squash op-dequant ops pair";

    GET_IR_NODE_FROM_SUBGRAPH(quant_in, quant_in, quant_any_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quant_op, quant_op, quant_any_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quant_out, quant_out, quant_any_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(next_op, next_op, quant_any_pattern);

    if (quant_out->outputs.size() == 1) {
      std::string input_name =
          FindInputNameByVarName(next_op->Op(), quant_out->Name());

      if (input_name.empty()) return;
      // Only support quant + conv2d_xpu/fc_xpu fusion
      if (!(next_op->Op()->Type() == "conv2d_xpu" ||
            next_op->Op()->Type() == "fc_xpu")) {
        return;
      }
      if (next_op->Op()->Type() == "conv2d_xpu" && input_name == "branch") {
        return;
      }
      next_op->Op()->SetInput(input_name,
                              std::vector<std::string>({quant_in->Name()}));
      IR_NODE_LINK_TO(quant_in, next_op);
      GraphSafeRemoveNodes(graph, {quant_out, quant_op});
      found_quant_op_squash_count++;
    }
  };
  gpd(graph, handler);
  AddStatis(found_quant_op_squash_count);
  PrettyLogDetail("---    squashed %d quantize with ops",
                  found_quant_op_squash_count);
}

// conv2d_xpu
void XPUQuantizeSquashPass::QuantConv2dFusionDequantSquash(Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::QuantConv2dFusionDequantXPU quant_conv2d_fusion_dequant_pattern{
      gpd.mutable_pattern(), "quant_conv2d_fusion_dequant_xpu"};
  quant_conv2d_fusion_dequant_pattern();

  int found_quant_op_squash_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "squash conv2d_xpu op [branch quantize - out dequantize pair]";

    GET_IR_NODE_FROM_SUBGRAPH(
        quant_in, quant_in, quant_conv2d_fusion_dequant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        quant_op, quant_op, quant_conv2d_fusion_dequant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        quant_out, quant_out, quant_conv2d_fusion_dequant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        conv_op, conv_op, quant_conv2d_fusion_dequant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        conv_out, conv_out, quant_conv2d_fusion_dequant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        dequant_op, dequant_op, quant_conv2d_fusion_dequant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        dequant_out, dequant_out, quant_conv2d_fusion_dequant_pattern);

    if (quant_out->outputs.size() == 1 && dequant_out->outputs.size() == 1) {
      conv_op->Op()->SetInput("branch",
                              std::vector<std::string>({quant_in->Name()}));
      conv_op->Op()->SetOutput("out",
                               std::vector<std::string>({dequant_out->Name()}));
      conv_op->Op()->SetAttr("out_dtype", dequant_out->Var()->GetDataType());
      IR_NODE_LINK_TO(quant_in, conv_op);
      IR_NODE_LINK_TO(conv_op, dequant_out);
      GraphSafeRemoveNodes(graph, {quant_op, quant_out, conv_out, dequant_op});
      found_quant_op_squash_count++;
    }
  };
  gpd(graph, handler);
  AddStatis(found_quant_op_squash_count);
  PrettyLogDetail("---    squashed %d branch quantize - out dequantize pairs",
                  found_quant_op_squash_count);
}

void XPUQuantizeSquashPass::MultipleQuantizeSquash(Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::MultipleQuantizeXPU multiple_quantize_pattern{
      gpd.mutable_pattern(), "multiple_quantize_xpu"};
  multiple_quantize_pattern();

  int found_multiple_quantize_squash_count = 0;
  int removed_quantize = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "fuse multiple quantize ops";

    GET_IR_NODE_FROM_SUBGRAPH(prev_out, prev_out, multiple_quantize_pattern);

    auto* first_quant_op = *(std::find_if(
        prev_out->outputs.begin(), prev_out->outputs.end(), [&](Node* node) {
          return (node->IsOp() && node->Op()->Type() == "quantize_xpu");
        }));
    auto* first_quant_out = first_quant_op->outputs[0];
    float scale = first_quant_op->Op()->GetAttrIfExists<float>("scale");

    PADDLE_ENFORCE_NE(scale,
                      0,
                      common::errors::InvalidArgument(
                          "Quantize scale(%f) should not be equal 0.", scale));

    for (int iter = prev_out->outputs.size() - 1; iter >= 0; iter--) {
      auto quant_op = prev_out->outputs[iter];
      if (quant_op->IsOp() && quant_op->Op()->Type() == "quantize_xpu" &&
          quant_op->id() != first_quant_op->id() &&
          quant_op->Op()->GetAttrIfExists<float>("scale") == scale) {
        auto quant_out = quant_op->outputs[0];
        auto last_op = quant_out->outputs[0];
        auto last_op_op = last_op->Op();

        std::string last_op_input_name =
            FindInputNameByVarName(last_op_op, quant_out->Name());

        PADDLE_ENFORCE_NE(
            last_op_input_name.empty(),
            true,
            common::errors::NotFound("Operator after quantize operator(%s) "
                                     "should have quantize output as input.",
                                     quant_out->Name()));

        // update the next operator input,
        // by replacing quant_out with first_quant_out
        auto last_op_names = last_op->Op()->Inputs().at(last_op_input_name);
        std::replace(last_op_names.begin(),
                     last_op_names.end(),
                     quant_out->Name(),
                     first_quant_out->Name());
        last_op_op->SetInput(last_op_input_name,
                             std::vector<std::string>(last_op_names));

        IR_NODE_LINK_TO(first_quant_out, last_op);
        GraphSafeRemoveNodes(graph, {quant_op, quant_out});
        removed_quantize++;
      }
    }
    found_multiple_quantize_squash_count++;
  };
  gpd(graph, handler);
  AddStatis(found_multiple_quantize_squash_count);
  PrettyLogDetail("---    squashed %d quantize op", removed_quantize);
}

void XPUQuantizeSquashPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph,
      common::errors::InvalidArgument(
          "The graph in function XPUQuantizeSquashPass::ApplyImpl is null."));
  FusePassBase::Init("xpu_quantize_squash_pass", graph);

  std::unordered_map<const Node*, int> nodes_keep_counter;
  FindNodesToKeep(graph, &nodes_keep_counter);
  DequantQuantSquash(graph, &nodes_keep_counter);
  OpDequantSquash(graph);
  QuantOpSquash(graph);
  QuantConv2dFusionDequantSquash(graph);
  MultipleQuantizeSquash(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(xpu_quantize_squash_pass,
              paddle::framework::ir::XPUQuantizeSquashPass);
