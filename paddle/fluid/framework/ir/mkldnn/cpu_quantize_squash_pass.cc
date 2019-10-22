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

#include "paddle/fluid/framework/ir/mkldnn/cpu_quantize_squash_pass.h"
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

void CPUQuantizeSquashPass::FindNodesToKeep(
    Graph* graph,
    std::unordered_map<const Node*, int>* nodes_keep_counter) const {
  GraphPatternDetector gpd;
  patterns::DequantAny deq_any_pattern{gpd.mutable_pattern(), "deqant_any"};
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

void CPUQuantizeSquashPass::DequantQuantSquash(
    Graph* graph,
    std::unordered_map<const Node*, int>* nodes_keep_counter) const {
  GraphPatternDetector gpd;
  patterns::DequantQuantAny squash_pattern{gpd.mutable_pattern(), "squash"};
  squash_pattern();

  int found_dequant_quant_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "squash requantize-quantize ops pair";

    GET_IR_NODE_FROM_SUBGRAPH(dequant_in, dequant_in, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequant_op, dequant_op, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequant_out, dequant_out, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quant_op, quant_op, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quant_out, quant_out, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(next_op, next_op, squash_pattern);

    auto* next_op_desc = next_op->Op();
    float dequant_scale = boost::get<float>(dequant_op->Op()->GetAttr("Scale"));
    float quant_scale = boost::get<float>(quant_op->Op()->GetAttr("Scale"));
    PADDLE_ENFORCE(nodes_keep_counter->find(dequant_out) !=
                   nodes_keep_counter->end());

    // check if dequantize op should be kept or removed, decrease the counter
    bool keep_dequant = (*nodes_keep_counter)[dequant_out]-- > 1;

    if (dequant_scale == quant_scale) {
      // squash dequantize-quantize to nothing
      auto quant_out_var_name = quant_out->Name();
      auto next_op_inputs = next_op_desc->InputNames();
      for (const auto& name : next_op_inputs) {
        auto input_names = next_op_desc->Input(name);
        std::replace(input_names.begin(), input_names.end(), quant_out_var_name,
                     dequant_in->Name());
        next_op_desc->SetInput(name, input_names);
      }

      if (keep_dequant)
        GraphSafeRemoveNodes(graph, {quant_op, quant_out});
      else
        GraphSafeRemoveNodes(graph,
                             {dequant_op, quant_op, dequant_out, quant_out});

      IR_NODE_LINK_TO(dequant_in, next_op);

      found_dequant_quant_count++;
    } else {
      // squash dequantize-quantize to requantize op
      OpDesc desc;
      desc.SetType("requantize");
      desc.SetInput("Input", std::vector<std::string>({dequant_in->Name()}));
      desc.SetOutput("Output", std::vector<std::string>({quant_out->Name()}));
      desc.SetAttr("Scale_in", dequant_scale);
      desc.SetAttr("Scale_out", quant_scale);

      auto requant_op = g->CreateOpNode(&desc);

      if (keep_dequant)
        GraphSafeRemoveNodes(graph, {quant_op});
      else
        GraphSafeRemoveNodes(graph, {dequant_op, quant_op, dequant_out});

      IR_NODE_LINK_TO(dequant_in, requant_op);
      IR_NODE_LINK_TO(requant_op, quant_out);

      found_dequant_quant_count++;
    }
  };
  gpd(graph, handler);
  AddStatis(found_dequant_quant_count);
  PrettyLogDetail("---    squashed %d dequantize-quantize pairs",
                  found_dequant_quant_count);
}

void CPUQuantizeSquashPass::ConvRequantSquash(Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::ConvRequant conv_requant_pattern{gpd.mutable_pattern(),
                                             "conv_requant"};
  conv_requant_pattern();

  int found_requant_squash_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "squash conv-requantize ops pair";

    GET_IR_NODE_FROM_SUBGRAPH(conv_op, conv_op, conv_requant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_out, conv_out, conv_requant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(requant_op, requant_op, conv_requant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(requant_out, requant_out, conv_requant_pattern);

    // if conv2d has one output squash
    if (conv_out->outputs.size() == 1) {
      float requant_scale_out =
          boost::get<float>(requant_op->Op()->GetAttr("Scale_out"));
      conv_op->Op()->SetAttr("Scale_out", requant_scale_out);
      conv_op->Op()->SetOutput("Output",
                               std::vector<std::string>({requant_out->Name()}));
      IR_NODE_LINK_TO(conv_op, requant_out);
      GraphSafeRemoveNodes(graph, {conv_out, requant_op});

      found_requant_squash_count++;
    }
  };
  gpd(graph, handler);
  AddStatis(found_requant_squash_count);
  PrettyLogDetail("---    squashed %d requantize with convs",
                  found_requant_squash_count);
}

void CPUQuantizeSquashPass::ConvDequantSquash(Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::ConvDequant conv_dequant_pattern{gpd.mutable_pattern(),
                                             "conv_dequant"};
  conv_dequant_pattern();

  int found_conv_dequant_squash_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "squash conv-dequant ops pair";

    GET_IR_NODE_FROM_SUBGRAPH(conv_op, conv_op, conv_dequant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_out, conv_out, conv_dequant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequant_op, dequant_op, conv_dequant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequant_out, dequant_out, conv_dequant_pattern);

    // if conv2d has one output
    // and there is no fuse residual connection
    // because residual fusion does not support force output with fp32
    if (conv_out->outputs.size() == 1 &&
        !(conv_op->Op()->GetAttrIfExists<bool>("fuse_residual_connection"))) {
      conv_op->Op()->SetAttr("force_fp32_output", true);
      conv_op->Op()->SetOutput("Output",
                               std::vector<std::string>({dequant_out->Name()}));
      IR_NODE_LINK_TO(conv_op, dequant_out);
      GraphSafeRemoveNodes(graph, {conv_out, dequant_op});
      found_conv_dequant_squash_count++;
    }
  };
  gpd(graph, handler);
  AddStatis(found_conv_dequant_squash_count);
  PrettyLogDetail("---    squashed %d dequant with convs",
                  found_conv_dequant_squash_count);
}

void CPUQuantizeSquashPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE(graph);
  FusePassBase::Init("cpu_quantize_squash_pass", graph);

  std::unordered_map<const Node*, int> nodes_keep_counter;
  FindNodesToKeep(graph, &nodes_keep_counter);
  DequantQuantSquash(graph, &nodes_keep_counter);
  ConvDequantSquash(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(cpu_quantize_squash_pass,
              paddle::framework::ir::CPUQuantizeSquashPass);
