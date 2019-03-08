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

#include "paddle/fluid/framework/ir/cpu_quantize_squash_pass.h"
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void CPUQuantizeSquashPass::FindNodesToKeep(
    Graph* graph,
    std::unordered_map<const Node*, int>& nodes_keep_counter) const {
  GraphPatternDetector gpd;
  patterns::DequantAny deq_any_pattern{gpd.mutable_pattern(),
                                       "deqant_not_quant"};
  deq_any_pattern();

  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    auto* scope = param_scope();
    PADDLE_ENFORCE(scope);

    GET_IR_NODE_FROM_SUBGRAPH(dequant_out, dequant_out, deq_any_pattern);

    if (nodes_keep_counter.find(dequant_out) == nodes_keep_counter.end())
      nodes_keep_counter[dequant_out] = 1;
    else
      nodes_keep_counter[dequant_out] += 1;
    found_count++;
  };
  gpd(graph, handler);
  AddStatis(found_count);
}

void CPUQuantizeSquashPass::Squash(
    Graph* graph,
    std::unordered_map<const Node*, int>& nodes_keep_counter) const {
  GraphPatternDetector gpd;
  patterns::DequantQuantRM squash_pattern{gpd.mutable_pattern(), "squash_pass"};
  squash_pattern();

  int found_squash_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle cpu quantize squash pass";

    auto* scope = param_scope();
    PADDLE_ENFORCE(scope);

    GET_IR_NODE_FROM_SUBGRAPH(dequant, dequantize, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quant, quantize, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(next_op, next_op, squash_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(int8_out, int8_out, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequant_out, dequant_out, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quant_out, quant_out, squash_pattern);

    auto* next_op_desc = next_op->Op();
    float dequant_scale = boost::get<float>(dequant->Op()->GetAttr("Scale"));
    float quant_scale = boost::get<float>(quant->Op()->GetAttr("Scale"));
    bool is_negative =
        boost::get<bool>(quant->Op()->GetAttr("is_negative_input"));
    PADDLE_ENFORCE(nodes_keep_counter.find(dequant_out) !=
                   nodes_keep_counter.end());
    bool keep_dequant = nodes_keep_counter[dequant_out]-- > 1;

    if (dequant_scale == quant_scale) {
      auto quant_out_var_name = quant_out->Name();
      auto next_op_inputs = next_op_desc->InputNames();
      for (auto name : next_op_inputs) {
        auto var_name = next_op_desc->Input(name)[0];
        if (var_name.compare(quant_out_var_name) == 0) {
          next_op_desc->SetInput(name,
                                 std::vector<std::string>({int8_out->Name()}));
          break;
        }
      }
      // remove the dequantize and quantize op
      if (keep_dequant)
        GraphSafeRemoveNodes(graph, {quant, quant_out});
      else
        GraphSafeRemoveNodes(graph, {dequant, quant, dequant_out, quant_out});

      IR_NODE_LINK_TO(int8_out, next_op);

      found_squash_count++;
    } else {
      // Create an requantize Node
      OpDesc desc;
      desc.SetType("requantize");
      desc.SetInput("Input", std::vector<std::string>({int8_out->Name()}));
      desc.SetOutput("Output", std::vector<std::string>({quant_out->Name()}));
      desc.SetAttr("Scale_in", dequant_scale);
      desc.SetAttr("Scale_out", quant_scale);

      auto requant_node = g->CreateOpNode(&desc);  // OpDesc will be copied.

      if (keep_dequant)
        GraphSafeRemoveNodes(graph, {quant});
      else
        GraphSafeRemoveNodes(graph, {dequant, quant, dequant_out});

      IR_NODE_LINK_TO(int8_out, requant_node);
      IR_NODE_LINK_TO(requant_node, quant_out);

      found_squash_count++;
    }
  };
  gpd(graph, handler);
  std::cout << "--  squashed " << found_squash_count << std::endl;
  AddStatis(found_squash_count);
}

std::unique_ptr<ir::Graph> CPUQuantizeSquashPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init("cpu_quantize_squash_pass", graph.get());

  std::unordered_map<const Node*, int> nodes_keep_counter;
  FindNodesToKeep(graph.get(), nodes_keep_counter);
  Squash(graph.get(), nodes_keep_counter);

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(cpu_quantize_squash_pass,
              paddle::framework::ir::CPUQuantizeSquashPass);
