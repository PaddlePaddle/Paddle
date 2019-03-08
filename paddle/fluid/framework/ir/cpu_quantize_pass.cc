// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/cpu_quantize_pass.h"
#include <map>
#include <utility>
#include <vector>
#include "paddle/fluid/inference/api/paddle_quantizer_config.h"  // for QuantMax

namespace paddle {
namespace framework {
namespace ir {

namespace {

void UnlinkNodes(ir::Node* a, ir::Node* b) {
  a->outputs.erase(std::remove(a->outputs.begin(), a->outputs.end(), b),
                   a->outputs.end());
  b->inputs.erase(std::remove(b->inputs.begin(), b->inputs.end(), a),
                  b->inputs.end());
}

}  // namespace

template <typename OutT>
void CPUQuantizePass::QuantizeInput(Graph* g, Node* op, Node* input,
                                    std::string input_name, std::string prefix,
                                    float scale, bool is_negative) const {
  // Create quantize output variable
  VarDesc quantize_out_desc(patterns::PDNodeName(prefix + "quantize", "out"));
  quantize_out_desc.SetDataType(framework::ToDataType(typeid(OutT)));
  quantize_out_desc.SetShape(input->Var()->GetShape());
  auto* quantize_out_node = g->CreateVarNode(&quantize_out_desc);

  // create a quantize op node
  OpDesc q_desc;
  q_desc.SetType("quantize");
  q_desc.SetInput("Input", std::vector<std::string>({input->Name()}));
  q_desc.SetOutput("Output",
                   std::vector<std::string>({quantize_out_node->Name()}));
  q_desc.SetAttr("Scale", scale);
  q_desc.SetAttr("is_negative_input", is_negative);
  auto quantize_op = g->CreateOpNode(&q_desc);  // OpDesc will be copied.

  // update op's input
  op->Op()->SetInput(input_name,
                     std::vector<std::string>({quantize_out_node->Name()}));

  // link quantize op
  UnlinkNodes(input, op);
  IR_NODE_LINK_TO(input, quantize_op);
  IR_NODE_LINK_TO(quantize_op, quantize_out_node);
  IR_NODE_LINK_TO(quantize_out_node, op);
}

template <typename InT>
void CPUQuantizePass::DequantizeOutput(Graph* g, Node* op, Node* output,
                                       std::string output_name,
                                       std::string prefix, float scale) const {
  // Create dequantize input variable
  VarDesc dequantize_in_desc(patterns::PDNodeName(prefix + "dequantize", "in"));
  dequantize_in_desc.SetDataType(framework::ToDataType(typeid(InT)));
  dequantize_in_desc.SetShape(output->Var()->GetShape());
  auto* dequantize_in_node = g->CreateVarNode(&dequantize_in_desc);

  // create a dequantize op node for output.
  OpDesc deq_desc;
  deq_desc.SetType("dequantize");
  deq_desc.SetInput("Input",
                    std::vector<std::string>({dequantize_in_node->Name()}));
  deq_desc.SetOutput("Output", std::vector<std::string>({output->Name()}));
  deq_desc.SetAttr("Scale", scale);
  auto dequantize_op = g->CreateOpNode(&deq_desc);  // OpDesc will be copied.

  // update op's output
  op->Op()->SetOutput(output_name,
                      std::vector<std::string>({dequantize_in_node->Name()}));

  // link dequantize op
  UnlinkNodes(op, output);
  IR_NODE_LINK_TO(op, dequantize_in_node);
  IR_NODE_LINK_TO(dequantize_in_node, dequantize_op);
  IR_NODE_LINK_TO(dequantize_op, output);
}

void CPUQuantizePass::QuantizeConv(Graph* graph,
                                   bool with_residual_data) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::Conv conv_pattern{pattern, name_scope_};
  conv_pattern(with_residual_data);

  int quantize_conv_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle Conv2d with residual connection quantization";
    GET_IR_NODE_FROM_SUBGRAPH(conv_op, conv_op, conv_pattern);

    auto* conv_op_desc = conv_op->Op();
    if (!conv_op_desc->HasAttr("use_quantizer") ||
        !boost::get<bool>(conv_op_desc->GetAttr("use_quantizer")))
      return;

    if (conv_op_desc->HasAttr("quantized") &&
        boost::get<bool>(conv_op_desc->GetAttr("quantized")))
      return;

    conv_op_desc->SetAttr("quantized", true);

    std::stringstream prefix_ss;
    if (with_residual_data) prefix_ss << "rc_";
    auto prefix = prefix_ss.str();

    GET_IR_NODE_FROM_SUBGRAPH(conv_filter, conv_filter, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_input, conv_input, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_output, conv_output, conv_pattern);

    auto scales = Get<VarQuantMaxAndScale>("quant_var_scales");
    auto conv_input_scale = scales[conv_input->Name()].second.data<float>()[0];
    bool is_input_negative =
        scales[conv_input->Name()].first == QuantMax::S8_MAX;
    auto conv_filter_scale_tensor = scales[conv_filter->Name()].second;
    std::vector<float> conv_filter_scale{
        conv_filter_scale_tensor.data<float>(),
        conv_filter_scale_tensor.data<float>() +
            conv_filter_scale_tensor.numel()};
    auto conv_output_scale =
        scales[conv_output->Name()].second.data<float>()[0];

    QuantizeInput<int8_t>(g, conv_op, conv_input, "Input", prefix,
                          conv_input_scale, is_input_negative);
    conv_op->Op()->SetAttr("Scale_in", conv_input_scale);
    conv_op->Op()->SetAttr("Scale_weights", conv_filter_scale);

    if (with_residual_data) {
      GET_IR_NODE_FROM_SUBGRAPH(conv_residual_data, conv_residual_data,
                                conv_pattern);
      auto conv_res_conn_scale =
          scales[conv_residual_data->Name()].second.data<float>()[0];

      QuantizeInput<int8_t>(g, conv_op, conv_residual_data, "ResidualData",
                            prefix, conv_res_conn_scale, true);
      conv_op->Op()->SetAttr("Scale_in_eltwise", conv_res_conn_scale);
    }

    DequantizeOutput<int8_t>(g, conv_op, conv_output, "Output", prefix,
                             conv_output_scale);
    conv_op->Op()->SetAttr("Scale_out", conv_output_scale);

    ++quantize_conv_count;
  };

  gpd(graph, handler);
  std::stringstream msg_ss;
  msg_ss << "---  Quantized " << quantize_conv_count << " conv2d ops";
  if (with_residual_data) msg_ss << " with residual connection";
  msg_ss << "." << std::endl;
  std::cout << msg_ss.str();
  AddStatis(quantize_conv_count);
}

void CPUQuantizePass::QuantizePool(Graph* graph) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::Pool pool_pattern{pattern, name_scope_};
  pool_pattern();

  int quantize_pool_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "Handle quantization of pool2d op";
    GET_IR_NODE_FROM_SUBGRAPH(pool_op, pool_op, pool_pattern);

    auto* pool_op_desc = pool_op->Op();
    if (!pool_op_desc->HasAttr("use_quantizer") ||
        !boost::get<bool>(pool_op_desc->GetAttr("use_quantizer")))
      return;

    if (pool_op_desc->HasAttr("quantized") &&
        boost::get<bool>(pool_op_desc->GetAttr("quantized")))
      return;

    pool_op_desc->SetAttr("quantized", true);

    GET_IR_NODE_FROM_SUBGRAPH(pool_input, pool_input, pool_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(pool_output, pool_output, pool_pattern);

    auto scales = Get<VarQuantMaxAndScale>("quant_var_scales");
    auto input_scale = scales[pool_input->Name()].second.data<float>()[0];
    bool is_input_negative =
        scales[pool_input->Name()].first == QuantMax::S8_MAX;

    std::string prefix{"q_pool"};
    QuantizeInput<int8_t>(g, pool_op, pool_input, "X", prefix, input_scale,
                          is_input_negative);
    DequantizeOutput<int8_t>(g, pool_op, pool_output, "Out", prefix,
                             input_scale);

    ++quantize_pool_count;
  };

  gpd(graph, handler);
  std::stringstream msg_ss;
  msg_ss << "---  Quantized " << quantize_pool_count << " pool2d ops."
         << std::endl;
  std::cout << msg_ss.str();
  AddStatis(quantize_pool_count);
}

std::unique_ptr<ir::Graph> CPUQuantizePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  VLOG(3) << "Quantizing the graph.";
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init(name_scope_, graph.get());

  PADDLE_ENFORCE(param_scope());

  QuantizeConv(graph.get(), true /* with_residual_data */);
  QuantizeConv(graph.get());
  QuantizePool(graph.get());

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(cpu_quantize_pass, paddle::framework::ir::CPUQuantizePass)
    .RequirePassAttr("quant_var_scales");
