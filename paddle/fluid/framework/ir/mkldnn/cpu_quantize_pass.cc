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

#include "paddle/fluid/framework/ir/mkldnn/cpu_quantize_pass.h"
#include <limits>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/string/pretty_log.h"

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

enum { U8_MAX = 255, S8_MAX = 127 };

using EigenVectorArrayMap = Eigen::Map<Eigen::Array<double, Eigen::Dynamic, 1>>;
using string::PrettyLogDetail;

void CPUQuantizePass::QuantizeInput(Graph* g, Node* op, Node* input,
                                    std::string input_name, double scale_to_one,
                                    bool is_unsigned,
                                    std::string scale_attr_name) const {
  auto inputs = op->Op()->InputNames();
  bool name_found =
      std::find(inputs.begin(), inputs.end(), input_name) != inputs.end();
  PADDLE_ENFORCE_EQ(
      name_found, true,
      platform::errors::InvalidArgument("%s isn't the input of the %s operator",
                                        input_name, op->Op()->Type()));
  unsigned max = is_unsigned ? U8_MAX : S8_MAX;
  float scale = scale_to_one * max;

  // Create quantize output variable
  VarDesc quantize_out_desc(patterns::PDNodeName("quantize", "out"));
  auto* quantize_out_node = g->CreateVarNode(&quantize_out_desc);

  // create a quantize op node
  OpDesc q_desc;
  q_desc.SetType("quantize");
  q_desc.SetInput("Input", std::vector<std::string>({input->Name()}));
  q_desc.SetOutput("Output",
                   std::vector<std::string>({quantize_out_node->Name()}));
  q_desc.SetAttr("Scale", scale);
  q_desc.SetAttr("is_negative_input", !is_unsigned);

  q_desc.SetAttr("output_format",
                 Has("data_layout") ? Get<std::string>("data_layout") : "NHWC");
  auto quantize_op = g->CreateOpNode(&q_desc);  // OpDesc will be copied.

  // update op's input
  op->Op()->SetInput(input_name,
                     std::vector<std::string>({quantize_out_node->Name()}));

  // link quantize op
  UnlinkNodes(input, op);
  IR_NODE_LINK_TO(input, quantize_op);
  IR_NODE_LINK_TO(quantize_op, quantize_out_node);
  IR_NODE_LINK_TO(quantize_out_node, op);

  if (!scale_attr_name.empty()) op->Op()->SetAttr(scale_attr_name, scale);
}

void CPUQuantizePass::QuantizeInputs(Graph* g, Node* op, std::string input_name,
                                     VarQuantScale* scales, bool are_unsigned,
                                     std::string scale_attr_name) const {
  auto inputs = op->inputs;
  auto output = op->outputs[0];
  PADDLE_ENFORCE_GE(inputs.size(), 1);
  PADDLE_ENFORCE_EQ(op->outputs.size(), 1);

  // create a quantize op desc prototype
  OpDesc q_desc;
  q_desc.SetType("quantize");

  std::vector<Node*> quantize_out_nodes(inputs.size());
  std::vector<std::string> quantize_out_node_names(inputs.size());

  double scale_out = (*scales)[output->Name()].second.data<double>()[0];
  unsigned max = are_unsigned ? U8_MAX : S8_MAX;
  float scale = scale_out * max;

  for (size_t i = 0; i < inputs.size(); i++) {
    // Create quantize output variable
    VarDesc quantize_out_desc(patterns::PDNodeName("quantize", "out"));
    quantize_out_nodes[i] = g->CreateVarNode(&quantize_out_desc);
    quantize_out_node_names[i] = quantize_out_nodes[i]->Name();

    q_desc.SetAttr("Scale", scale);
    q_desc.SetInput("Input", std::vector<std::string>({inputs[i]->Name()}));
    q_desc.SetOutput("Output",
                     std::vector<std::string>({quantize_out_node_names[i]}));
    q_desc.SetAttr("is_negative_input", !are_unsigned);
    auto quantize_op = g->CreateOpNode(&q_desc);  // OpDesc will be copied.

    // link quantize op
    UnlinkNodes(inputs[i], op);
    IR_NODE_LINK_TO(inputs[i], quantize_op);
    IR_NODE_LINK_TO(quantize_op, quantize_out_nodes[i]);
    IR_NODE_LINK_TO(quantize_out_nodes[i], op);
  }

  // update op's input
  op->Op()->SetInput(input_name, quantize_out_node_names);

  if (!scale_attr_name.empty()) op->Op()->SetAttr(scale_attr_name, scale);
}

void CPUQuantizePass::DequantizeOutput(Graph* g, Node* op, Node* output,
                                       std::string output_name,
                                       double scale_to_one, bool is_unsigned,
                                       std::string scale_attr_name) const {
  auto outputs = op->Op()->OutputNames();
  bool name_found =
      std::find(outputs.begin(), outputs.end(), output_name) != outputs.end();
  PADDLE_ENFORCE_EQ(name_found, true,
                    platform::errors::InvalidArgument(
                        "%s isn't the output of the %s operator", output_name,
                        op->Op()->Type()));
  unsigned max = is_unsigned ? U8_MAX : S8_MAX;
  float scale = scale_to_one * max;

  // Create dequantize input variable
  VarDesc dequantize_in_desc(patterns::PDNodeName("dequantize", "in"));
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

  if (!scale_attr_name.empty()) op->Op()->SetAttr(scale_attr_name, scale);
}

void CPUQuantizePass::QuantizeConv(Graph* graph,
                                   bool with_residual_data) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::ConvResidual conv_pattern{pattern, name_scope_};
  conv_pattern(with_residual_data);

  int quantize_conv_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "Quantize conv2d op";
    GET_IR_NODE_FROM_SUBGRAPH(conv_op, conv_op, conv_pattern);
    auto* conv_op_desc = conv_op->Op();

    // skip if should not be quantized
    if (!conv_op_desc->GetAttrIfExists<bool>("use_quantizer")) return;

    GET_IR_NODE_FROM_SUBGRAPH(conv_filter, conv_filter, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_input, conv_input, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_output, conv_output, conv_pattern);

    // get scales calculated after warmup, they scale variables to MAX=1.0
    auto scales = Get<VarQuantScale>("quant_var_scales");

    auto input_scale = scales[conv_input->Name()].second.data<double>()[0];
    bool is_input_unsigned = scales[conv_input->Name()].first;
    QuantizeInput(g, conv_op, conv_input, "Input", input_scale,
                  is_input_unsigned, "Scale_in");

    auto filter_scale_tensor = scales[conv_filter->Name()].second;
    EigenVectorArrayMap eigen_tensor{filter_scale_tensor.data<double>(),
                                     filter_scale_tensor.numel(), 1};
    eigen_tensor *= static_cast<double>(S8_MAX);
    std::vector<float> filter_scale{
        filter_scale_tensor.data<double>(),
        filter_scale_tensor.data<double>() + filter_scale_tensor.numel()};

    conv_op->Op()->SetAttr("Scale_weights", filter_scale);

    if (with_residual_data) {
      GET_IR_NODE_FROM_SUBGRAPH(conv_residual_data, conv_residual_data,
                                conv_pattern);
      auto residual_scale =
          scales[conv_residual_data->Name()].second.data<double>()[0];
      bool is_residual_unsigned = scales[conv_residual_data->Name()].first;

      QuantizeInput(g, conv_op, conv_residual_data, "ResidualData",
                    residual_scale, is_residual_unsigned, "Scale_in_eltwise");
    }

    auto output_scale = scales[conv_output->Name()].second.data<double>()[0];
    bool is_output_unsigned = scales[conv_output->Name()].first;
    DequantizeOutput(g, conv_op, conv_output, "Output", output_scale,
                     is_output_unsigned, "Scale_out");

    // change threshold in bounded ReLu
    if (conv_op->Op()->GetAttrIfExists<std::string>("fuse_activation") ==
        "relu6") {
      float scale_out = boost::get<float>(conv_op->Op()->GetAttr("Scale_out"));
      float threshold = boost::get<float>(conv_op->Op()->GetAttr("fuse_alpha"));
      conv_op->Op()->SetAttr("fuse_alpha", scale_out * threshold);
    }

    ++quantize_conv_count;
  };

  gpd(graph, handler);
  AddStatis(quantize_conv_count);

  std::stringstream msg_ss;
  msg_ss << "---    quantized " << quantize_conv_count << " conv2d ops";
  if (with_residual_data) msg_ss << " with residual connection";
  PrettyLogDetail(msg_ss.str().c_str());
}

void CPUQuantizePass::QuantizeFc(Graph* graph) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::FCMKLDNN fc_pattern{pattern, name_scope_};
  auto* fc_input = gpd.mutable_pattern()
                       ->NewNode("fc_quantizer/input")
                       ->AsInput()
                       ->assert_is_op_input("fc", "Input");
  fc_pattern(fc_input, false);

  int quantize_fc_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "Quantize fc op";
    GET_IR_NODE_FROM_SUBGRAPH(fc, fc, fc_pattern);
    auto* fc_op_desc = fc->Op();

    // skip if should not be quantized
    if (fc_op_desc->GetAttrIfExists<bool>("use_quantizer") != true ||
        fc_op_desc->GetAttrIfExists<bool>("use_mkldnn") != true)
      return;

    GET_IR_NODE_FROM_SUBGRAPH(weights, weights, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(input, input, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(output, output, fc_pattern);

    // get scales calculated after warmup, they scale variables to MAX=1.0
    auto scales = Get<VarQuantScale>("quant_var_scales");

    auto input_scale = scales[input->Name()].second.data<double>()[0];
    bool is_input_unsigned = scales[input->Name()].first;
    QuantizeInput(g, fc, input, "Input", input_scale, is_input_unsigned,
                  "Scale_in");

    auto weight_scale_tensor = scales[weights->Name()].second;
    EigenVectorArrayMap eigen_tensor{weight_scale_tensor.data<double>(),
                                     weight_scale_tensor.numel(), 1};
    eigen_tensor *= static_cast<double>(S8_MAX);
    std::vector<float> filter_scale{
        weight_scale_tensor.data<double>(),
        weight_scale_tensor.data<double>() + weight_scale_tensor.numel()};

    fc->Op()->SetAttr("Scale_weights", filter_scale);

    auto output_scale = scales[output->Name()].second.data<double>()[0];
    bool is_output_unsigned = scales[output->Name()].first;
    DequantizeOutput(g, fc, output, "Out", output_scale, is_output_unsigned,
                     "Scale_out");

    ++quantize_fc_count;
  };

  gpd(graph, handler);
  AddStatis(quantize_fc_count);

  std::stringstream msg_ss;
  msg_ss << "---    quantized " << quantize_fc_count << " fc ops";
  PrettyLogDetail(msg_ss.str().c_str());
}

void CPUQuantizePass::QuantizePool(Graph* graph) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::Pool pool_pattern{pattern, name_scope_};
  pool_pattern();

  int quantize_pool_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "Quantize pool2d op";
    GET_IR_NODE_FROM_SUBGRAPH(pool_op, pool_op, pool_pattern);
    auto* pool_op_desc = pool_op->Op();

    // skip if should not be quantized
    if (!pool_op_desc->GetAttrIfExists<bool>("use_quantizer")) return;

    GET_IR_NODE_FROM_SUBGRAPH(pool_input, pool_input, pool_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(pool_output, pool_output, pool_pattern);

    // get scales calculated after warmup, they scale variables to MAX=1.0
    auto scales = Get<VarQuantScale>("quant_var_scales");

    auto input_scale = scales[pool_input->Name()].second.data<double>()[0];
    bool is_input_unsigned = scales[pool_input->Name()].first;
    QuantizeInput(g, pool_op, pool_input, "X", input_scale, is_input_unsigned);

    auto output_scale = scales[pool_output->Name()].second.data<double>()[0];
    bool is_output_unsigned = scales[pool_output->Name()].first;
    DequantizeOutput(g, pool_op, pool_output, "Out", output_scale,
                     is_output_unsigned);

    ++quantize_pool_count;
  };

  gpd(graph, handler);
  AddStatis(quantize_pool_count);

  PrettyLogDetail("---    quantized %d pool2d ops", quantize_pool_count);
}

void CPUQuantizePass::QuantizeConcat(Graph* graph) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::Concat concat_pattern{pattern, name_scope_};
  concat_pattern();

  int quantize_concat_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "Quantize concat op";
    GET_IR_NODE_FROM_SUBGRAPH(concat_op, concat_op, concat_pattern);
    auto* concat_op_desc = concat_op->Op();

    // skip if should not be quantized
    if (!concat_op_desc->GetAttrIfExists<bool>("use_quantizer")) return;

    GET_IR_NODE_FROM_SUBGRAPH(concat_out, concat_out, concat_pattern);

    // get scales calculated after warmup, they scale variables to MAX=1.0
    auto scales = Get<VarQuantScale>("quant_var_scales");

    // if all inputs were unsigned, then the output was set to unsigned
    // during the scale calculation step
    bool are_all_inputs_unsigned = scales[concat_out->Name()].first;
    QuantizeInputs(g, concat_op, "X", &scales, are_all_inputs_unsigned);

    auto output_scale = scales[concat_out->Name()].second.data<double>()[0];

    DequantizeOutput(g, concat_op, concat_out, "Out", output_scale,
                     are_all_inputs_unsigned);

    ++quantize_concat_count;
  };

  gpd(graph, handler);
  AddStatis(quantize_concat_count);

  PrettyLogDetail("---    quantized %d concat ops", quantize_concat_count);
}

void CPUQuantizePass::QuantizePriorBox(Graph* graph) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::PriorBox prior_box_pattern{pattern, name_scope_};
  prior_box_pattern();

  int quantize_prior_box_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "Quantize prior_box op";
    GET_IR_NODE_FROM_SUBGRAPH(prior_box_op, prior_box_op, prior_box_pattern);
    auto* prior_box_op_desc = prior_box_op->Op();

    // skip if should not be quantized
    if (!prior_box_op_desc->GetAttrIfExists<bool>("use_quantizer")) return;

    GET_IR_NODE_FROM_SUBGRAPH(prior_box_input, prior_box_input,
                              prior_box_pattern);

    // get scales calculated after warmup, they scale variables to MAX=1.0
    auto scales = Get<VarQuantScale>("quant_var_scales");

    auto input_scale = scales[prior_box_input->Name()].second.data<double>()[0];
    bool is_input_unsigned = scales[prior_box_input->Name()].first;
    QuantizeInput(g, prior_box_op, prior_box_input, "Input", input_scale,
                  is_input_unsigned);

    ++quantize_prior_box_count;
  };

  gpd(graph, handler);
  AddStatis(quantize_prior_box_count);

  PrettyLogDetail("---    quantized %d prior_box ops",
                  quantize_prior_box_count);
}

void CPUQuantizePass::QuantizeTranspose(Graph* graph) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::Transpose transpose_pattern{pattern, name_scope_};
  transpose_pattern();

  int quantize_transpose_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "Quantize transpose op";
    GET_IR_NODE_FROM_SUBGRAPH(transpose_op, transpose_op, transpose_pattern);
    auto* transpose_op_desc = transpose_op->Op();

    // skip if should not be quantized
    if (!transpose_op_desc->GetAttrIfExists<bool>("use_quantizer")) {
      return;
    }
    GET_IR_NODE_FROM_SUBGRAPH(prev_op, prev_op, transpose_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(next_op, next_op, transpose_pattern);

    // skip if prev op and next op are not quantized
    if (!(prev_op->Op()->Type() == "dequantize" ||
          (prev_op->Op()->GetAttrIfExists<bool>("use_quantizer"))) &&
        !(next_op->Op()->Type() == "quantize" ||
          (next_op->Op()->GetAttrIfExists<bool>("use_quantizer")))) {
      return;
    }
    GET_IR_NODE_FROM_SUBGRAPH(transpose_in, transpose_in, transpose_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose_out, transpose_out, transpose_pattern);

    // get scales calculated after warmup, they scale variables to MAX=1.0
    auto scales = Get<VarQuantScale>("quant_var_scales");
    auto input_scale = scales[transpose_in->Name()].second.data<double>()[0];
    bool is_input_unsigned = scales[transpose_in->Name()].first;
    QuantizeInput(g, transpose_op, transpose_in, "X", input_scale,
                  is_input_unsigned);

    auto output_scale = scales[transpose_out->Name()].second.data<double>()[0];
    bool is_output_unsigned = scales[transpose_out->Name()].first;
    DequantizeOutput(g, transpose_op, transpose_out, "Out", output_scale,
                     is_output_unsigned);

    ++quantize_transpose_count;
  };

  gpd(graph, handler);
  AddStatis(quantize_transpose_count);

  PrettyLogDetail("---    quantized %d transpose ops",
                  quantize_transpose_count);
}

void CPUQuantizePass::QuantizeReshape(Graph* graph) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::Reshape reshape_pattern{pattern, name_scope_};
  reshape_pattern();

  int quantize_reshape_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "Quantize reshape op";
    GET_IR_NODE_FROM_SUBGRAPH(reshape_op, reshape_op, reshape_pattern);
    auto* reshape_op_desc = reshape_op->Op();

    // skip if should not be quantized
    if (!reshape_op_desc->GetAttrIfExists<bool>("use_quantizer")) {
      return;
    }
    GET_IR_NODE_FROM_SUBGRAPH(prev_op, prev_op, reshape_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(next_op, next_op, reshape_pattern);

    // skip if prev op  and next op is not quantized
    if (!(prev_op->Op()->Type() == "dequantize" ||
          (prev_op->Op()->GetAttrIfExists<bool>("use_quantizer"))) &&
        !(next_op->Op()->Type() == "quantize" ||
          (next_op->Op()->GetAttrIfExists<bool>("use_quantizer")))) {
      return;
    }

    GET_IR_NODE_FROM_SUBGRAPH(reshape_in, reshape_in, reshape_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape_out, reshape_out, reshape_pattern);

    // get scales calculated after warmup, they scale variables to MAX=1.0
    auto scales = Get<VarQuantScale>("quant_var_scales");
    auto input_scale = scales[reshape_in->Name()].second.data<double>()[0];
    bool is_input_unsigned = scales[reshape_in->Name()].first;
    QuantizeInput(g, reshape_op, reshape_in, "X", input_scale,
                  is_input_unsigned);

    auto output_scale = scales[reshape_out->Name()].second.data<double>()[0];
    bool is_output_unsigned = scales[reshape_out->Name()].first;
    DequantizeOutput(g, reshape_op, reshape_out, "Out", output_scale,
                     is_output_unsigned);

    ++quantize_reshape_count;
  };

  gpd(graph, handler);
  AddStatis(quantize_reshape_count);

  PrettyLogDetail("---    quantized %d reshape ops", quantize_reshape_count);
}

void CPUQuantizePass::ApplyImpl(ir::Graph* graph) const {
  VLOG(3) << "Quantizing the graph.";
  PADDLE_ENFORCE(graph);
  FusePassBase::Init(name_scope_, graph);

  PADDLE_ENFORCE(param_scope());

  QuantizeConv(graph, false /* with_residual_data */);
  QuantizeConv(graph, true /* with_residual_data */);
  QuantizePool(graph);
  QuantizeConcat(graph);
  QuantizePriorBox(graph);
  QuantizeTranspose(graph);
  QuantizeFc(graph);
  QuantizeReshape(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(cpu_quantize_pass, paddle::framework::ir::CPUQuantizePass)
    .RequirePassAttr("quant_var_scales");
