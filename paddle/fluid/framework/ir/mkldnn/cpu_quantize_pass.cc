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
#include <sstream>
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
                                     bool are_unsigned,
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

  double scale_out = GetScaleValueForNode(output);
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

bool CPUQuantizePass::AreScalesPresentForNodes(
    const Node* op_node, std::initializer_list<Node*> nodes) const {
  auto& scales = Get<VarQuantScale>("quant_var_scales");
  bool present = true;
  for (auto node : nodes) {
    if (scales.count(node->Name()) == 0) {
      present = false;
      std::stringstream msg_ss;
      msg_ss << "Quantization scale for the variable " << node->Name()
             << " is missing.";
      PrettyLogDetail(msg_ss.str().c_str());
    }
  }
  if (!present) {
    std::stringstream msg_ss;
    msg_ss << "Cannot quantize operator " << op_node->Name()
           << " (type: " << op_node->Op()->Type() << ").";
    PrettyLogDetail(msg_ss.str().c_str());
  }
  return present;
}

std::pair<bool, LoDTensor> CPUQuantizePass::GetScaleDataForNode(
    const Node* node) const {
  auto& scales = Get<VarQuantScale>("quant_var_scales");
  return scales[node->Name()];
}

LoDTensor CPUQuantizePass::GetScaleTensorForNode(const Node* node) const {
  return GetScaleDataForNode(node).second;
}

double CPUQuantizePass::GetScaleValueForNode(const Node* node,
                                             bool* is_unsigned) const {
  auto scale_data = GetScaleDataForNode(node);
  if (is_unsigned != nullptr) *is_unsigned = scale_data.first;
  return scale_data.second.data<double>()[0];
}

bool CPUQuantizePass::IsOpDequantized(const Node* node) const {
  return node->Op()->Type() == "dequantize" ||
         node->Op()->GetAttrIfExists<bool>("use_quantizer");
}

bool CPUQuantizePass::IsOpQuantized(const Node* node) const {
  return node->Op()->Type() == "quantize" ||
         node->Op()->GetAttrIfExists<bool>("use_quantizer");
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

    if (with_residual_data) {
      GET_IR_NODE_FROM_SUBGRAPH(conv_residual_data, conv_residual_data,
                                conv_pattern);
      if (!AreScalesPresentForNodes(conv_op, {conv_input, conv_filter,
                                              conv_residual_data, conv_output}))
        return;

      bool is_residual_unsigned{false};
      auto residual_scale =
          GetScaleValueForNode(conv_residual_data, &is_residual_unsigned);

      QuantizeInput(g, conv_op, conv_residual_data, "ResidualData",
                    residual_scale, is_residual_unsigned, "Scale_in_eltwise");
    } else {
      if (!AreScalesPresentForNodes(conv_op,
                                    {conv_input, conv_filter, conv_output}))
        return;
    }

    bool is_input_unsigned{false};
    auto input_scale = GetScaleValueForNode(conv_input, &is_input_unsigned);
    QuantizeInput(g, conv_op, conv_input, "Input", input_scale,
                  is_input_unsigned, "Scale_in");

    auto filter_scale_tensor = GetScaleTensorForNode(conv_filter);
    EigenVectorArrayMap eigen_tensor{filter_scale_tensor.data<double>(),
                                     filter_scale_tensor.numel(), 1};
    eigen_tensor *= static_cast<double>(S8_MAX);
    std::vector<float> filter_scale{
        filter_scale_tensor.data<double>(),
        filter_scale_tensor.data<double>() + filter_scale_tensor.numel()};

    conv_op->Op()->SetAttr("Scale_weights", filter_scale);

    bool is_output_unsigned{false};
    auto output_scale = GetScaleValueForNode(conv_output, &is_output_unsigned);
    DequantizeOutput(g, conv_op, conv_output, "Output", output_scale,
                     is_output_unsigned, "Scale_out");

    // change threshold in bounded ReLu
    if (conv_op->Op()->GetAttrIfExists<std::string>("fuse_activation") ==
        "relu6") {
      float scale_out =
          BOOST_GET_CONST(float, conv_op->Op()->GetAttr("Scale_out"));
      float threshold =
          BOOST_GET_CONST(float, conv_op->Op()->GetAttr("fuse_alpha"));
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

    if (!AreScalesPresentForNodes(fc, {input, weights, output})) return;

    bool is_input_unsigned{false};
    auto input_scale = GetScaleValueForNode(input, &is_input_unsigned);
    QuantizeInput(g, fc, input, "Input", input_scale, is_input_unsigned,
                  "Scale_in");

    auto weight_scale_tensor = GetScaleTensorForNode(weights);
    EigenVectorArrayMap eigen_tensor{weight_scale_tensor.data<double>(),
                                     weight_scale_tensor.numel(), 1};
    eigen_tensor *= static_cast<double>(S8_MAX);
    std::vector<float> filter_scale{
        weight_scale_tensor.data<double>(),
        weight_scale_tensor.data<double>() + weight_scale_tensor.numel()};

    fc->Op()->SetAttr("Scale_weights", filter_scale);

    bool is_output_unsigned{false};
    auto output_scale = GetScaleValueForNode(output, &is_output_unsigned);
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

    if (!AreScalesPresentForNodes(pool_op, {pool_input, pool_output})) return;

    bool is_input_unsigned{false};
    auto input_scale = GetScaleValueForNode(pool_input, &is_input_unsigned);
    QuantizeInput(g, pool_op, pool_input, "X", input_scale, is_input_unsigned);

    bool is_output_unsigned{false};
    auto output_scale = GetScaleValueForNode(pool_output, &is_output_unsigned);
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

    if (!AreScalesPresentForNodes(concat_op, {concat_out})) return;

    // if all inputs were unsigned, then the output was set to unsigned
    // during the scale calculation step
    bool are_all_inputs_unsigned{false};
    auto output_scale =
        GetScaleValueForNode(concat_out, &are_all_inputs_unsigned);

    QuantizeInputs(g, concat_op, "X", are_all_inputs_unsigned);

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

    if (!AreScalesPresentForNodes(prior_box_op, {prior_box_input})) return;

    bool is_input_unsigned{false};
    auto input_scale =
        GetScaleValueForNode(prior_box_input, &is_input_unsigned);
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

    // skip if prev op and next op is not quantized
    if (!(IsOpDequantized(prev_op)) && !(IsOpQuantized(next_op))) {
      return;
    }
    GET_IR_NODE_FROM_SUBGRAPH(transpose_in, transpose_in, transpose_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose_out, transpose_out, transpose_pattern);

    if (!AreScalesPresentForNodes(transpose_op, {transpose_in, transpose_out}))
      return;

    bool is_input_unsigned{false};
    auto input_scale = GetScaleValueForNode(transpose_in, &is_input_unsigned);
    QuantizeInput(g, transpose_op, transpose_in, "X", input_scale,
                  is_input_unsigned);

    bool is_output_unsigned{false};
    auto output_scale =
        GetScaleValueForNode(transpose_out, &is_output_unsigned);
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

    // skip if prev op and next op is not quantized
    if (!(IsOpDequantized(prev_op)) && !(IsOpQuantized(next_op))) {
      return;
    }

    GET_IR_NODE_FROM_SUBGRAPH(reshape_in, reshape_in, reshape_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape_out, reshape_out, reshape_pattern);

    if (!AreScalesPresentForNodes(reshape_op, {reshape_in, reshape_out}))
      return;

    bool is_input_unsigned{false};
    auto input_scale = GetScaleValueForNode(reshape_in, &is_input_unsigned);
    QuantizeInput(g, reshape_op, reshape_in, "X", input_scale,
                  is_input_unsigned);

    bool is_output_unsigned{false};
    auto output_scale = GetScaleValueForNode(reshape_out, &is_output_unsigned);
    DequantizeOutput(g, reshape_op, reshape_out, "Out", output_scale,
                     is_output_unsigned);

    ++quantize_reshape_count;
  };

  gpd(graph, handler);
  AddStatis(quantize_reshape_count);

  PrettyLogDetail("---    quantized %d reshape ops", quantize_reshape_count);
}

void CPUQuantizePass::QuantizeMatmul(Graph* graph) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::Matmul matmul_pattern{pattern, name_scope_};
  matmul_pattern();

  int quantize_matmul_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "Quantize matmul op";
    GET_IR_NODE_FROM_SUBGRAPH(matmul_op, matmul_op, matmul_pattern);
    auto* matmul_op_desc = matmul_op->Op();

    // skip if should not be quantized
    if (!matmul_op_desc->GetAttrIfExists<bool>("use_quantizer")) {
      return;
    }
    GET_IR_NODE_FROM_SUBGRAPH(prev_op_x, prev_op_x, matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(prev_op_y, prev_op_y, matmul_pattern);

    // skip if prev ops are not quantized
    if (!IsOpDequantized(prev_op_x) || !IsOpDequantized(prev_op_y)) {
      return;
    }
    GET_IR_NODE_FROM_SUBGRAPH(matmul_in_x, matmul_in_x, matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_in_y, matmul_in_y, matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_out, matmul_out, matmul_pattern);

    if (!AreScalesPresentForNodes(matmul_op,
                                  {matmul_in_x, matmul_in_y, matmul_out}))
      return;

    bool is_x_unsigned{false}, is_y_unsigned{false};
    auto input_x_scale = GetScaleValueForNode(matmul_in_x, &is_x_unsigned);
    auto input_y_scale = GetScaleValueForNode(matmul_in_y, &is_y_unsigned);
    PADDLE_ENFORCE_EQ(
        is_x_unsigned, is_y_unsigned,
        platform::errors::InvalidArgument(
            "Matmul inputs should have the same value of is_unsigned"));
    QuantizeInput(g, matmul_op, matmul_in_x, "X", input_x_scale, is_x_unsigned,
                  "Scale_x");
    QuantizeInput(g, matmul_op, matmul_in_y, "Y", input_y_scale, is_y_unsigned,
                  "Scale_y");

    bool is_output_unsigned{false};
    auto output_scale = GetScaleValueForNode(matmul_out, &is_output_unsigned);
    DequantizeOutput(g, matmul_op, matmul_out, "Out", output_scale,
                     is_output_unsigned, "Scale_out");

    ++quantize_matmul_count;
  };
  gpd(graph, handler);
  AddStatis(quantize_matmul_count);

  PrettyLogDetail("---    quantized %d matmul ops", quantize_matmul_count);
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
  QuantizeMatmul(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(cpu_quantize_pass, paddle::framework::ir::CPUQuantizePass)
    .RequirePassAttr("quant_var_scales");
