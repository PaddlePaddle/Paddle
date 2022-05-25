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

#include "paddle/fluid/framework/ir/mkldnn/params_quantization_mkldnn_pass.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

namespace {

template <typename T_out>
void QuantizeParams(const LoDTensor& params, LoDTensor* int_params,
                    const std::vector<float>& scales) {
  auto length = params.numel() / scales.size();
  int_params->Resize(params.dims());
  auto int_params_data = int_params->mutable_data<T_out>(CPUPlace());
  const float* params_data = params.data<float>();
  for (int64_t i = 0; i < params.numel(); i++) {
    int_params_data[i] =
        static_cast<T_out>(std::round(params_data[i] * scales[i / length]));
  }
}

ir::Node* FindOpInput(ir::Node* op, const std::string& input_name) {
  auto op_input_it =
      std::find_if(op->inputs.begin(), op->inputs.end(),
                   [&](ir::Node* node) { return node->Name() == input_name; });

  PADDLE_ENFORCE_NE(
      op_input_it, op->inputs.end(),
      platform::errors::InvalidArgument("Not found input %s of operator %s.",
                                        input_name, op->Op()->Type()));
  return (*op_input_it);
}

void ConnectInput(ir::Node* conv_op, const std::string& input_name,
                  ir::Node* int_node) {
  conv_op->Op()->SetInput(input_name, {int_node->Name()});
  IR_NODE_LINK_TO(int_node, conv_op);
}

bool HasBias(ir::Node* conv_op) {
  auto input_names = conv_op->Op()->InputNames();
  return std::find(input_names.begin(), input_names.end(), "Bias") !=
             input_names.end() &&
         conv_op->Op()->Input("Bias").size() > 0;
}

bool ShouldSkipConv(ir::Node* conv_op, Scope* scope, ir::Node* conv_filter) {
  if (!platform::HasOpINT8DataType(conv_op->Op())) {
    return true;
  }

  auto filter_var = scope->FindVar(conv_filter->Name());
  auto filter_v = conv_filter->Var();
  if (filter_var == nullptr || filter_v == nullptr ||
      filter_var->Get<LoDTensor>().dtype() != phi::DataType::FLOAT32) {
    VLOG(4) << "Skipping convolution (id: " << conv_op->id()
            << ") because it's a bug that it is detected again.";
    return true;
  }

  VLOG(4) << "Not skipping convolution (id: " << conv_op->id() << ")";
  return false;
}

VarDesc CreatePersistableVarDesc(const std::string& name,
                                 const proto::VarType_Type& type,
                                 const std::vector<int64_t>& shape) {
  VarDesc var_desc(name);
  var_desc.SetShape(shape);
  var_desc.SetDataType(type);
  var_desc.SetPersistable(true);
  return var_desc;
}

void QuantizeConvFilter(Scope* scope, ir::Graph* g,
                        const std::string& name_scope, ir::Node* conv_op,
                        ir::Node* conv_filter) {
  VarDesc int_weights_desc = CreatePersistableVarDesc(
      patterns::PDNodeName(name_scope, "conv2d_int8_filter"),
      proto::VarType_Type::VarType_Type_INT8, conv_filter->Var()->GetShape());

  ir::Node* int_weights_node = g->CreateVarNode(&int_weights_desc);
  auto* int_weights =
      scope->Var(int_weights_node->Name())->GetMutable<LoDTensor>();

  const auto scale_weights =
      conv_op->Op()->GetAttrIfExists<std::vector<float>>("Scale_weights");
  QuantizeParams<int8_t>(scope->FindVar(conv_filter->Name())->Get<LoDTensor>(),
                         int_weights, scale_weights);
  conv_op->Op()->SetAttr("Scale_weights", std::vector<float>(1, 1));

  ConnectInput(conv_op, "Filter", int_weights_node);
  GraphSafeRemoveNodes(g, {conv_filter});
}

void QuantizeConvBias(Scope* scope, ir::Graph* g, const std::string& name_scope,
                      ir::Node* conv_op) {
  std::string conv_bias_name = conv_op->Op()->Input("Bias")[0];
  auto bias = scope->FindVar(conv_bias_name)->Get<LoDTensor>();

  VarDesc int_biases_desc = CreatePersistableVarDesc(
      patterns::PDNodeName(name_scope, "conv2d_int32_bias"),
      proto::VarType::Type::VarType_Type_INT32, phi::vectorize(bias.dims()));

  ir::Node* int_biases_node = g->CreateVarNode(&int_biases_desc);
  auto* int_bias = scope->Var(int_biases_node->Name())->GetMutable<LoDTensor>();

  const auto bias_scales =
      conv_op->Op()->GetAttrIfExists<std::vector<float>>("Bias_scales");
  QuantizeParams<int32_t>(bias, int_bias, bias_scales);
  conv_op->Op()->SetAttr("Bias_scales", std::vector<float>(1, 1));

  ConnectInput(conv_op, "Bias", int_biases_node);
  GraphSafeRemoveNodes(g, {FindOpInput(conv_op, conv_bias_name)});
}

}  // namespace

ParamsQuantizationMkldnnPass::ParamsQuantizationMkldnnPass() {
  AddOpCompat(OpCompat("conv2d"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("Filter")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .IsOptional()
      .End()
      .AddInput("ResidualData")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Output")
      .IsTensor()
      .End()
      .AddAttr("strides")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("paddings")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("padding_algorithm")
      .IsOptional()
      .IsStringIn({"EXPLICIT", "SAME", "VALID"})
      .End()
      .AddAttr("groups")
      .IsNumGE(1)
      .End()
      .AddAttr("dilations")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("data_format")
      .IsStringIn({"NCHW", "AnyLayout"})
      .End();
}

void ParamsQuantizationMkldnnPass::Conv(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::Conv conv_pattern(gpd.mutable_pattern(), name_scope_);
  conv_pattern();

  int params_to_int8_conv_found = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    VLOG(4) << "handle convolution in params_quantization_mkldnn_pass";

    GET_IR_NODE_FROM_SUBGRAPH(conv_op, conv_op, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_input, conv_input, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_filter, conv_filter, conv_pattern);

    // get scope to interact with tensors
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, platform::errors::InvalidArgument("Scope cannot be nullptr."));

    if (ShouldSkipConv(conv_op, scope, conv_filter)) {
      return;
    }

    QuantizeConvFilter(scope, g, name_scope_, conv_op, conv_filter);

    if (HasBias(conv_op)) {
      QuantizeConvBias(scope, g, name_scope_, conv_op);
    }
    params_to_int8_conv_found++;
  };
  gpd(graph, handler);
  AddStatis(params_to_int8_conv_found);

  std::stringstream msg_ss;
  msg_ss << "Quantized params of " << params_to_int8_conv_found
         << " conv2d ops";
  paddle::string::PrettyLogDetail(msg_ss.str().c_str());
}

void ParamsQuantizationMkldnnPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          platform::errors::InvalidArgument(
                              "Pointer to graph argument should not be NULL."));
  FusePassBase::Init(name_scope_, graph);
  Conv(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(params_quantization_mkldnn_pass,
              paddle::framework::ir::ParamsQuantizationMkldnnPass);
REGISTER_PASS_CAPABILITY(params_quantization_mkldnn_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().LE(
            "conv2d", 1));
