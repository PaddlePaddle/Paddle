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

#include "paddle/fluid/framework/ir/mkldnn/params_to_int8_pass.h"

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

  PADDLE_ENFORCE_NE(op_input_it, op->inputs.end(),
                    platform::errors::InvalidArgument("Op input not found."));
  return (*op_input_it);
}

void ConnectNode(ir::Node* conv_op, const std::string& input_name,
                 ir::Node* int_node) {
  conv_op->Op()->SetInput(input_name, {int_node->Name()});
  IR_NODE_LINK_TO(int_node, conv_op);
}

};  // namespace

ParamsToInt8Pass::ParamsToInt8Pass() {
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

void ParamsToInt8Pass::Conv(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::Conv conv_pattern(gpd.mutable_pattern(), name_scope);
  conv_pattern();

  int params_to_int8_conv_found = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    VLOG(4) << "handle convolution params_to_int8_pass";

    GET_IR_NODE_FROM_SUBGRAPH(conv_op, conv_op, conv_pattern);

    if (!platform::HasOpINT8DataType(conv_op->Op())) {
      return;
    }

    GET_IR_NODE_FROM_SUBGRAPH(conv_input, conv_input, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_filter, conv_filter, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_output, conv_output, conv_pattern);

    std::vector<int64_t> weights_tz = conv_filter->Var()->GetShape();
    const int groups =
        std::max(conv_op->Op()->GetAttrIfExists<int>("groups"), 1);
    platform::GetGroupConvWeightsTz(weights_tz, groups);

    const auto& scale_weights_data =
        conv_op->Op()->GetAttrIfExists<std::vector<float>>("Scale_weights");

    bool is_multi_channel = scale_weights_data.size() > 1;

    int64_t scale_count = 1;
    if (is_multi_channel) {
      scale_count *= weights_tz[0];
      if (groups > 1) {
        scale_count *= weights_tz[1];
      }
    }

    // get scope to interact with tensors
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, platform::errors::InvalidArgument("Scope cannot be nullptr."));

    // get float weights
    auto weights = scope->FindVar(conv_filter->Name())->Get<LoDTensor>();

    if (weights.dtype() != phi::DataType::FLOAT32) {
      VLOG(4) << "Skipping convolution (id: " << conv_op->id()
              << ") because it's a bug that it is detected again.";
      return;
    }

    VarDesc int_weights_desc = CreatePersistableVarDesc(
        "conv2d_int8_weights", proto::VarType_Type::VarType_Type_INT8,
        weights_tz);

    ir::Node* int_weights_node = g->CreateVarNode(&int_weights_desc);
    auto* int_weights_tensor =
        scope->Var(int_weights_node->Name())->GetMutable<LoDTensor>();

    PADDLE_ENFORCE_EQ(scale_count, scale_weights_data.size());
    QuantizeParams<int8_t>(weights, int_weights_tensor, scale_weights_data);

    ConnectNode(conv_op, "Filter", int_weights_node);
    std::unordered_set<const ir::Node*> nodes_to_remove{conv_filter};

    conv_op->Op()->SetAttr("Scale_weights", std::vector<float>(1, 1));

    // Get float biases
    auto input_names = conv_op->Op()->InputNames();
    bool has_bias = std::find(input_names.begin(), input_names.end(), "Bias") !=
                        input_names.end() &&
                    conv_op->Op()->Input("Bias").size() > 0;
    if (has_bias) {
      std::string conv_bias_name = conv_op->Op()->Input("Bias")[0];
      auto bias = scope->FindVar(conv_bias_name)->GetMutable<LoDTensor>();
      PADDLE_ENFORCE_EQ(scale_count, bias->numel());

      VarDesc int_biases_desc = CreatePersistableVarDesc(
          "conv2d_int32_biases", proto::VarType::Type::VarType_Type_INT32,
          phi::vectorize(bias->dims()));

      ir::Node* int_biases_node = g->CreateVarNode(&int_biases_desc);
      auto* int_biases_tensor =
          scope->Var(int_biases_node->Name())->GetMutable<LoDTensor>();

      const auto& scale_bias_data =
          conv_op->Op()->GetAttrIfExists<std::vector<float>>("Bias_scales");

      PADDLE_ENFORCE_EQ(scale_count, scale_bias_data.size());
      QuantizeParams<int32_t>(*bias, int_biases_tensor, scale_bias_data);

      ConnectNode(conv_op, "Bias", int_biases_node);
      nodes_to_remove.insert(FindOpInput(conv_op, conv_bias_name));

      conv_op->Op()->SetAttr("Bias_scales", std::vector<float>(1, 1));
    }
    GraphSafeRemoveNodes(graph, nodes_to_remove);
    params_to_int8_conv_found++;
  };
  gpd(graph, handler);
  AddStatis(params_to_int8_conv_found);

  std::stringstream msg_ss;
  msg_ss << "Quantized weights of " << params_to_int8_conv_found
         << " conv2d ops";
  paddle::string::PrettyLogDetail(msg_ss.str().c_str());
}

VarDesc ParamsToInt8Pass::CreatePersistableVarDesc(
    const std::string& name, const proto::VarType_Type& type,
    const std::vector<int64_t>& shape) const {
  VarDesc var_desc(patterns::PDNodeName(name_scope, "conv2d_int8_weights"));
  var_desc.SetShape(shape);
  var_desc.SetDataType(type);
  var_desc.SetPersistable(true);
  return var_desc;
}

void ParamsToInt8Pass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          platform::errors::InvalidArgument(
                              "Pointer to graph argument should not be NULL."));
  FusePassBase::Init("params_to_int8_pass", graph);
  Conv(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(params_to_int8_pass, paddle::framework::ir::ParamsToInt8Pass);
REGISTER_PASS_CAPABILITY(params_to_int8_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().LE(
            "conv2d", 1));
