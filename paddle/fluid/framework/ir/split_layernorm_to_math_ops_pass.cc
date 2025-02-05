// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/split_layernorm_to_math_ops_pass.h"

#include <vector>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/utils/string/pretty_log.h"
#include "paddle/utils/string/printf.h"

namespace paddle::framework::ir {

// cpplint complaints (wrong!) for not included <string> header in below line.
using string::PrettyLogDetail;  // NOLINT

SplitLayerNormPass::SplitLayerNormPass() {
  AddOpCompat(OpCompat("layer_norm"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Scale")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .End()
      .AddOutput("Y")
      .IsTensor()
      .End()
      .AddOutput("Mean")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Variance")
      .IsTensor()
      .IsOptional()
      .End()
      .AddAttr("epsilon")
      .IsNumGE(0.0f)
      .IsNumLE(0.001f)
      .End()
      .AddAttr("begin_norm_axis")
      .End();
  AddOpCompat(OpCompat("reduce_mean"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("dim")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("keep_dim")
      .End();
  AddOpCompat(OpCompat("sqrt"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();
  AddOpCompat(OpCompat("elementwise_sub"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .End();
  AddOpCompat(OpCompat("elementwise_pow"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .End();
  AddOpCompat(OpCompat("elementwise_add"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .End();
  AddOpCompat(OpCompat("elementwise_div"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .End();
  AddOpCompat(OpCompat("elementwise_mul"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .End();
}

void SplitLayerNormPass::ApplyImpl(Graph* graph) const {
  FusePassBase::Init(scope_name_, graph);

  auto* scope = param_scope();

  GraphPatternDetector gpd;
  patterns::SplitLayerNorm layer_norm_pattern(gpd.mutable_pattern(),
                                              scope_name_);
  layer_norm_pattern();

  int found_layer_norm_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    VLOG(4) << "Split LayerNorm from subgraph.";
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_in, layer_norm_in, layer_norm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_op, layer_norm_op, layer_norm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_bias, layer_norm_bias, layer_norm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_scale, layer_norm_scale, layer_norm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_out, layer_norm_out, layer_norm_pattern);

    auto* input_var = layer_norm_in->Var();
    const std::vector<int64_t> input_shape = input_var->GetShape();
    int begin_norm_axis =
        PADDLE_GET_CONST(int, layer_norm_op->Op()->GetAttr("begin_norm_axis"));
    float eps =
        PADDLE_GET_CONST(float, layer_norm_op->Op()->GetAttr("epsilon"));

    std::vector<int32_t> reduce_dim;
    std::vector<int64_t> shape_int64;
    int feature_size = 1;
    for (int i = begin_norm_axis; i < static_cast<int>(input_shape.size());
         i++) {
      feature_size *= input_shape[i];
      reduce_dim.push_back(i);
      shape_int64.push_back(input_shape[i]);
    }

    // small feature size has low performance
    constexpr int FEATURE_SIZE_THRESHOLD = 128;
    if (feature_size > FEATURE_SIZE_THRESHOLD) {
      return;
    }
    // since gamma and beta are constant vars, dynamic shape should be useless
    auto min_input_shape =
        Get<std::map<std::string, std::vector<int>>>("min_input_shape");
    if (min_input_shape.find(layer_norm_in->Name()) != min_input_shape.end()) {
      auto max_input_shape =
          Get<std::map<std::string, std::vector<int>>>("max_input_shape");
      auto opt_input_shape =
          Get<std::map<std::string, std::vector<int>>>("optim_input_shape");
      auto min_shape = min_input_shape[layer_norm_in->Name()];
      auto max_shape = max_input_shape[layer_norm_in->Name()];
      auto opt_shape = opt_input_shape[layer_norm_in->Name()];

      for (int i = begin_norm_axis; i < static_cast<int>(input_shape.size());
           i++) {
        if (min_shape[i] != max_shape[i] || max_shape[i] != opt_shape[i]) {
          return;
        }
      }
    }
    auto* dev_ctx = static_cast<phi::CPUContext*>(
        phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
    auto reduce_mean0_out_name(
        patterns::PDNodeName("split_layernorm", "reduce0"));
    auto* block = layer_norm_op->Op()->Block();
    OpDesc reduce_mean0(block);
    reduce_mean0.SetType("reduce_mean");
    reduce_mean0.SetInput("X", {layer_norm_in->Name()});
    reduce_mean0.SetOutput("Out", {reduce_mean0_out_name});
    reduce_mean0.SetAttr("reduce_all", false);
    reduce_mean0.SetAttr("keep_dim", true);
    reduce_mean0.SetAttr("dim", reduce_dim);
    reduce_mean0.Flush();
    auto reduce_mean0_node = g->CreateOpNode(&reduce_mean0);
    auto* reduce_mean0_out = block->Var(reduce_mean0_out_name);
    reduce_mean0_out->SetShape(input_shape);
    reduce_mean0_out->SetDataType(input_var->GetDataType());
    reduce_mean0_out->SetLoDLevel(layer_norm_in->Var()->GetLoDLevel());
    auto* reduce_mean0_out_node = graph->CreateVarNode(reduce_mean0_out);

    auto sub_out_name(
        patterns::PDNodeName("split_layernorm", "elementwise_sub"));
    OpDesc elementwise_sub(block);
    elementwise_sub.SetType("elementwise_sub");
    elementwise_sub.SetInput("X", {layer_norm_in->Name()});
    elementwise_sub.SetInput("Y", {reduce_mean0_out_name});
    elementwise_sub.SetOutput("Out", {sub_out_name});
    elementwise_sub.SetAttr("axis", -1);
    elementwise_sub.Flush();
    auto elementwise_sub_node = g->CreateOpNode(&elementwise_sub);
    auto* sub_out = block->Var(sub_out_name);
    sub_out->SetShape(input_shape);
    sub_out->SetDataType(input_var->GetDataType());
    sub_out->SetLoDLevel(layer_norm_in->Var()->GetLoDLevel());
    auto* sub_out_node = graph->CreateVarNode(sub_out);

    auto pow_out_name(
        patterns::PDNodeName("split_layernorm", "elementwise_pow_out"));
    auto pow_y_name(
        patterns::PDNodeName("split_layernorm", "elementwise_pow_y"));
    OpDesc elementwise_pow(block);
    elementwise_pow.SetType("elementwise_pow");
    elementwise_pow.SetInput("X", {sub_out_name});
    elementwise_pow.SetInput("Y", {pow_y_name});
    elementwise_pow.SetOutput("Out", {pow_out_name});
    elementwise_pow.SetAttr("axis", -1);
    elementwise_pow.Flush();
    auto elementwise_pow_node = g->CreateOpNode(&elementwise_pow);
    auto* pow_out = block->Var(pow_out_name);
    pow_out->SetShape(input_shape);
    pow_out->SetDataType(input_var->GetDataType());
    pow_out->SetLoDLevel(layer_norm_in->Var()->GetLoDLevel());
    auto* pow_out_node = graph->CreateVarNode(pow_out);
    auto* pow_y = block->Var(pow_y_name);
    pow_y->SetShape({1});
    pow_y->SetDataType(input_var->GetDataType());
    pow_y->SetLoDLevel(layer_norm_in->Var()->GetLoDLevel());
    pow_y->SetPersistable(true);
    auto* pow_y_node = graph->CreateVarNode(pow_y);
    auto* pow_y_tensor = scope->Var(pow_y_name)->GetMutable<phi::DenseTensor>();
    pow_y_tensor->Resize(common::make_ddim({1}));
    dev_ctx->Alloc<float>(pow_y_tensor);
    (pow_y_tensor->data<float>())[0] = 2.0f;

    auto reduce_mean1_out_name(
        patterns::PDNodeName("split_layernorm", "reduce1"));
    OpDesc reduce_mean1(block);
    reduce_mean1.SetType("reduce_mean");
    reduce_mean1.SetInput("X", {pow_out_name});
    reduce_mean1.SetOutput("Out", {reduce_mean1_out_name});
    reduce_mean1.SetAttr("reduce_all", false);
    reduce_mean1.SetAttr("keep_dim", true);
    reduce_mean1.SetAttr("dim", reduce_dim);
    reduce_mean1.Flush();
    auto reduce_mean1_node = g->CreateOpNode(&reduce_mean1);
    auto* reduce_mean1_out = block->Var(reduce_mean1_out_name);
    reduce_mean1_out->SetShape(input_shape);
    reduce_mean1_out->SetDataType(input_var->GetDataType());
    reduce_mean1_out->SetLoDLevel(layer_norm_in->Var()->GetLoDLevel());
    auto* reduce_mean1_out_node = graph->CreateVarNode(reduce_mean1_out);

    auto add0_out_name(patterns::PDNodeName("split_layernorm", "add0"));
    auto add_y_name(
        patterns::PDNodeName("split_layernorm", "elementwise_add_y"));
    OpDesc elementwise_add0(block);
    elementwise_add0.SetType("elementwise_add");
    elementwise_add0.SetInput("X", {reduce_mean1_out_name});
    elementwise_add0.SetInput("Y", {add_y_name});
    elementwise_add0.SetOutput("Out", {add0_out_name});
    elementwise_add0.SetAttr("axis", -1);
    elementwise_add0.Flush();
    auto elementwise_add0_node = g->CreateOpNode(&elementwise_add0);
    auto* add0_out = block->Var(add0_out_name);
    add0_out->SetShape(input_shape);
    add0_out->SetDataType(input_var->GetDataType());
    add0_out->SetLoDLevel(layer_norm_in->Var()->GetLoDLevel());
    auto* add0_out_node = graph->CreateVarNode(add0_out);
    auto* add_y = block->Var(add_y_name);
    add_y->SetShape({1});
    add_y->SetDataType(input_var->GetDataType());
    add_y->SetLoDLevel(layer_norm_in->Var()->GetLoDLevel());
    add_y->SetPersistable(true);
    auto* add_y_node = graph->CreateVarNode(add_y);
    auto* add_y_tensor = scope->Var(add_y_name)->GetMutable<phi::DenseTensor>();
    add_y_tensor->Resize(common::make_ddim({1}));
    dev_ctx->Alloc<float>(add_y_tensor);
    (add_y_tensor->data<float>())[0] = eps;

    auto sqrt_out_name(patterns::PDNodeName("split_layernorm", "sqrt"));
    OpDesc sqrt(block);
    sqrt.SetType("sqrt");
    sqrt.SetInput("X", {add0_out_name});
    sqrt.SetOutput("Out", {sqrt_out_name});
    sqrt.Flush();
    auto sqrt_node = g->CreateOpNode(&sqrt);
    auto* sqrt_out = block->Var(sqrt_out_name);
    sqrt_out->SetShape(input_shape);
    sqrt_out->SetDataType(input_var->GetDataType());
    sqrt_out->SetLoDLevel(layer_norm_in->Var()->GetLoDLevel());
    auto* sqrt_out_node = graph->CreateVarNode(sqrt_out);

    auto div_out_name(patterns::PDNodeName("split_layernorm", "div"));
    OpDesc elementwise_div(block);
    elementwise_div.SetType("elementwise_div");
    elementwise_div.SetInput("X", {sub_out_name});
    elementwise_div.SetInput("Y", {sqrt_out_name});
    elementwise_div.SetOutput("Out", {div_out_name});
    elementwise_div.SetAttr("axis", 0);
    elementwise_div.Flush();
    auto elementwise_div_node = g->CreateOpNode(&elementwise_div);
    auto* div_out = block->Var(div_out_name);
    div_out->SetShape(input_shape);
    div_out->SetDataType(input_var->GetDataType());
    div_out->SetLoDLevel(layer_norm_in->Var()->GetLoDLevel());
    auto* div_out_node = graph->CreateVarNode(div_out);

    auto mul_out_name(patterns::PDNodeName("split_layernorm", "mul"));
    auto new_scale_name(patterns::PDNodeName("split_layernorm", "new_scale"));
    OpDesc elementwise_mul(block);
    elementwise_mul.SetType("elementwise_mul");
    elementwise_mul.SetInput("X", {div_out_name});
    elementwise_mul.SetInput("Y", {new_scale_name});
    elementwise_mul.SetOutput("Out", {mul_out_name});
    elementwise_mul.SetAttr("axis", -1);
    elementwise_mul.Flush();
    auto elementwise_mul_node = g->CreateOpNode(&elementwise_mul);
    auto* scale = block->Var(new_scale_name);
    scale->SetShape(shape_int64);
    scale->SetDataType(layer_norm_scale->Var()->GetDataType());
    scale->SetLoDLevel(layer_norm_scale->Var()->GetLoDLevel());
    scale->SetPersistable(true);
    auto* new_scale_node = graph->CreateVarNode(scale);
    auto* new_scale_tensor =
        scope->Var(new_scale_name)->GetMutable<phi::DenseTensor>();
    auto* scale_tensor =
        scope->Var(layer_norm_scale->Name())->GetMutable<phi::DenseTensor>();
    new_scale_tensor->Resize(common::make_ddim(shape_int64));
    dev_ctx->Alloc<float>(new_scale_tensor);
    memcpy(new_scale_tensor->data<float>(),
           scale_tensor->data<float>(),
           sizeof(float) * feature_size);
    auto* mul_out = block->Var(mul_out_name);
    mul_out->SetShape(input_shape);
    mul_out->SetDataType(input_var->GetDataType());
    mul_out->SetLoDLevel(layer_norm_in->Var()->GetLoDLevel());
    auto* mul_out_node = graph->CreateVarNode(mul_out);

    auto new_bias_name(patterns::PDNodeName("split_layernorm", "new_bias"));
    OpDesc elementwise_add1(block);
    elementwise_add1.SetType("elementwise_add");
    elementwise_add1.SetInput("X", {mul_out_name});
    elementwise_add1.SetInput("Y", {new_bias_name});
    elementwise_add1.SetOutput("Out", {layer_norm_out->Name()});
    elementwise_add1.SetAttr("axis", -1);
    elementwise_add1.Flush();
    auto* new_bias = block->Var(new_bias_name);
    new_bias->SetShape(shape_int64);
    new_bias->SetDataType(layer_norm_bias->Var()->GetDataType());
    new_bias->SetLoDLevel(layer_norm_bias->Var()->GetLoDLevel());
    new_bias->SetPersistable(true);
    auto* new_bias_node = graph->CreateVarNode(new_bias);
    auto* new_bias_tensor =
        scope->Var(new_bias_name)->GetMutable<phi::DenseTensor>();
    auto* bias_tensor =
        scope->Var(layer_norm_bias->Name())->GetMutable<phi::DenseTensor>();
    new_bias_tensor->Resize(common::make_ddim(shape_int64));
    dev_ctx->Alloc<float>(new_bias_tensor);
    memcpy(new_bias_tensor->data<float>(),
           bias_tensor->data<float>(),
           sizeof(float) * feature_size);
    auto elementwise_add1_node = g->CreateOpNode(&elementwise_add1);

    IR_NODE_LINK_TO(layer_norm_in, reduce_mean0_node);
    IR_NODE_LINK_TO(reduce_mean0_node, reduce_mean0_out_node);
    IR_NODE_LINK_TO(reduce_mean0_out_node, elementwise_sub_node);
    IR_NODE_LINK_TO(layer_norm_in, elementwise_sub_node);
    IR_NODE_LINK_TO(elementwise_sub_node, sub_out_node);
    IR_NODE_LINK_TO(sub_out_node, elementwise_pow_node);
    IR_NODE_LINK_TO(pow_y_node, elementwise_pow_node);
    IR_NODE_LINK_TO(elementwise_pow_node, pow_out_node);
    IR_NODE_LINK_TO(pow_out_node, reduce_mean1_node);
    IR_NODE_LINK_TO(reduce_mean1_node, reduce_mean1_out_node);
    IR_NODE_LINK_TO(reduce_mean1_out_node, elementwise_add0_node);
    IR_NODE_LINK_TO(add_y_node, elementwise_add0_node);
    IR_NODE_LINK_TO(elementwise_add0_node, add0_out_node);
    IR_NODE_LINK_TO(add0_out_node, sqrt_node);
    IR_NODE_LINK_TO(sqrt_node, sqrt_out_node);
    IR_NODE_LINK_TO(sqrt_out_node, elementwise_div_node);
    IR_NODE_LINK_TO(sub_out_node, elementwise_div_node);
    IR_NODE_LINK_TO(elementwise_div_node, div_out_node);
    IR_NODE_LINK_TO(div_out_node, elementwise_mul_node);
    IR_NODE_LINK_TO(new_scale_node, elementwise_mul_node);
    IR_NODE_LINK_TO(elementwise_mul_node, mul_out_node);
    IR_NODE_LINK_TO(mul_out_node, elementwise_add1_node);
    IR_NODE_LINK_TO(new_bias_node, elementwise_add1_node);
    IR_NODE_LINK_TO(elementwise_add1_node, layer_norm_out);

    std::unordered_set<const Node*> nodes2rm = {};
    nodes2rm.insert(layer_norm_op);
    if (layer_norm_bias->outputs.size() <= 1UL)
      nodes2rm.insert(layer_norm_bias);
    if (layer_norm_scale->outputs.size() <= 1UL)
      nodes2rm.insert(layer_norm_scale);

    GraphSafeRemoveNodes(g, nodes2rm);
    found_layer_norm_count++;
  };

  gpd(graph, handler);
  AddStatis(found_layer_norm_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(split_layernorm_to_math_ops_pass,
              paddle::framework::ir::SplitLayerNormPass);
REGISTER_PASS_CAPABILITY(split_layernorm_to_math_ops_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("elementwise_add", 1)
            .LE("elementwise_div", 1)
            .LE("elementwise_mul", 1)
            .LE("elementwise_pow", 1)
            .LE("elementwise_sub", 1)
            .EQ("reduce_mean", 0)
            .EQ("sqrt", 0));
