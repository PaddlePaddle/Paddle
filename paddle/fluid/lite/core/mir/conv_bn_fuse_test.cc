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

#include <gtest/gtest.h>
#include <memory>
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/core/mir/graph_visualize_pass.h"
#include "paddle/fluid/lite/core/mir/pattern_matcher_high_api.h"
#include "paddle/fluid/lite/core/program.h"

namespace paddle {
namespace lite {
namespace mir {

class ConvBatchNormFuser : public FuseBase {
 public:
  void BuildPattern() override {
    // create nodes.
    auto* conv_input = VarNode("conv_input")->AsInput();
    auto* conv_weight = VarNode("conv_weight")->AsInput();
    auto* conv_bias = VarNode("conv_bias")->AsInput();
    auto* conv = OpNode("conv2d", "conv2d");
    auto* conv_out = VarNode("conv_out");

    auto* bn_scale = VarNode("bn_scale");
    auto* bn_bias = VarNode("bn_bias")->AsInput();
    auto* bn_mean = VarNode("bn_mean");
    auto* bn_var = VarNode("bn_variance");
    auto* bn = OpNode("bn", "batch_norm");

    auto* bn_out = VarNode("bn_out")->AsOutput();
    auto* bn_mean_out = VarNode("bn_mean_out");
    auto* bn_var_out = VarNode("bn_var_out");
    auto* bn_saved_mean = VarNode("bn_saved_mean");
    auto* bn_saved_var = VarNode("bn_saved_var");

    conv->LinksFrom({conv_input, conv_weight, conv_bias}).LinksTo({conv_out});

    bn->LinksFrom({conv_out, bn_scale, bn_bias, bn_mean, bn_var})
        .LinksTo(
            {bn_out, bn_mean_out, bn_saved_mean, bn_saved_var, bn_var_out});

    // Some op specialities.
    bn_scale->AsIntermediate();
    bn_mean->AsIntermediate();
    bn_var->AsIntermediate();
    bn->AsIntermediate();
    bn_mean_out->AsIntermediate();
    bn_var_out->AsIntermediate();
    bn_saved_mean->AsIntermediate();
    bn_saved_var->AsIntermediate();
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    LOG(INFO) << "hello";
    auto op_desc = GenOpDesc(matched);
    auto eltwise_op = LiteOpRegistry::Global().Create("elementwise_add");
    std::cout << "1" << matched.count("conv2d") << std::endl;
    auto conv = matched.at("conv2d")->stmt()->op;
    auto* scope = conv->scope();
    auto& valid_places = conv->valid_places();
    eltwise_op->Attach(op_desc, scope);

    auto* new_op_node =
        graph->GraphCreateInstructNode(eltwise_op, valid_places);

    std::cout << "2" << matched.count("conv_out") << std::endl;
    IR_NODE_LINK_TO(matched.at("conv_out"), new_op_node);
    std::cout << "3" << matched.count("bn_bias") << std::endl;
    IR_NODE_LINK_TO(matched.at("bn_bias"), new_op_node);
    std::cout << "4" << matched.count("bn_out") << std::endl;
    IR_NODE_LINK_TO(new_op_node, matched.at("bn_out"));
  }

 private:
  cpp::OpDesc GenOpDesc(const key2nodes_t& matched) override {
    std::cout << "11" << matched.count("conv2d") << std::endl;
    std::cout << "2" << matched.count("conv_out") << std::endl;
    std::cout << "3" << matched.count("bn_bias") << std::endl;
    std::cout << "4" << matched.count("bn_out") << std::endl;
    cpp::OpDesc op_desc;
    op_desc.SetType("elementwise_add");
    op_desc.SetInput("X", {matched.at("conv_out")->arg()->name});
    op_desc.SetInput("Y", {matched.at("bn_bias")->arg()->name});
    op_desc.SetOutput("Out", {matched.at("bn_out")->arg()->name});
    op_desc.SetAttr("axis", 1);
    return op_desc;
  }
};

std::unique_ptr<SSAGraph> BuildGraph(framework::ProgramDesc* program_desc,
                                     const std::shared_ptr<Scope>& scope,
                                     const std::vector<Place>& valid_places) {
  auto* main_block = program_desc->MutableBlock(0);
  auto* conv_op = main_block->AppendOp();
  auto* bn_op = main_block->AppendOp();
  main_block->Var("conv_i");
  main_block->Var("conv_param");
  main_block->Var("conv_bias");
  main_block->Var("conv_out");

  main_block->Var("bn_scale");
  main_block->Var("bn_bias");
  main_block->Var("bn_mean");
  main_block->Var("bn_var");
  main_block->Var("bn_out");
  main_block->Var("bn_mean_out");
  main_block->Var("bn_var_out");
  main_block->Var("bn_saved_mean");
  main_block->Var("bn_saved_var");

  scope->Var("conv_i")->GetMutable<lite::Tensor>();
  scope->Var("conv_param")->GetMutable<lite::Tensor>();
  scope->Var("conv_bias")->GetMutable<lite::Tensor>();
  scope->Var("conv_out")->GetMutable<lite::Tensor>();

  scope->Var("bn_scale")->GetMutable<lite::Tensor>();
  scope->Var("bn_bias")->GetMutable<lite::Tensor>();
  scope->Var("bn_mean")->GetMutable<lite::Tensor>();
  scope->Var("bn_var")->GetMutable<lite::Tensor>();
  scope->Var("bn_out")->GetMutable<lite::Tensor>();
  scope->Var("bn_mean_out")->GetMutable<lite::Tensor>();
  scope->Var("bn_var_out")->GetMutable<lite::Tensor>();
  scope->Var("bn_saved_mean")->GetMutable<lite::Tensor>();
  scope->Var("bn_saved_var")->GetMutable<lite::Tensor>();

  conv_op->SetType("conv2d");
  conv_op->SetInput("Input", {"conv_i"});
  conv_op->SetInput("Filter", {"conv_param"});
  conv_op->SetInput("Bias", {"conv_bias"});
  conv_op->SetOutput("Output", {"conv_out"});
  const std::vector<int> strides({1, 1});
  const std::vector<int> paddings({1, 1});
  const std::vector<int> dilations({1, 1});
  const int groups = 1;
  conv_op->SetAttr("strides", strides);
  conv_op->SetAttr("paddings", paddings);
  conv_op->SetAttr("dilations", dilations);
  conv_op->SetAttr("groups", groups);

  bn_op->SetType("batch_norm");
  bn_op->SetInput("X", {"conv_out"});
  bn_op->SetInput("Bias", {"bn_bias"});
  bn_op->SetInput("Mean", {"bn_mean"});
  bn_op->SetInput("Scale", {"bn_scale"});
  bn_op->SetInput("Variance", {"bn_var"});

  bn_op->SetOutput("Y", {"bn_out"});
  bn_op->SetOutput("MeanOut", {"bn_mean"});
  bn_op->SetOutput("VarianceOut", {"bn_var_out"});
  bn_op->SetOutput("SavedMean", {"bn_saved_mean"});
  bn_op->SetOutput("SavedVariance", {"bn_saved_var"});
  float eps = 1e-5;
  bn_op->SetAttr("epsilon", eps);

  program_desc->Flush();

  lite::Program program(*program_desc->Proto(), scope, valid_places);
  auto graph = std::unique_ptr<SSAGraph>(new SSAGraph());
  graph->Build(program, valid_places);

  return graph;
}

TEST(pattern_matcher2, test) {
  framework::ProgramDesc program_desc;
  std::vector<Place> places{{TARGET(kHost), PRECISION(kFloat)}};
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(&program_desc, scope, places);
  Visualize(graph.get());
  ConvBatchNormFuser fuser;
  fuser(graph.get());
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(conv2d);
USE_LITE_OP(batch_norm);
USE_LITE_OP(elementwise_add);
