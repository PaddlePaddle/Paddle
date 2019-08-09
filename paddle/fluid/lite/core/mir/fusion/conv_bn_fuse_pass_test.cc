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

#include "paddle/fluid/lite/core/mir/fusion/conv_bn_fuse_pass.h"
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <vector>
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/core/mir/graph_visualize_pass.h"
#include "paddle/fluid/lite/core/program.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

std::unique_ptr<SSAGraph> BuildGraph(framework::ProgramDesc* program_desc,
                                     const std::shared_ptr<Scope>& scope,
                                     const std::vector<Place>& valid_places) {
  auto* main_block = program_desc->MutableBlock(0);
  auto* conv_op = main_block->AppendOp();
  auto* bn_op = main_block->AppendOp();
  main_block->Var("conv_i");
  main_block->Var("conv_param");
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
  auto conv_param_t = scope->Var("conv_param")->GetMutable<lite::Tensor>();
  std::vector<int64_t> conv_param_shape = {3, 1, 2, 2};
  conv_param_t->Resize(lite::DDim(conv_param_shape));
  conv_param_t->mutable_data<float>();
  scope->Var("conv_out")->GetMutable<lite::Tensor>();
  auto bn_scale_t = scope->Var("bn_scale")->GetMutable<lite::Tensor>();
  std::vector<int64_t> bn_scale_shape = {3};
  bn_scale_t->Resize(lite::DDim(bn_scale_shape));
  bn_scale_t->mutable_data<float>();

  auto bn_bias_t = scope->Var("bn_bias")->GetMutable<lite::Tensor>();
  std::vector<int64_t> bn_bias_shape = {3};
  bn_bias_t->Resize(lite::DDim(bn_bias_shape));
  bn_bias_t->mutable_data<float>();

  auto bn_mean_t = scope->Var("bn_mean")->GetMutable<lite::Tensor>();
  bn_mean_t->Resize(lite::DDim(bn_bias_shape));
  bn_mean_t->mutable_data<float>();

  auto bn_var_t = scope->Var("bn_var")->GetMutable<lite::Tensor>();
  bn_var_t->Resize(lite::DDim(bn_bias_shape));
  bn_var_t->mutable_data<float>();

  scope->Var("bn_out")->GetMutable<lite::Tensor>();
  scope->Var("bn_mean_out")->GetMutable<lite::Tensor>();
  scope->Var("bn_var_out")->GetMutable<lite::Tensor>();
  scope->Var("bn_saved_mean")->GetMutable<lite::Tensor>();
  scope->Var("bn_saved_var")->GetMutable<lite::Tensor>();

  conv_op->SetType("conv2d");
  conv_op->SetInput("Input", {"conv_i"});
  conv_op->SetInput("Filter", {"conv_param"});
  conv_op->SetOutput("Output", {"conv_out"});
  const std::vector<int> strides({1, 1});
  const std::vector<int> paddings({1, 1});
  const std::vector<int> dilations({1, 1});
  const int groups = 1;
  conv_op->SetAttr("strides", strides);
  conv_op->SetAttr("paddings", paddings);
  conv_op->SetAttr("dilations", dilations);
  conv_op->SetAttr("groups", groups);
  conv_op->SetAttr("fuse_relu", false);

  bn_op->SetType("batch_norm");
  bn_op->SetInput("X", {"conv_out"});
  bn_op->SetInput("Bias", {"bn_bias"});
  bn_op->SetInput("Mean", {"bn_mean"});
  bn_op->SetInput("Scale", {"bn_scale"});
  bn_op->SetInput("Variance", {"bn_var"});

  bn_op->SetOutput("Y", {"bn_out"});
  bn_op->SetOutput("MeanOut", {"bn_mean_out"});
  bn_op->SetOutput("VarianceOut", {"bn_var_out"});
  bn_op->SetOutput("SavedMean", {"bn_saved_mean"});
  bn_op->SetOutput("SavedVariance", {"bn_saved_var"});
  float eps = 1e-5;
  bn_op->SetAttr("epsilon", eps);
  bn_op->SetAttr("is_test", static_cast<int>(1));
  bn_op->SetAttr("use_global_stats", false);
  bn_op->SetAttr("momentum", 0.9f);
  bn_op->SetAttr("data_layout", std::string("NCHW"));

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
  const int num_nodes = graph->nodes().size();
  auto* fuser = new ConvBNFusePass;
  fuser->Apply(graph);
  ASSERT_EQ(graph->nodes().size(),
            num_nodes - 8UL /*nodes removed */ + 1UL /* eltwise_add node*/);
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(conv2d);
USE_LITE_OP(batch_norm);
USE_LITE_OP(elementwise_add);
