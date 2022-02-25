/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <memory>
#include <random>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/paddle2cinn/build_cinn_pass.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/core/ddim.h"

namespace paddle::operators {

using LoDTensor = framework::LoDTensor;
using Variable = framework::Variable;
using Graph = framework::ir::Graph;
using Node = framework::ir::Node;
using framework::paddle2cinn::Name2VarInfoMap;

std::unique_ptr<Graph> CreateOnlyElementwiseAddGraph(
    const std::string& x_name, const std::string& y_name,
    const std::string& out_name) {
  auto g = std::make_unique<Graph>(framework::ProgramDesc());
  framework::OpDesc feed_op_x, feed_op_y;
  feed_op_x.SetType("feed");
  feed_op_x.SetOutput("Out", {x_name});
  feed_op_y.SetType("feed");
  feed_op_y.SetOutput("Out", {y_name});

  framework::VarDesc x_var(x_name);
  framework::VarDesc y_var(y_name);
  framework::VarDesc out_var(out_name);

  framework::OpDesc elementwise_add_op;
  elementwise_add_op.SetType("add");
  elementwise_add_op.SetInput("X", {x_name});
  elementwise_add_op.SetInput("Y", {y_name});
  elementwise_add_op.SetOutput("Out", {out_name});

  auto* feed_op_node_x = g->CreateOpNode(&feed_op_x);
  auto* feed_op_node_y = g->CreateOpNode(&feed_op_y);
  auto* elementwise_add_node = g->CreateOpNode(&elementwise_add_op);
  auto* x_node = g->CreateVarNode(&x_var);
  auto* y_node = g->CreateVarNode(&y_var);
  auto* out_node = g->CreateVarNode(&out_var);

  // fill op node
  feed_op_node_x->outputs = {x_node};
  feed_op_node_y->outputs = {y_node};
  elementwise_add_node->inputs = {x_node, y_node};
  elementwise_add_node->outputs = {out_node};

  // fill variable node
  x_node->inputs = {feed_op_node_x};
  x_node->outputs = {elementwise_add_node};
  y_node->inputs = {feed_op_node_y};
  y_node->outputs = {elementwise_add_node};
  out_node->inputs = {elementwise_add_node};
  // set necessary attributes
  g->Set<std::vector<std::string>>(
      framework::paddle2cinn::kInputVars,
      new std::vector<std::string>({x_name, y_name}));
  g->Set<std::vector<std::string>>(framework::paddle2cinn::kInternalVars,
                                   new std::vector<std::string>({}));
  g->Set<std::vector<std::string>>(framework::paddle2cinn::kOutputVars,
                                   new std::vector<std::string>({out_name}));
  g->GetOrInit<Name2VarInfoMap>(
      framework::paddle2cinn::kMemOptVarInfoFromMainGraph);
  return g;
}

template <typename DataType>
void InitVariablesWithRandomValue(const std::vector<std::string>& var_names,
                                  const framework::DDim& common_ddim,
                                  const platform::Place& place,
                                  framework::Scope* scope) {
  std::random_device seed;
  std::default_random_engine engine(seed());
  std::uniform_real_distribution<float> dist(0, 100);

  LoDTensor tmp_tensor;
  auto* tmp_data =
      tmp_tensor.mutable_data<DataType>(common_ddim, platform::CPUPlace());
  for (const auto& var_name : var_names) {
    auto* tensor = scope->Var(var_name)->GetMutable<LoDTensor>();
    for (auto i = 0; i < tensor->numel(); ++i) {
      tmp_data[i] = static_cast<DataType>(dist(engine));
    }
    paddle::framework::TensorCopySync(tmp_tensor, place, tensor);
  }
}

template <typename DataType>
void CompareOpResult(Variable* test_out, Variable* expected_out) {
  LoDTensor test_tensor, expected_tensor;
  paddle::framework::TensorCopySync(test_out->Get<LoDTensor>(),
                                    platform::CPUPlace(), &test_tensor);
  paddle::framework::TensorCopySync(expected_out->Get<LoDTensor>(),
                                    platform::CPUPlace(), &expected_tensor);

  ASSERT_TRUE(test_tensor.IsInitialized());
  ASSERT_TRUE(expected_tensor.IsInitialized());
  ASSERT_EQ(test_tensor.dims(), expected_tensor.dims());
  const auto* test_data = test_tensor.data<DataType>();
  const auto* excepted_data = expected_tensor.data<DataType>();
  for (auto i = 0; i < expected_tensor.numel(); ++i) {
    EXPECT_EQ(test_data[i], excepted_data[i]);
  }
}

}  // namespace paddle::operators
