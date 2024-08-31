// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <random>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"

template <typename DataType>
void AddVarToScope(const std::string var_name,
                   paddle::framework::Scope* scope,
                   const phi::DDim& dims) {
  std::random_device seed;
  std::default_random_engine engine(seed());
  std::uniform_real_distribution<float> dist(0, 100);

  phi::DenseTensor tmp_tensor;
  auto* tmp_data = tmp_tensor.mutable_data<DataType>(dims, phi::CPUPlace());
  auto* tensor = scope->Var(var_name)->GetMutable<phi::DenseTensor>();
  tensor->mutable_data<DataType>(dims, phi::CPUPlace());
  for (auto i = 0; i < tensor->numel(); ++i) {
    tmp_data[i] = static_cast<DataType>(dist(engine));
  }
  paddle::framework::TensorCopySync(tmp_tensor, phi::CPUPlace(), tensor);
}
TEST(test_conv2d_output, fp32) {
  paddle::framework::Scope scope;
  phi::CPUPlace cpu_place;

  paddle::framework::OpDesc conv2d_op(nullptr);
  conv2d_op.SetType("conv2d");
  conv2d_op.SetInput("Input", {"conv2d-X"});
  conv2d_op.SetInput("Filter", {"conv2d-Y"});
  conv2d_op.SetOutput("Output", {"conv2d-Out"});

  AddVarToScope<float>("conv2d-X", &scope, {1, 3, 224, 224});
  AddVarToScope<float>("conv2d-Y", &scope, {64, 3, 7, 7});
  AddVarToScope<float>("conv2d-Out", &scope, {1, 64, 218, 218});

  const std::vector<int> strides({1, 1});
  const std::vector<int> paddings({1, 1});
  const std::vector<int> dilations({1, 1});
  const int groups = 1;

  conv2d_op.SetAttr("strides", strides);
  conv2d_op.SetAttr("paddings", paddings);
  conv2d_op.SetAttr("dilations", dilations);
  conv2d_op.SetAttr("groups", groups);
  conv2d_op.SetAttr("use_mkldnn", true);

  auto op = paddle::framework::OpRegistry::CreateOp(conv2d_op);

  op->Run(scope, cpu_place);
}
TEST(test_conv2d_output, int8) {
  paddle::framework::Scope scope;
  phi::CPUPlace cpu_place;

  paddle::framework::OpDesc conv2d_op(nullptr);
  conv2d_op.SetType("conv2d");
  conv2d_op.SetInput("Input", {"conv2d-X"});
  conv2d_op.SetInput("Filter", {"conv2d-Y"});
  conv2d_op.SetOutput("Output", {"conv2d-Out"});

  AddVarToScope<int8_t>("conv2d-X", &scope, {1, 3, 224, 224});
  AddVarToScope<int8_t>("conv2d-Y", &scope, {64, 3, 7, 7});
  AddVarToScope<int8_t>("conv2d-Out", &scope, {1, 64, 218, 218});

  const std::vector<int> strides({1, 1});
  const std::vector<int> paddings({1, 1});
  const std::vector<int> dilations({1, 1});
  const int groups = 1;

  conv2d_op.SetAttr("strides", strides);
  conv2d_op.SetAttr("paddings", paddings);
  conv2d_op.SetAttr("dilations", dilations);
  conv2d_op.SetAttr("groups", groups);
  conv2d_op.SetAttr("use_mkldnn", true);
  conv2d_op.SetAttr("mkldnn_data_type", std::string("int8"));
  conv2d_op.SetAttr("force_fp32_output", false);

  auto op = paddle::framework::OpRegistry::CreateOp(conv2d_op);

  op->Run(scope, cpu_place);
}
TEST(test_conv2d_output, ic1) {
  paddle::framework::Scope scope;
  phi::CPUPlace cpu_place;

  paddle::framework::OpDesc conv2d_op(nullptr);
  conv2d_op.SetType("conv2d");
  conv2d_op.SetInput("Input", {"conv2d-X"});
  conv2d_op.SetInput("Filter", {"conv2d-Y"});
  conv2d_op.SetOutput("Output", {"conv2d-Out"});

  AddVarToScope<float>("conv2d-X", &scope, {1, 1, 224, 224});
  AddVarToScope<float>("conv2d-Y", &scope, {64, 1, 7, 7});
  AddVarToScope<float>("conv2d-Out", &scope, {1, 64, 218, 218});

  const std::vector<int> strides({1, 1});
  const std::vector<int> paddings({1, 1});
  const std::vector<int> dilations({1, 1});
  const int groups = 1;

  conv2d_op.SetAttr("strides", strides);
  conv2d_op.SetAttr("paddings", paddings);
  conv2d_op.SetAttr("dilations", dilations);
  conv2d_op.SetAttr("groups", groups);
  conv2d_op.SetAttr("use_mkldnn", true);

  auto op = paddle::framework::OpRegistry::CreateOp(conv2d_op);

  op->Run(scope, cpu_place);
}

TEST(test_conv2d_output, ic2) {
  paddle::framework::Scope scope;
  phi::CPUPlace cpu_place;

  paddle::framework::OpDesc conv2d_op(nullptr);
  conv2d_op.SetType("conv2d");
  conv2d_op.SetInput("Input", {"conv2d-X"});
  conv2d_op.SetInput("Filter", {"conv2d-Y"});
  conv2d_op.SetOutput("Output", {"conv2d-Out"});

  AddVarToScope<float>("conv2d-X", &scope, {1, 2, 224, 224});
  AddVarToScope<float>("conv2d-Y", &scope, {64, 2, 7, 7});
  AddVarToScope<float>("conv2d-Out", &scope, {1, 64, 218, 218});

  const std::vector<int> strides({1, 1});
  const std::vector<int> paddings({1, 1});
  const std::vector<int> dilations({1, 1});
  const int groups = 1;

  conv2d_op.SetAttr("strides", strides);
  conv2d_op.SetAttr("paddings", paddings);
  conv2d_op.SetAttr("dilations", dilations);
  conv2d_op.SetAttr("groups", groups);
  conv2d_op.SetAttr("use_mkldnn", true);

  auto op = paddle::framework::OpRegistry::CreateOp(conv2d_op);

  op->Run(scope, cpu_place);
}

TEST(test_conv2d_output, ic4) {
  paddle::framework::Scope scope;
  phi::CPUPlace cpu_place;

  paddle::framework::OpDesc conv2d_op(nullptr);
  conv2d_op.SetType("conv2d");
  conv2d_op.SetInput("Input", {"conv2d-X"});
  conv2d_op.SetInput("Filter", {"conv2d-Y"});
  conv2d_op.SetOutput("Output", {"conv2d-Out"});

  AddVarToScope<float>("conv2d-X", &scope, {1, 4, 224, 224});
  AddVarToScope<float>("conv2d-Y", &scope, {64, 4, 7, 7});
  AddVarToScope<float>("conv2d-Out", &scope, {1, 64, 218, 218});

  const std::vector<int> strides({1, 1});
  const std::vector<int> paddings({1, 1});
  const std::vector<int> dilations({1, 1});
  const int groups = 1;

  conv2d_op.SetAttr("strides", strides);
  conv2d_op.SetAttr("paddings", paddings);
  conv2d_op.SetAttr("dilations", dilations);
  conv2d_op.SetAttr("groups", groups);
  conv2d_op.SetAttr("use_mkldnn", true);

  auto op = paddle::framework::OpRegistry::CreateOp(conv2d_op);

  op->Run(scope, cpu_place);
}
