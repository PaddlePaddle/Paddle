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

#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/kernel_registry.h"

USE_OP_ITSELF(elementwise_add_grad);
PD_DECLARE_KERNEL(add_grad, CPU, ALL_LAYOUT);

namespace paddle {
namespace operators {

TEST(op_debug_str, test_unknown_dtype) {
  platform::Place place = platform::CPUPlace();
  framework::DDim dim{3, 4, 5, 6};
  const std::string unknown_dtype = "unknown_dtype";

  framework::OpDesc desc;
  framework::Scope scope;

  desc.SetType("elementwise_add_grad");
  desc.SetInput("X", {"X"});
  desc.SetInput("Y", {"Y"});
  desc.SetInput(framework::GradVarName("Out"), {framework::GradVarName("Out")});
  desc.SetOutput(framework::GradVarName("X"), {framework::GradVarName("X")});
  desc.SetOutput(framework::GradVarName("Y"), {framework::GradVarName("Y")});
  desc.SetAttr("axis", -1);
  desc.SetAttr("use_mkldnn", false);
  desc.SetAttr("x_data_format", "");
  desc.SetAttr("y_data_format", "");

  auto x_tensor = scope.Var("X")->GetMutable<framework::LoDTensor>();
  x_tensor->Resize(dim);
  x_tensor->mutable_data<float>(place);

  auto y_tensor = scope.Var("Y")->GetMutable<framework::LoDTensor>();
  y_tensor->Resize(dim);
  y_tensor->mutable_data<float>(place);

  auto out_grad_tensor = scope.Var(framework::GradVarName("Out"))
                             ->GetMutable<framework::LoDTensor>();
  out_grad_tensor->Resize(dim);
  out_grad_tensor->mutable_data<float>(place);

  scope.Var(framework::GradVarName("X"))->GetMutable<framework::LoDTensor>();
  scope.Var(framework::GradVarName("Y"))->GetMutable<framework::LoDTensor>();

  auto op = framework::OpRegistry::CreateOp(desc);

  auto before_run_str = op->DebugStringEx(&scope);
  LOG(INFO) << before_run_str;
  ASSERT_TRUE(before_run_str.find(unknown_dtype) != std::string::npos);

  op->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();

  auto after_run_str = op->DebugStringEx(&scope);
  LOG(INFO) << after_run_str;
  ASSERT_TRUE(after_run_str.find(unknown_dtype) != std::string::npos);
}

}  // namespace operators
}  // namespace paddle
