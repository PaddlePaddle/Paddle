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

namespace paddle {
namespace operators {

TEST(op_debug_str, test_unknown_dtype) {
  phi::Place place = phi::CPUPlace();
  phi::DDim dim{3, 4, 5, 6};
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

  auto x_tensor = scope.Var("X")->GetMutable<phi::DenseTensor>();
  x_tensor->Resize(dim);
  x_tensor->mutable_data<float>(place);

  auto y_tensor = scope.Var("Y")->GetMutable<phi::DenseTensor>();
  y_tensor->Resize(dim);
  y_tensor->mutable_data<float>(place);

  auto out_grad_tensor =
      scope.Var(framework::GradVarName("Out"))->GetMutable<phi::DenseTensor>();
  out_grad_tensor->Resize(dim);
  out_grad_tensor->mutable_data<float>(place);

  scope.Var(framework::GradVarName("X"))->GetMutable<phi::DenseTensor>();
  scope.Var(framework::GradVarName("Y"))->GetMutable<phi::DenseTensor>();

  auto op = framework::OpRegistry::CreateOp(desc);

  auto before_run_str = op->DebugStringEx(&scope);
  LOG(INFO) << before_run_str;
  ASSERT_TRUE(before_run_str.find(unknown_dtype) != std::string::npos);

  op->Run(scope, place);
  phi::DeviceContextPool::Instance().Get(place)->Wait();

  auto after_run_str = op->DebugStringEx(&scope);
  LOG(INFO) << after_run_str;
  ASSERT_TRUE(after_run_str.find(unknown_dtype) != std::string::npos);
}

}  // namespace operators
}  // namespace paddle
