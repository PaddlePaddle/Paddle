/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/cinn/cinn_launch_context.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace operators {
namespace details {

using CinnShape = ::cinn::hlir::framework::Shape;

std::unique_ptr<CinnLaunchContext> CreateDefaultLaunchContext() {
  static std::once_flag initialized;
  static std::unordered_map<std::string, std::string> paddle2cinn_varmap;
  static std::shared_ptr<CinnScope> cinn_scope;
  std::call_once(initialized, [&paddle2cinn_varmap, &cinn_scope]() {
    auto& scope = cinn_scope;
    scope = std::make_shared<CinnScope>();

    scope->Var<CinnTensor>("cinn_var1");
    scope->GetTensor("cinn_var1")->Resize(CinnShape({3, 4}));
    scope->Var<CinnTensor>("cinn_var2");
    scope->GetTensor("cinn_var2")->Resize(CinnShape({6, 7, 8}));
    scope->Var<CinnTensor>("cinn_var3");
    scope->GetTensor("cinn_var3")->Resize(CinnShape({10, 16}));

    paddle2cinn_varmap = {
        {"var1", "cinn_var1"}, {"var3", "cinn_var3"}, {"var4", "cinn_var4"}};
  });

  return std::make_unique<CinnLaunchContext>(paddle2cinn_varmap, cinn_scope);
}

TEST(CinnLaunchContextTest, TestIsVariableUsed) {
  auto launch_context = CreateDefaultLaunchContext();

  ASSERT_EQ(launch_context->IsVariableUsed("var1"), true);
  ASSERT_EQ(launch_context->IsVariableUsed("var4"), false);
}

TEST(CinnLaunchContextTest, TestGetInternalVariableNames) {
  auto launch_context = CreateDefaultLaunchContext();
  auto internal_variable_names = launch_context->GetInternalVariableNames();
  ASSERT_EQ(internal_variable_names.size(), 3);
  EXPECT_NE(internal_variable_names.find("cinn_var2"),
            internal_variable_names.end());
}

TEST(CinnLaunchContextTest, TestCheckTensorEquivalent) {
  auto launch_context = CreateDefaultLaunchContext();
  platform::CPUPlace place;
  framework::Scope scope;
  auto* tensor1 = scope.Var("var1")->GetMutable<LoDTensor>();

  // CheckTensorEquivalent: tensor dimension not equivalent
  tensor1->mutable_data<float>(framework::make_ddim({3, 5}), place);
  ASSERT_THROW(launch_context->AssignExternalVariable("var1", place, tensor1),
               paddle::platform::EnforceNotMet);
}

TEST(CinnLaunchContextTest, TestAssignVariablePreCondition) {
  auto launch_context = CreateDefaultLaunchContext();
  platform::CPUPlace place;
  framework::Scope scope;
  auto* tensor4 = scope.Var("var4")->GetMutable<LoDTensor>();

  // not used
  ASSERT_THROW(launch_context->AssignExternalVariable("var4", place, tensor4),
               paddle::platform::EnforceNotMet);
  // not found
  ASSERT_THROW(
      launch_context->AssignExternalVariable("cinn_var4", place, tensor4),
      paddle::platform::EnforceNotMet);
}

TEST(CinnLaunchContextTest, TestSetArgument) {
  auto launch_context = CreateDefaultLaunchContext();

  platform::CPUPlace place;
  framework::Scope scope;
  auto* tensor1 = scope.Var("var1")->GetMutable<LoDTensor>();
  float* data1 =
      tensor1->mutable_data<float>(framework::make_ddim({3, 4}), place);
  data1[0] = 9.99f;
  data1[10] = 19.99f;

  // assign external variable
  ASSERT_NO_THROW(
      launch_context->AssignExternalVariable("var1", place, tensor1));
  auto* tensor2 = scope.Var("var2")->GetMutable<LoDTensor>();
  tensor2->mutable_data<float>(framework::make_ddim({6, 7, 8}), place);
  ASSERT_NO_THROW(
      launch_context->AssignInternalVariable("cinn_var2", place, tensor2));
  // FinalizeArguments not missed check
  ASSERT_THROW(launch_context->FinalizeArguments(),
               paddle::platform::EnforceNotMet);
  auto* tensor3 = scope.Var("var3")->GetMutable<LoDTensor>();
  tensor3->mutable_data<float>(framework::make_ddim({10, 16}), place);
  ASSERT_NO_THROW(
      launch_context->AssignExternalVariable("var3", place, tensor3));

  auto name2argument = launch_context->FinalizeArguments();
  ASSERT_EQ(name2argument.size(), 3);
  ASSERT_EQ(name2argument.count("cinn_var1"), 1);
  // check ShareTensorWithCinnBuffer
  auto* cinn_buffer =
      static_cast<cinn_buffer_t*>(name2argument.at("cinn_var1"));

  ASSERT_EQ(cinn_buffer->memory, nullptr);
  cinn_buffer->external_malloc->operator()(nullptr, cinn_buffer);
  ASSERT_NE(cinn_buffer->memory, nullptr);
  ASSERT_EQ(cinn_buffer->num_elements(), 12);
  auto* shadow_data = reinterpret_cast<float*>(cinn_buffer->memory);
  EXPECT_FLOAT_EQ(shadow_data[0], 9.99f);
  EXPECT_FLOAT_EQ(shadow_data[10], 19.99f);
}

}  // namespace details
}  // namespace operators
}  // namespace paddle
