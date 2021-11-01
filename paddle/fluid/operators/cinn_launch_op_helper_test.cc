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

#include "paddle/fluid/operators/cinn_launch_op_helper.h"
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace details {

using LoDTensor = framework::LoDTensor;
using Scope = framework::Scope;

using CinnShape = cinn::hlir::framework::Shape;
using CinnTensor = cinn::hlir::framework::Tensor;
using CinnScope = cinn::hlir::framework::Scope;

TEST(CinnLaunchOpHelperTest, TestPlaceToCinnTarget) {
  ASSERT_EQ(PlaceToCinnTarget(platform::CPUPlace()),
            cinn::common::DefaultHostTarget());
  ASSERT_EQ(PlaceToCinnTarget(platform::CUDAPlace(0)),
            cinn::common::DefaultNVGPUTarget());
}

TEST(CinnLaunchOpHelperTest, TestGetConstTensors) {
  // build test data
  Scope scope;
  auto* var1 = scope.Var("lodtensor_var_1");
  var1->GetMutable<LoDTensor>();
  auto* var2 = scope.Var("lodtensor_var_2");
  var2->GetMutable<LoDTensor>();
  auto* var3 = scope.Var("selectedrows_var_1");
  var3->GetMutable<framework::SelectedRows>();
  // get expected result with legal input
  auto name2tensor =
      GetConstTensors(scope, {"lodtensor_var_1", "lodtensor_var_2"});
  ASSERT_EQ(name2tensor.size(), 2);
  EXPECT_EQ(name2tensor.at("lodtensor_var_1"), &var1->Get<LoDTensor>());
  EXPECT_EQ(name2tensor.at("lodtensor_var_2"), &var2->Get<LoDTensor>());
}

TEST(CinnLaunchOpHelperTest, TestGetCompiledTensors) {
  // build test data
  std::unordered_map<std::string, std::string> paddle2cinn_varmap(
      {{"pd_var1", "cinn_var1"},
       {"pd_var2", "cinn_var2"},
       {"pd_var3", "cinn_var3"}});
  CinnScope compiled_scope;
  compiled_scope.Var<CinnTensor>("cinn_var1");
  compiled_scope.Var<CinnTensor>("cinn_var2");
  // get expected result with legal input
  auto name2tensor = GetCompiledTensors({"pd_var1", "pd_var2"}, compiled_scope,
                                        paddle2cinn_varmap);
  ASSERT_EQ(name2tensor.size(), 2);
  EXPECT_EQ(name2tensor.at("pd_var1").get(),
            compiled_scope.GetTensor("cinn_var1").get());
  EXPECT_EQ(name2tensor.at("pd_var2").get(),
            compiled_scope.GetTensor("cinn_var2").get());
}

TEST(CinnLaunchOpHelperTest, TestCheckTensorEquivalent) {
  // build test data
  platform::CPUPlace place;
  Scope scope;
  CinnScope compiled_scope;
  auto* tensor1 = scope.Var("var1")->GetMutable<LoDTensor>();
  auto dims1 = std::vector<int>({2, 3});
  tensor1->mutable_data<float>(framework::make_ddim(dims1), place);
  auto* tensor2 = scope.Var("var2")->GetMutable<LoDTensor>();
  auto dims2 = std::vector<int>({5, 6, 7});
  tensor2->mutable_data<float>(framework::make_ddim(dims2), place);
  auto* tensor3 = scope.Var("var3")->GetMutable<LoDTensor>();
  tensor3->mutable_data<float>(framework::make_ddim({10, 20}), place);
  auto* tensor4 = scope.Var("var4")->GetMutable<LoDTensor>();
  tensor4->mutable_data<float>(framework::make_ddim({2, 4, 6}), place);
  compiled_scope.Var<CinnTensor>("var1");
  compiled_scope.Var<CinnTensor>("var2");
  compiled_scope.Var<CinnTensor>("var3");
  auto compiled_tensor1 = compiled_scope.GetTensor("var1");
  compiled_tensor1->Resize(CinnShape(dims1));
  auto compiled_tensor2 = compiled_scope.GetTensor("var2");
  compiled_tensor2->Resize(CinnShape(dims2));
  auto compiled_tensor3 = compiled_scope.GetTensor("var3");
  compiled_tensor3->Resize(CinnShape({10}));
  // expected equality
  CheckTensorEquivalent(
      {{"var1", tensor1}, {"var2", tensor2}},
      {{"var1", compiled_tensor1}, {"var2", compiled_tensor2}});
}

TEST(CinnLaunchOpHelperTest, TestInitializeOutputVar) {
  // build test data
  platform::CPUPlace place;
  Scope scope;
  scope.Var("var1");
  scope.Var("var2");
  CinnScope compiled_scope;
  compiled_scope.Var<CinnTensor>("var1");
  compiled_scope.Var<CinnTensor>("var2");
  compiled_scope.Var<CinnTensor>("var3");
  auto compiled_tensor1 = compiled_scope.GetTensor("var1");
  compiled_tensor1->Resize(CinnShape({2, 3}));
  auto compiled_tensor2 = compiled_scope.GetTensor("var2");
  compiled_tensor2->Resize(CinnShape({5, 6, 7}));
  auto compiled_tensor3 = compiled_scope.GetTensor("var3");
  compiled_tensor3->Resize(CinnShape({10}));
  // expected result
  InitializeOutputVar(scope, place,
                      {{"var1", compiled_tensor1}, {"var2", compiled_tensor2}});
  auto* var1 = scope.FindVar("var1");
  ASSERT_TRUE(var1->IsType<LoDTensor>());
  EXPECT_TRUE(var1->Get<LoDTensor>().IsInitialized());
  EXPECT_EQ(var1->Get<LoDTensor>().dims(), framework::make_ddim({2, 3}));
  auto* var2 = scope.FindVar("var2");
  ASSERT_TRUE(var2->IsType<LoDTensor>());
  EXPECT_TRUE(var2->Get<LoDTensor>().IsInitialized());
  EXPECT_EQ(var2->Get<LoDTensor>().dims(), framework::make_ddim({5, 6, 7}));
}

TEST(CinnLaunchOpHelperTest, TestSeperateTempVar) {
  CinnScope compiled_scope;
  compiled_scope.Var<CinnTensor>("cinn_temp_var1");
  compiled_scope.Var<CinnTensor>("cinn_input_var1");
  compiled_scope.Var<CinnTensor>("cinn_input_var2");
  compiled_scope.Var<CinnTensor>("cinn_temp_var2");
  compiled_scope.Var<CinnTensor>("cinn_output_var1");
  auto variable_names =
      SeperateTempVar(compiled_scope, {{"input_var1", "cinn_input_var1"},
                                       {"input_var2", "cinn_input_var2"},
                                       {"output_var1", "cinn_output_var1"}},
                      {"input_var1", "input_var2"}, {"output_var1"});
  ASSERT_EQ(variable_names.size(), 2);
}

TEST(CinnLaunchOpHelperTest, TestInitializeTempVar) {
  // build test data
  Scope temp_scope;
  platform::CPUPlace place;
  CinnScope compiled_scope;
  compiled_scope.Var<CinnTensor>("temp_var1");
  compiled_scope.Var<CinnTensor>("temp_var2");
  compiled_scope.Var<CinnTensor>("var3");
  auto compiled_tensor1 = compiled_scope.GetTensor("temp_var1");
  compiled_tensor1->Resize(CinnShape({2, 3}));
  auto compiled_tensor2 = compiled_scope.GetTensor("temp_var2");
  compiled_tensor2->Resize(CinnShape({5, 6, 7}));
  auto compiled_tensor3 = compiled_scope.GetTensor("var3");
  compiled_tensor3->Resize(CinnShape({10}));
  // expected result
  InitializeTempVar({"temp_var1", "temp_var2"}, compiled_scope, place,
                    &temp_scope);
  ASSERT_EQ(temp_scope.LocalVarNames().size(), 2);
  auto* temp_var1 = temp_scope.FindVar("temp_var1");
  ASSERT_NE(temp_var1, nullptr);
  EXPECT_TRUE(temp_var1->IsType<LoDTensor>());
  EXPECT_TRUE(temp_var1->Get<LoDTensor>().IsInitialized());
  EXPECT_EQ(temp_var1->Get<LoDTensor>().dims(), framework::make_ddim({2, 3}));
  auto* temp_var2 = temp_scope.FindVar("temp_var2");
  ASSERT_NE(temp_var2, nullptr);
  EXPECT_TRUE(temp_var2->IsType<LoDTensor>());
  EXPECT_TRUE(temp_var2->Get<LoDTensor>().IsInitialized());
  EXPECT_EQ(temp_var2->Get<LoDTensor>().dims(),
            framework::make_ddim({5, 6, 7}));
}

TEST(CinnLaunchOpHelperTest, TestSharePaddleTensorWithCinnBuffer) {
  // build test data
  Scope scope;
  platform::CPUPlace place;
  auto* var1 = scope.Var("var1");
  auto* tensor1 = var1->GetMutable<LoDTensor>();
  tensor1->mutable_data<float>(framework::make_ddim({5, 6}), place);
  auto* data1 = tensor1->data<float>();
  data1[0] = 9.99;
  data1[10] = 19.99;
  ASSERT_EQ(tensor1->numel(), 30);
  ASSERT_EQ(tensor1->dims().size(), 2);
  // excepted result
  cinn_buffer_t cinn_buffer;
  SharePaddleTensorWithCinnBuffer(tensor1, &cinn_buffer);
  ASSERT_NE(cinn_buffer.memory, nullptr);
  ASSERT_EQ(cinn_buffer.num_elements(), 30);
  auto* shadow_data = reinterpret_cast<float*>(cinn_buffer.memory);
  EXPECT_FLOAT_EQ(shadow_data[0], 9.99);
  EXPECT_FLOAT_EQ(shadow_data[10], 19.99);
}

TEST(CinnLaunchOpHelperTest, TestAppendExecutionArguments) {
  // build test data
  Scope scope;
  platform::CPUPlace place;
  auto* var1 = scope.Var("var1");
  auto* tensor1 = var1->GetMutable<LoDTensor>();
  tensor1->mutable_data<float>(framework::make_ddim({5, 6}), place);
  auto* var2 = scope.Var("temp_var2");
  auto* tensor2 = var2->GetMutable<LoDTensor>();
  tensor2->mutable_data<float>(framework::make_ddim({10}), place);
  // expected result
  std::map<std::string, cinn_pod_value_t> name2argument;
  std::vector<std::unique_ptr<cinn_buffer_t>> hold_buffers;
  AppendExecutionArguments(scope, {"var1", "temp_var2"},
                           {{"var1", "cinn_var1"}}, &name2argument,
                           &hold_buffers);
  ASSERT_EQ(name2argument.size(), 2);
  ASSERT_EQ(hold_buffers.size(), 2);
  EXPECT_NE(name2argument.count("cinn_var1"), 0);
  EXPECT_NE(name2argument.count("temp_var2"), 0);
  EXPECT_EQ(static_cast<cinn_buffer_t*>(name2argument.at("cinn_var1")),
            hold_buffers.front().get());
  EXPECT_EQ(static_cast<cinn_buffer_t*>(name2argument.at("temp_var2")),
            hold_buffers.back().get());
}

}  // namespace details
}  // namespace operators
}  // namespace paddle
