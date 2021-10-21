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

using CinnShape = cinn::hlir::framework::Shape;

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

  // cast throw
  ASSERT_THROW(GetConstTensors(scope, {"lodtensor_var_1", "var_not_exist"}),
               paddle::platform::EnforceNotMet);
  ASSERT_THROW(
      GetConstTensors(scope, {"lodtensor_var_1", "selectedrows_var_1"}),
      paddle::platform::EnforceNotMet);
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
  // cast throw
  ASSERT_THROW(GetCompiledTensors({"pd_var1", "var_not_exist"}, compiled_scope,
                                  paddle2cinn_varmap),
               paddle::platform::EnforceNotMet);
  ASSERT_THROW(GetCompiledTensors({"pd_var1", "pd_var3"}, compiled_scope,
                                  paddle2cinn_varmap),
               paddle::platform::EnforceNotMet);
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
  auto compiled_tensor3 = compiled_scope.GetTensor("var4");
  compiled_tensor3->Resize(CinnShape({10}));

  // expected equality
  CheckTensorEquivalent(
      {{"var1", tensor1}, {"var2", tensor2}},
      {{"var1", compiled_tensor1}, {"var2", compiled_tensor2}});

  // cast throw
  ASSERT_THROW(CheckTensorEquivalent(
                   {{"var1", tensor1}, {"var4", tensor2}},
                   {{"var1", compiled_tensor1}, {"var2", compiled_tensor2}}),
               paddle::platform::EnforceNotMet);
  ASSERT_THROW(CheckTensorEquivalent(
                   {{"var1", tensor1}, {"var3", tensor2}},
                   {{"var1", compiled_tensor1}, {"var3", compiled_tensor3}}),
               paddle::platform::EnforceNotMet);
}

TEST(CinnLaunchOpHelperTest, TestInitializeOutputVar) {}

}  // namespace details
}  // namespace operators
}  // namespace paddle
