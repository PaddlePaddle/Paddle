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

#include "paddle/fluid/operators/cinn_launch_op.h"
#include <stdlib.h>
#include <random>
#include <string>
#include "gtest/gtest.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/paddle2cinn/cinn_compiler.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/init.h"

USE_OP(cinn_launch);
USE_OP(elementwise_add);

namespace paddle {
namespace operators {

using framework::ir::Graph;
using framework::ir::Node;

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
  return g;
}

void CreateInputVariablesWithRandomData(
    const std::vector<std::string>& variable_names,
    const framework::DDim& common_ddim, framework::Scope* scope) {
  std::random_device seed;
  std::default_random_engine engine(seed());
  std::uniform_real_distribution<float> dist(0.f, 2.f);

  for (const auto& var_name : variable_names) {
    auto* tensor = scope->Var(var_name)->GetMutable<LoDTensor>();
    auto* data = tensor->mutable_data<float>(common_ddim, platform::CPUPlace());
    for (auto i = 0; i < tensor->numel(); ++i) {
      data[i] = dist(engine);
    }
  }
}

void CopyInputDataToPlace(const framework::Scope& scope,
                          const platform::Place& dst_place,
                          framework::Scope* dst_scope) {
  for (const auto& var_name : scope.LocalVarNames()) {
    const auto& src_tensor = scope.GetVar(var_name)->Get<LoDTensor>();
    auto* dst_tensor = dst_scope->Var(var_name)->GetMutable<LoDTensor>();
    TensorCopySync(src_tensor, dst_place, dst_tensor);
  }
}

TEST(CinnLaunchOpTest, TestElementwiseAddPass) {
  paddle::framework::InitDevices();
  platform::SetNumThreads(1);
  // cache test graph into CinnCompiler
  const auto& test_out_name = "test_out";
  const auto& expected_out_name = "expected_out";
  auto compilation_key = CinnCompiler::GetInstance()->AddGraph(
      CreateOnlyElementwiseAddGraph("test_x", "test_y", test_out_name));
  // create cinn_launch_op and elementwise_add op
  auto cinn_launch_op = paddle::framework::OpRegistry::CreateOp(
      "cinn_launch", {{"X", {"test_x", "test_y"}}}, {{"Out", {test_out_name}}},
      {{"compilation_key", compilation_key}});
  auto elementwise_add_op = paddle::framework::OpRegistry::CreateOp(
      "elementwise_add", {{"X", {"test_x"}}, {"Y", {"test_y"}}},
      {{"Out", {expected_out_name}}}, {{}});
  // prepare input data
  framework::Scope init_scope;
  CreateInputVariablesWithRandomData({"test_x", "test_y"}, {10, 20},
                                     &init_scope);
  // Run ops and check the computation results
  auto run_and_check_fn = [&](const platform::Place& place) {
    framework::Scope scope;
    CopyInputDataToPlace(init_scope, place, &scope);
    scope.Var(test_out_name)->GetMutable<LoDTensor>();
    scope.Var(expected_out_name)->GetMutable<LoDTensor>();

    cinn_launch_op->Run(scope, place);
    elementwise_add_op->Run(scope, place);

    LoDTensor test_out, expected_out;
    if (platform::is_cpu_place(place)) {
      test_out.ShareDataWith(scope.Var(test_out_name)->Get<LoDTensor>());
      expected_out.ShareDataWith(
          scope.Var(expected_out_name)->Get<LoDTensor>());
    } else {
      TensorCopySync(scope.Var(test_out_name)->Get<LoDTensor>(),
                     platform::CPUPlace(), &test_out);
      TensorCopySync(scope.Var(expected_out_name)->Get<LoDTensor>(),
                     platform::CPUPlace(), &expected_out);
    }

    ASSERT_TRUE(test_out.IsInitialized());
    ASSERT_TRUE(expected_out.IsInitialized());
    ASSERT_EQ(test_out.dims(), expected_out.dims());
    const auto* test_data = test_out.data<float>();
    const auto* excepted_data = expected_out.data<float>();
    for (auto i = 0; i < expected_out.numel(); ++i) {
      EXPECT_FLOAT_EQ(test_data[i], excepted_data[i]);
    }
  };

  LOG(INFO) << "Check compute result on cpu";
  run_and_check_fn(platform::CPUPlace());
  run_and_check_fn(platform::CPUPlace());

#ifdef PADDLE_WITH_CUDA
  // create an new elementwise_add op
  // because the above one cached the cpu kernel
  LOG(INFO) << "Check compute result on gpu";
  cinn_launch_op = paddle::framework::OpRegistry::CreateOp(
      "cinn_launch", {{"X", {"test_x", "test_y"}}}, {{"Out", {test_out_name}}},
      {{"compilation_key", compilation_key}});
  elementwise_add_op = paddle::framework::OpRegistry::CreateOp(
      "elementwise_add", {{"X", {"test_x"}}, {"Y", {"test_y"}}},
      {{"Out", {expected_out_name}}}, {{}});
  run_and_check_fn(platform::CUDAPlace());
  run_and_check_fn(platform::CUDAPlace());
#endif
}

namespace details {
// Testing helper function used on CinnLaunchOpKernel in the following:
// firstly build test data, then check both expected and illegal situations

using CinnShape = ::cinn::hlir::framework::Shape;

TEST(CinnLaunchOpHelperTest, TestPlaceToCinnTarget) {
  ASSERT_EQ(PlaceToCinnTarget(platform::CPUPlace()),
            ::cinn::common::DefaultHostTarget());
  ASSERT_EQ(PlaceToCinnTarget(platform::CUDAPlace(0)),
            ::cinn::common::DefaultNVGPUTarget());
  ASSERT_THROW(PlaceToCinnTarget(platform::XPUPlace()),
               paddle::platform::EnforceNotMet);
}

TEST(CinnLaunchOpHelperTest, TestMapPaddleVariablesToCinn) {
  std::unordered_map<std::string, std::string> varmap(
      {{"var1", "cinn_var1"}, {"var2", "cinn_var2"}, {"var3", "cinn_var3"}});

  auto cinn_names = MapPaddleVariablesToCinn({"var1", "var3"}, varmap);
  ASSERT_EQ(cinn_names.size(), 2);
  EXPECT_EQ(cinn_names, std::vector<std::string>({"cinn_var1", "cinn_var3"}));
  ASSERT_THROW(MapPaddleVariablesToCinn({"var1", "not_exist"}, varmap),
               paddle::platform::EnforceNotMet);
}

TEST(CinnLaunchOpHelperTest, TestGetCinnTensorsFromCompiledScope) {
  CinnScope cinn_scope;
  cinn_scope.Var<CinnTensor>("cinn_var1");
  cinn_scope.Var<CinnTensor>("cinn_var2");
  cinn_scope.Var<CinnTensor>("cinn_var3");

  auto cinn_tensors =
      GetCinnTensorsFromCompiledScope({"cinn_var1", "cinn_var3"}, cinn_scope);
  ASSERT_EQ(cinn_tensors.size(), 2);
  ASSERT_EQ(cinn_tensors.front().get(),
            cinn_scope.GetTensor("cinn_var1").get());
  ASSERT_EQ(cinn_tensors.back().get(), cinn_scope.GetTensor("cinn_var3").get());
  ASSERT_THROW(
      GetCinnTensorsFromCompiledScope({"cinn_var1", "not_exist"}, cinn_scope),
      paddle::platform::EnforceNotMet);
}

TEST(CinnLaunchOpHelperTest, TestCheckTensorEquivalent) {
  platform::CPUPlace place;
  framework::Scope scope;
  auto* tensor1 = scope.Var("var1")->GetMutable<LoDTensor>();
  tensor1->mutable_data<float>(framework::make_ddim({5, 8}), place);

  CinnScope cinn_scope;
  cinn_scope.Var<CinnTensor>("cinn_var1");
  auto cinn_tensor1 = cinn_scope.GetTensor("cinn_var1");
  cinn_tensor1->Resize(CinnShape({5, 8}));
  cinn_tensor1->set_type(::cinn::common::type_of<float>());

  ASSERT_NO_THROW(CheckTensorEquivalent("var1", tensor1, cinn_tensor1));
  auto tensor2 = scope.Var("var2")->GetMutable<LoDTensor>();
  ASSERT_THROW(CheckTensorEquivalent("var2", tensor2, cinn_tensor1),
               paddle::platform::EnforceNotMet);

  cinn_tensor1->Resize(CinnShape({5, 7}));
  ASSERT_THROW(CheckTensorEquivalent("var1", tensor1, cinn_tensor1),
               paddle::platform::EnforceNotMet);
}

TEST(CinnLaunchOpHelperTest, TestTensorMutableDataWithCinnInfo) {
  platform::CPUPlace place;
  framework::Scope scope;
  auto* tensor1 = scope.Var("var1")->GetMutable<LoDTensor>();
  CinnScope cinn_scope;
  cinn_scope.Var<CinnTensor>("cinn_var1");
  auto cinn_tensor1 = cinn_scope.GetTensor("cinn_var1");
  cinn_tensor1->Resize(CinnShape({5, 8}));

  ASSERT_NO_THROW(TensorMutableDataWithCinnInfo(place, cinn_tensor1, tensor1));
  ASSERT_TRUE(tensor1->IsInitialized());
  ASSERT_EQ(tensor1->dims(), framework::make_ddim({5, 8}));
}

TEST(CinnLaunchOpHelperTest, TestSeperateTempVar) {
  CinnScope cinn_scope;
  cinn_scope.Var<CinnTensor>("cinn_var1");
  cinn_scope.Var<CinnTensor>("cinn_var2");
  cinn_scope.Var<CinnTensor>("cinn_var3");
  cinn_scope.Var<CinnTensor>("cinn_var4");

  auto temp_names =
      SeperateTempVar(cinn_scope, {"cinn_var1", "cinn_var2"}, {"cinn_var4"});
  ASSERT_EQ(temp_names.size(), 1);
  EXPECT_EQ(temp_names.front(), "cinn_var3");
}

TEST(CinnLaunchOpHelperTest, TestShareTensorWithCinnBuffer) {
  platform::CPUPlace place;
  framework::Scope scope;
  auto* tensor1 = scope.Var("var1")->GetMutable<LoDTensor>();
  tensor1->mutable_data<float>(framework::make_ddim({5, 6}), place);
  auto* data1 = tensor1->data<float>();
  data1[0] = 9.99f;
  data1[10] = 19.99f;

  auto cinn_buffer = ShareTensorWithCinnBuffer(tensor1);
  ASSERT_NE(cinn_buffer->memory, nullptr);
  ASSERT_EQ(cinn_buffer->num_elements(), 30);
  auto* shadow_data = reinterpret_cast<float*>(cinn_buffer->memory);
  EXPECT_FLOAT_EQ(shadow_data[0], 9.99f);
  EXPECT_FLOAT_EQ(shadow_data[10], 19.99f);
}

TEST(CinnLaunchOpHelperTest, TestCheckArgumentsNotMissed) {
  CinnScope cinn_scope;
  cinn_scope.Var<CinnTensor>("cinn_var1");
  cinn_scope.Var<CinnTensor>("cinn_var2");
  std::map<std::string, cinn_pod_value_t> name2argument(
      {{"cinn_var1", cinn_pod_value_t()}, {"cinn_var2", cinn_pod_value_t()}});

  ASSERT_NO_THROW(CheckArgumentsNotMissed(cinn_scope, name2argument));
  name2argument.erase("cinn_var2");
  ASSERT_THROW(CheckArgumentsNotMissed(cinn_scope, name2argument),
               paddle::platform::EnforceNotMet);
}

}  // namespace details

}  // namespace operators
}  // namespace paddle
