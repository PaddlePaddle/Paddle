// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/basic_engine.h"
#include "paddle/fluid/imperative/hooks.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/memory/memcpy.h"

PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(matmul_with_flatten, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(matmul_with_flatten_grad, CPU, ALL_LAYOUT);

COMMON_DECLARE_bool(sort_sum_gradient);

namespace paddle {
namespace imperative {

using vb_vector = std::vector<std::shared_ptr<imperative::VarBase>>;
using var_pair = std::pair<std::string, vb_vector>;

std::shared_ptr<imperative::VariableWrapper> DoubleHook(
    const std::shared_ptr<imperative::VariableWrapper>& var) {
  // 1. create out var
  auto out_var = std::make_shared<imperative::VariableWrapper>(var->Name());
  out_var->SetType(var->Type());
  out_var->SetDataType(var->DataType());
  out_var->SetForwardDataType(var->ForwardDataType());
  out_var->InnerSetOverriddenStopGradient(var->InnerOverriddenStopGradient());

  // 2. get input and output var's tensor
  auto* out_tensor = out_var->MutableVar()->GetMutable<phi::DenseTensor>();
  auto& tensor = var->Var().Get<phi::DenseTensor>();
  out_tensor->Resize(tensor.dims());

  // 3. double calc
  auto* data = tensor.data<float>();
  auto* out_data = out_tensor->mutable_data<float>(phi::CPUPlace());
  for (int64_t i = 0; i < out_tensor->numel(); ++i) {
    out_data[i] = data[i] * 2.0;  // NOLINT
  }

  return out_var;
}

TEST(TestHooks, TestGradVarLeafBackwardHook) {
  // 1. prepare
  Tracer tracer;
  std::shared_ptr<VarBase> x(new VarBase(true, "x"));
  std::shared_ptr<VarBase> y(new VarBase(true, "y"));
  std::shared_ptr<VarBase> out(new VarBase(true, "out"));
  x->SetOverriddenStopGradient(false);
  y->SetOverriddenStopGradient(false);

  phi::CPUPlace place;
  std::vector<float> src_data(10, 2.0);
  std::vector<int64_t> x_dims = {2, 5};
  std::vector<int64_t> y_dims = {5, 2};

  auto* x_tensor = x->MutableVar()->GetMutable<phi::DenseTensor>();
  auto* y_tensor = y->MutableVar()->GetMutable<phi::DenseTensor>();

  x_tensor->Resize(common::make_ddim(x_dims));
  auto* mutable_x = x_tensor->mutable_data<float>(place);
  memory::Copy(place,
               mutable_x,
               place,
               src_data.data(),
               sizeof(float) * src_data.size());

  y_tensor->Resize(common::make_ddim(y_dims));
  auto* mutable_y = y_tensor->mutable_data<float>(place);
  memory::Copy(place,
               mutable_y,
               place,
               src_data.data(),
               sizeof(float) * src_data.size());

  var_pair x_pair = var_pair("X", vb_vector(1, x));
  var_pair y_pair = var_pair("Y", vb_vector(1, y));
  var_pair out_pair = var_pair("Out", vb_vector(1, out));

  NameVarBaseMap ins = {x_pair, y_pair};
  NameVarBaseMap outs = {out_pair};
  framework::AttributeMap mul_attr_map;
  mul_attr_map["use_mkldnn"] = false;

  // add VariableWrapper hook
  x->GradVarBase()->AddVariableWrapperHook(
      std::make_shared<imperative::CppVariableWrapperHook>(DoubleHook));

  // add Void hook
  int64_t hook_value = 0;
  x->GradVarBase()->AddVoidHook(
      std::make_shared<std::function<void()>>([&]() { hook_value = 10; }));

  // 2. forward
  tracer.TraceOp<VarBase>("mul", ins, outs, mul_attr_map, place, true);

  ASSERT_EQ(x->GradVarBase()->GradOpNum(), 0UL);
  ASSERT_EQ(y->GradVarBase()->GradOpNum(), 0UL);
  ASSERT_EQ(out->GradVarBase()->GradOpNum(), 1UL);

  // 3. backward
  std::vector<std::shared_ptr<imperative::VarBase>> tensors{out};
  std::vector<std::shared_ptr<imperative::VarBase>> grad_tensors{nullptr};
  BasicEngine engine;
  engine.Init(tensors, grad_tensors);
  engine.Execute();

  // verify VariableWrapper hook result
  phi::DenseTensor x_grad;
  framework::TensorCopySync(
      x->GradVar().Get<phi::DenseTensor>(), place, &x_grad);
  for (int i = 0; i < x_grad.numel(); ++i) {
    ASSERT_EQ(x_grad.data<float>()[i], 8.0);
  }
  // verify Void hook result
  ASSERT_EQ(hook_value, 10);

  phi::DenseTensor y_grad;
  framework::TensorCopySync(
      y->GradVar().Get<phi::DenseTensor>(), place, &y_grad);

  for (int i = 0; i < y_grad.numel(); ++i) {
    ASSERT_EQ(y_grad.data<float>()[i], 4.0);
  }
}

void GradVarLeafBackwardHookWithGradAccumulatedTest() {
  // 1. prepare
  Tracer tracer;
  std::shared_ptr<VarBase> x(new VarBase(true, "x"));
  std::shared_ptr<VarBase> y(new VarBase(true, "y"));
  std::shared_ptr<VarBase> z(new VarBase(true, "z"));
  std::shared_ptr<VarBase> out_xy(new VarBase(true, "out_xy"));
  std::shared_ptr<VarBase> out_xz(new VarBase(true, "out_xz"));
  std::shared_ptr<VarBase> out(new VarBase(true, "out"));
  x->SetOverriddenStopGradient(false);
  y->SetOverriddenStopGradient(false);
  z->SetOverriddenStopGradient(false);

  phi::CPUPlace place;
  std::vector<float> src_data(10, 2.0);
  std::vector<int64_t> x_dims = {2, 5};
  std::vector<int64_t> y_dims = {5, 2};
  std::vector<int64_t> z_dims = {5, 2};

  auto* x_tensor = x->MutableVar()->GetMutable<phi::DenseTensor>();
  auto* y_tensor = y->MutableVar()->GetMutable<phi::DenseTensor>();
  auto* z_tensor = z->MutableVar()->GetMutable<phi::DenseTensor>();

  x_tensor->Resize(common::make_ddim(x_dims));
  auto* mutable_x = x_tensor->mutable_data<float>(place);
  memory::Copy(place,
               mutable_x,
               place,
               src_data.data(),
               sizeof(float) * src_data.size());

  y_tensor->Resize(common::make_ddim(y_dims));
  auto* mutable_y = y_tensor->mutable_data<float>(place);
  memory::Copy(place,
               mutable_y,
               place,
               src_data.data(),
               sizeof(float) * src_data.size());

  z_tensor->Resize(common::make_ddim(z_dims));
  auto* mutable_z = z_tensor->mutable_data<float>(place);
  memory::Copy(place,
               mutable_z,
               place,
               src_data.data(),
               sizeof(float) * src_data.size());

  // add VariableWrapper hook
  x->GradVarBase()->AddVariableWrapperHook(
      std::make_shared<imperative::CppVariableWrapperHook>(DoubleHook));

  // add Void hook
  int64_t hook_value = 0;
  x->GradVarBase()->AddVoidHook(
      std::make_shared<std::function<void()>>([&]() { hook_value = 100; }));

  // 2. forward
  var_pair x_pair = var_pair("X", vb_vector(1, x));
  var_pair y_pair = var_pair("Y", vb_vector(1, y));
  var_pair out_xy_pair = var_pair("Out", vb_vector(1, out_xy));
  NameVarBaseMap ins = {x_pair, y_pair};
  NameVarBaseMap outs = {out_xy_pair};
  framework::AttributeMap mul_attr_map;
  mul_attr_map["use_mkldnn"] = false;
  tracer.TraceOp<VarBase>("mul", ins, outs, mul_attr_map, place, true);

  var_pair z_pair = var_pair("Y", vb_vector(1, z));
  var_pair out_xz_pair = var_pair("Out", vb_vector(1, out_xz));
  ins = {x_pair, z_pair};
  outs = {out_xz_pair};
  tracer.TraceOp<VarBase>("mul", ins, outs, mul_attr_map, place, true);

  var_pair xy_pair = var_pair("X", vb_vector(1, out_xy));
  var_pair xz_pair = var_pair("Y", vb_vector(1, out_xz));
  var_pair out_pair = var_pair("Out", vb_vector(1, out));
  ins = {xy_pair, xz_pair};
  outs = {out_pair};
  framework::AttributeMap add_attr_map;
  tracer.TraceOp<VarBase>(
      "elementwise_add", ins, outs, add_attr_map, place, true);

  ASSERT_EQ(x->GradVarBase()->GradOpNum(), 0UL);
  ASSERT_EQ(y->GradVarBase()->GradOpNum(), 0UL);
  ASSERT_EQ(z->GradVarBase()->GradOpNum(), 0UL);
  ASSERT_EQ(out->GradVarBase()->GradOpNum(), 1UL);

  // 3. backward
  std::vector<std::shared_ptr<imperative::VarBase>> tensors{out};
  std::vector<std::shared_ptr<imperative::VarBase>> grad_tensors{nullptr};
  BasicEngine engine;
  engine.Init(tensors, grad_tensors);
  engine.Execute();

  // verify VariableWrapper hook result
  phi::DenseTensor x_grad;
  framework::TensorCopySync(
      x->GradVar().Get<phi::DenseTensor>(), place, &x_grad);
  for (int i = 0; i < x_grad.numel(); ++i) {
    ASSERT_EQ(x_grad.data<float>()[i], 16.0);
  }
  // verify Void hook result
  ASSERT_EQ(hook_value, 100);

  phi::DenseTensor y_grad;
  framework::TensorCopySync(
      y->GradVar().Get<phi::DenseTensor>(), place, &y_grad);

  for (int i = 0; i < y_grad.numel(); ++i) {
    ASSERT_EQ(y_grad.data<float>()[i], 4.0);
  }

  phi::DenseTensor z_grad;
  framework::TensorCopySync(
      z->GradVar().Get<phi::DenseTensor>(), place, &z_grad);

  for (int i = 0; i < z_grad.numel(); ++i) {
    ASSERT_EQ(z_grad.data<float>()[i], 4.0);
  }
}

TEST(TestHooks, TestGradVarLeafBackwardHookWithGradAccumulated) {
  GradVarLeafBackwardHookWithGradAccumulatedTest();
}

TEST(TestHooks, TestGradVarLeafBackwardHookWithSortedGradAccumulated) {
  FLAGS_sort_sum_gradient = true;
  GradVarLeafBackwardHookWithGradAccumulatedTest();
  FLAGS_sort_sum_gradient = false;
}

}  // namespace imperative
}  // namespace paddle

USE_OP_ITSELF(mul);
USE_OP_ITSELF(mul_grad);
USE_OP_ITSELF(elementwise_add);
USE_OP_ITSELF(elementwise_add_grad);
