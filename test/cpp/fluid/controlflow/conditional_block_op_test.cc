/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/controlflow/conditional_block_op.h"

#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/scope.h"

using DenseTensorArray = phi::TensorArray;
using Scope = paddle::framework::Scope;
using Variable = paddle::framework::Variable;
using Place = phi::Place;

TEST(ConditionalBlockGrad, NoNeedRunDenseTensorArray) {
  Place place = phi::CPUPlace();
  Scope scope;

  Variable* cond_var = scope.Var("condition");
  phi::DenseTensor* cond_tensor = cond_var->GetMutable<phi::DenseTensor>();
  phi::DDim cond_dims = common::make_ddim({1});
  bool* cond_data = cond_tensor->mutable_data<bool>(cond_dims, place);
  cond_data[0] = false;

  Variable* input_var = scope.Var("input_lod_tensor_array");
  DenseTensorArray* input_tensors = input_var->GetMutable<phi::TensorArray>();
  for (int i = 0; i < 5; ++i) {
    phi::DDim in_dims = common::make_ddim({i + 1, i + 2});
    phi::DenseTensor lod_tensor;
    float* in_data = lod_tensor.mutable_data<float>(in_dims, place);
    for (int j = 0; j < (i + 1) * (i + 2); ++j) {
      in_data[j] = static_cast<float>(j);
    }
    input_tensors->push_back(lod_tensor);
  }

  Variable* input_grad_var = scope.Var("input_lod_tensor_array@GRAD");
  DenseTensorArray* grad_tensors =
      input_grad_var->GetMutable<phi::TensorArray>();
  grad_tensors->resize(5);

  paddle::framework::AttributeMap attrs;
  attrs.insert({"is_scalar_condition", true});

  auto conditional_grad_op = paddle::framework::OpRegistry::CreateOp(
      "conditional_block_grad",
      {{"Input", {"input_lod_tensor_array"}}, {"Cond", {"condition"}}},
      {{"Input@GRAD", {"input_lod_tensor_array@GRAD"}}},
      attrs);

  conditional_grad_op->Run(scope, place);

  const DenseTensorArray& out_tensors = input_grad_var->Get<phi::TensorArray>();
  for (int i = 0; i < 5; ++i) {
    phi::DDim out_dims = out_tensors[i].dims();
    EXPECT_EQ(common::make_ddim({i + 1, i + 2}), out_dims);
    const float* out_data = out_tensors[i].data<float>();
    for (int j = 0; j < (i + 1) * (i + 2); ++j) {
      EXPECT_EQ(0, out_data[j]);
    }
  }
}
