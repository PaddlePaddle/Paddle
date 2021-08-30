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

#pragma once

#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/string/printf.h"

namespace framework = paddle::framework;
namespace platform = paddle::platform;
namespace memory = paddle::memory;

USE_OP(dropout);

/**
 * @brief call paddle dropout op
 */
template <typename T>
void Dropout(const std::vector<T> &x, const framework::DDim &x_dim,
             std::vector<T> *out, std::vector<uint8_t> *mask,
             const platform::CUDADeviceContext &ctx, uint64_t seed,
             float dropout_prob, bool is_upscale_in_train, bool is_test) {
  framework::Scope scope;
  auto var_x = scope.Var("X");
  auto tensor_x = var_x->GetMutable<framework::LoDTensor>();
  framework::TensorFromVector(x, ctx, tensor_x);
  tensor_x->Resize(x_dim);

  auto var_out = scope.Var("Out");
  auto tensor_out = var_out->GetMutable<framework::LoDTensor>();

  auto var_mask = scope.Var("Mask");
  auto tensor_mask = var_mask->GetMutable<framework::LoDTensor>();

  framework::AttributeMap attrs;
  attrs.insert({"fix_seed", 1});
  attrs.insert({"seed", static_cast<int>(seed)});
  attrs.insert({"dropout_prob", dropout_prob});
  if (is_upscale_in_train) {
    attrs.insert({"dropout_implementation", std::string("upscale_in_train")});
  }

  if (is_test) {
    attrs.insert({"is_test", 1});
  }

  auto op = framework::OpRegistry::CreateOp(
      "dropout", {{"X", {"X"}}}, {{"Out", {"Out"}}, {"Mask", {"Mask"}}}, attrs);
  op->Run(scope, ctx.GetPlace());

  framework::TensorToVector<T>(*tensor_out, ctx, out);
  if (!is_test) {
    framework::TensorToVector<uint8_t>(*tensor_mask, ctx, mask);
  }
  ctx.Wait();
}

/**
 * @brief call paddle dropout_grad op
 */
template <typename T>
void DropoutGrad(T *dx, const framework::DDim &x_dim, const T *dout,
                 const uint8_t *mask, const platform::CUDADeviceContext &ctx,
                 float dropout_prob, bool is_upscale_in_train) {
  framework::Scope scope;
  const size_t n = x_dim[0] * x_dim[1];
  auto var_out = scope.Var("DOut");
  auto tensor_out = var_out->GetMutable<framework::LoDTensor>();
  tensor_out->Resize(x_dim);
  tensor_out->mutable_data<T>(ctx.GetPlace());
  cudaMemcpy(tensor_out->data<T>(), dout, n * sizeof(T),
             cudaMemcpyHostToDevice);

  auto var_mask = scope.Var("Mask");
  auto tensor_mask = var_mask->GetMutable<framework::LoDTensor>();
  tensor_mask->Resize(x_dim);
  tensor_mask->mutable_data<uint8_t>(ctx.GetPlace());
  cudaMemcpy(tensor_mask->data<uint8_t>(), mask, n * sizeof(uint8_t),
             cudaMemcpyHostToDevice);

  auto var_dx = scope.Var("DX");
  auto tensor_dx = var_dx->GetMutable<framework::LoDTensor>();

  framework::AttributeMap attrs;
  attrs.insert({"dropout_prob", dropout_prob});
  attrs.insert({"is_test", 0});
  if (is_upscale_in_train) {
    attrs.insert({"dropout_implementation", std::string("upscale_in_train")});
  } else {
    attrs.insert({"dropout_implementation", std::string("downgrade_in_infer")});
  }

  auto op = framework::OpRegistry::CreateOp(
      "dropout_grad", {{"Out@GRAD", {"DOut"}}, {"Mask", {"Mask"}}},
      {{"X@GRAD", {"DX"}}}, attrs);
  op->Run(scope, ctx.GetPlace());

  cudaMemcpy(dx, tensor_dx->data<T>(), x_dim[0] * x_dim[1] * sizeof(T),
             cudaMemcpyDeviceToHost);
  ctx.Wait();
}
