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
#include "paddle/fluid/operators/layer_norm_kernel.cu.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/pten/kernels/funcs/math_function.h"

namespace framework = paddle::framework;
namespace platform = paddle::platform;
namespace memory = paddle::memory;

USE_OP(dropout);
USE_OP(layer_norm);

template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;
template <typename T>
using LayerNormParamType = typename CudnnDataType<T>::BatchNormParamType;

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
    attrs.insert({"is_test", true});
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
void DropoutGrad(std::vector<T> *dx, const framework::DDim &x_dim,
                 const std::vector<T> &dout, const std::vector<uint8_t> &mask,
                 const platform::CUDADeviceContext &ctx, float dropout_prob,
                 bool is_upscale_in_train) {
  framework::Scope scope;
  const size_t n = x_dim[0] * x_dim[1];
  auto var_out = scope.Var("DOut");
  auto tensor_out = var_out->GetMutable<framework::LoDTensor>();
  framework::TensorFromVector(dout, ctx, tensor_out);
  tensor_out->Resize(x_dim);

  auto var_mask = scope.Var("Mask");
  auto tensor_mask = var_mask->GetMutable<framework::LoDTensor>();
  framework::TensorFromVector(mask, ctx, tensor_mask);
  tensor_mask->Resize(x_dim);

  auto var_dx = scope.Var("DX");
  auto tensor_dx = var_dx->GetMutable<framework::LoDTensor>();

  framework::AttributeMap attrs;
  attrs.insert({"dropout_prob", dropout_prob});
  attrs.insert({"is_test", false});
  if (is_upscale_in_train) {
    attrs.insert({"dropout_implementation", std::string("upscale_in_train")});
  } else {
    attrs.insert({"dropout_implementation", std::string("downgrade_in_infer")});
  }

  auto op = framework::OpRegistry::CreateOp(
      "dropout_grad", {{"Out@GRAD", {"DOut"}}, {"Mask", {"Mask"}}},
      {{"X@GRAD", {"DX"}}}, attrs);
  op->Run(scope, ctx.GetPlace());

  framework::TensorToVector(*tensor_dx, ctx, dx);
  ctx.Wait();
}

/**
 * @brief call paddle layer_norm op
 */
template <typename T>
void LayerNorm(const std::vector<LayerNormParamType<T>> &scale,
               const std::vector<LayerNormParamType<T>> &bias,
               const std::vector<T> &x,
               std::vector<LayerNormParamType<T>> *means,
               std::vector<LayerNormParamType<T>> *vars, std::vector<T> *y,
               const float epsilon, const int rows, const int cols,
               const platform::CUDADeviceContext &ctx) {
  framework::Scope scope;
  auto place = ctx.GetPlace();
  if (scale.size() > 0) {
    auto var_scale = scope.Var("Scale");
    auto tensor_scale = var_scale->GetMutable<framework::LoDTensor>();
    framework::TensorFromVector(scale, ctx, tensor_scale);
    tensor_scale->Resize({cols});
  }

  if (bias.size() > 0) {
    auto var_bias = scope.Var("Bias");
    auto tensor_bias = var_bias->GetMutable<framework::LoDTensor>();
    framework::TensorFromVector(bias, ctx, tensor_bias);
    tensor_bias->Resize({cols});
  }

  auto var_x = scope.Var("X");
  auto tensor_x = var_x->GetMutable<framework::LoDTensor>();
  framework::TensorFromVector(x, ctx, tensor_x);
  tensor_x->Resize({rows, cols});

  auto var_y = scope.Var("Y");
  auto tensor_y = var_y->GetMutable<framework::LoDTensor>();

  auto var_mean = scope.Var("Mean");
  auto tensor_mean = var_mean->GetMutable<framework::LoDTensor>();

  auto var_variance = scope.Var("Variance");
  auto tensor_variance = var_variance->GetMutable<framework::LoDTensor>();

  framework::AttributeMap attrs;
  attrs.insert({"epsilon", epsilon});

  auto op = framework::OpRegistry::CreateOp(
      "layer_norm", {{"X", {"X"}}, {"Scale", {"Scale"}}, {"Bias", {"Bias"}}},
      {{"Y", {"Y"}}, {"Mean", {"Mean"}}, {"Variance", {"Variance"}}}, attrs);
  op->Run(scope, place);
  framework::TensorToVector(*tensor_y, ctx, y);
  framework::TensorToVector(*tensor_mean, ctx, means);
  framework::TensorToVector(*tensor_variance, ctx, vars);
  ctx.Wait();
}

template <typename T>
inline void ReduceSum(const std::vector<T> &dout, std::vector<T> *dbias,
                      const int rows, const int cols) {
  for (int j = 0; j < cols; j++) {
    std::vector<T> tmp_dbias(rows);
    for (int i = 0; i < rows; i++) {
      tmp_dbias[i] = dout[i * cols + j];
    }
    int tmp_rows = rows / 2;
    while (tmp_rows) {
      for (int i = 0; i < tmp_rows; i++) {
        tmp_dbias[i] += tmp_dbias[i + tmp_rows];
      }
      tmp_rows /= 2;
    }
    (*dbias)[j] = tmp_dbias[0];
  }
}
