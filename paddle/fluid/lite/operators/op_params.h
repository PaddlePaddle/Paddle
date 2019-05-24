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

#pragma once
#include <vector>
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/utils/all.h"

/*
 * This file contains all the argument parameter data structure for operators.
 */

namespace paddle {
namespace lite {
namespace operators {

using param_t = Any;

/// ----------------------- Functional operators ------------------------------
struct FeedParam {
  const std::vector<lite::Tensor>* feed_list{};
  lite::Tensor* out{};
  int col;
};

struct FetchParam {
  const lite::Tensor* input{};
  std::vector<lite::Tensor>* fetch_list{};
  int col;
};

// Helper op for lite framework
struct IoCopyParam {
  const lite::Tensor* x{};
  lite::Tensor* y{};
};

/// -------------------------- NN operators ------------------------------------

struct FcParam {
  lite::Tensor* input{};
  lite::Tensor* w{};
  lite::Tensor* bias{};
  lite::Tensor* output{};
  lite::DDim in_mat_dims;
  int in_num_col_dims{1};
};

struct ReluParam {
  lite::Tensor* input{};
  lite::Tensor* output{};
};

// For Mul Op
struct MulParam {
  lite::Tensor* x{};
  lite::Tensor* y{};
  lite::Tensor* output{};

  int x_num_col_dims{1};
  int y_num_col_dims{1};
};

struct MulGradParam {
  const lite::Tensor* x{};
  const lite::Tensor* y{};
  const lite::Tensor* output_grad{};
  lite::Tensor* x_grad{};
  lite::Tensor* y_grad{};

  int x_num_col_dims{1};
  int y_num_col_dims{1};
};

// For Scale Op
struct ScaleParam {
  lite::Tensor* x{};
  lite::Tensor* output{};

  float scale{1.};
  float bias{};
  bool bias_after_scale{true};
};

/// ----------------------- element wise operators ----------------------
struct ElementwiseParam {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  lite::Tensor* Out{};
  int axis{-1};  // for broadcasting.
};

struct ElementwiseGradParam {
  const lite::Tensor* Y{};
  const lite::Tensor* Out_grad{};
  lite::Tensor* X_grad{};
  lite::Tensor* Y_grad{};
  int axis{-1};  // for broadcasting.
};

/// ----------------------- activation operators ----------------------
struct ActivationParam {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
};

struct ActivationGradParam {
  const lite::Tensor* X{};
  const lite::Tensor* Out{};
  // for backward
  lite::Tensor* X_grad{};
  const lite::Tensor* Out_grad{};
};

/// ----------------------- mean operators ----------------------
struct MeanParam {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
};

struct MeanGradParam {
  const lite::Tensor* X{};
  const lite::Tensor* Out_grad{};
  // for backward
  lite::Tensor* X_grad{};
};

/// ----------------------- fill_constant operators ----------------------
struct FillConstantParam {
  int dtype{framework::proto::VarType::FP32};
  std::vector<int64_t> shape{};
  float value{0.0f};
  // useless for x86, keep it for compatibility
  bool force_cpu{false};
  lite::Tensor* Out{};
};

/// ----------------------- sgd operators ----------------------
struct SGDParam {
  int dtype{framework::proto::VarType::FP32};

  const lite::Tensor* Param{};
  const lite::Tensor* LearningRate{};
  const lite::Tensor* Grad{};
  lite::Tensor* ParamOut{};
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
