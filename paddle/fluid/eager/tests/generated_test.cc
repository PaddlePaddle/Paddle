// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

// Eager Dygraph

#include <chrono>

#include "gtest/gtest.h"

#include "paddle/fluid/eager/api/api.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/backward.h"

#include "paddle/fluid/eager/tests/test_utils.h"
#include "paddle/fluid/imperative/tracer.h"

#include "paddle/fluid/eager/generated/dygraph_forward_api.h"

#include "gperftools/profiler.h"

// TODO(jiabin): remove nolint here!!!
using namespace egr;  // NOLINT

TEST(Generated, Sigmoid) {
  // Prepare Device Contexts
  InitEnv(paddle::platform::CPUPlace());

  auto tracer = std::make_shared<paddle::imperative::Tracer>();
  paddle::imperative::SetCurrentTracer(tracer);

  // 1. Prepare Input
  paddle::framework::DDim ddim = paddle::framework::make_ddim({2, 4, 4, 4});
  egr::EagerTensor tensor = EagerUtils::CreateTensorWithValue(
      ddim, pten::Backend::CPU, pten::DataType::FLOAT32, pten::DataLayout::NCHW,
      0.0, true);
  RetainGradForTensor(tensor);

  auto output_tensor = sigmoid_dygraph_function(
      tensor, false /*use_mkldnn*/, false /*use_cudnn*/, 0 /*op_role*/,
      {} /*op_role_var*/, "" /*op_namescope*/, {} /*op_callstack*/,
      "" /*op_device*/, false /*with_quant_attr*/, true /*trace_backward*/);

  PADDLE_ENFORCE(
      CompareTensorWithValue<float>(output_tensor, 0.5) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 0.5));

  std::vector<egr::EagerTensor> target_tensors = {output_tensor};
  RunBackward(target_tensors, {});

  PADDLE_ENFORCE(
      CompareGradTensorWithValue<float>(tensor, 0.25) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 0.25));
}

TEST(Generated, Matmul_v2) {
  // Prepare Device Contexts
  InitEnv(paddle::platform::CPUPlace());

  auto tracer = std::make_shared<paddle::imperative::Tracer>();
  paddle::imperative::SetCurrentTracer(tracer);

  // 1. Prepare Input
  paddle::framework::DDim ddimX = paddle::framework::make_ddim({4, 16});
  egr::EagerTensor X = EagerUtils::CreateTensorWithValue(
      ddimX, pten::Backend::CPU, pten::DataType::FLOAT32,
      pten::DataLayout::NCHW, 3.0, true);
  RetainGradForTensor(X);

  paddle::framework::DDim ddimY = paddle::framework::make_ddim({16, 20});
  egr::EagerTensor Y = EagerUtils::CreateTensorWithValue(
      ddimY, pten::Backend::CPU, pten::DataType::FLOAT32,
      pten::DataLayout::NCHW, 2.0, true);
  RetainGradForTensor(Y);

  auto output_tensor = matmul_v2_dygraph_function(
      X, Y, false /*trans_x*/, false /*trans_y*/, {} /**fused_reshape_Out**/,
      {} /**fused_transpose_Out**/, false /*use_mkldnn*/,
      "float32" /*mkldnn_data_type*/, 0 /*op_role*/, {} /*op_role_var*/,
      "" /*op_namescope*/, {} /*op_callstack*/, "" /*op_device*/,
      false /*with_quant_attr*/, true /*trace_backward*/);

  PADDLE_ENFORCE(
      CompareTensorWithValue<float>(output_tensor, 96) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 96));

  std::vector<egr::EagerTensor> target_tensors = {output_tensor};

  RunBackward(target_tensors, {});

  PADDLE_ENFORCE(
      CompareGradTensorWithValue<float>(X, 2.0 * 20) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 40.0));

  PADDLE_ENFORCE(
      CompareGradTensorWithValue<float>(Y, 3.0 * 4) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 12.0));
}
