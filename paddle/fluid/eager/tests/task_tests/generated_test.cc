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

#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/api/utils/tensor_utils.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/utils.h"

#include "paddle/fluid/eager/tests/test_utils.h"
#include "paddle/fluid/imperative/tracer.h"

#include "paddle/fluid/eager/api/generated/fluid_generated/dygraph_forward_api.h"
#include "paddle/phi/core/kernel_registry.h"

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(matmul, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(matmul_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sigmoid, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sigmoid_grad, CPU, ALL_LAYOUT);

namespace egr {

TEST(Generated, Sigmoid) {
  // Prepare Device Contexts
  eager_test::InitEnv(paddle::platform::CPUPlace());
  VLOG(6) << "Init Env";
  // 1. Prepare Input
  paddle::framework::DDim ddim = phi::make_ddim({2, 4, 4, 4});
  VLOG(6) << "Make Dim";
  paddle::experimental::Tensor tensor = egr_utils_api::CreateTensorWithValue(
      ddim, paddle::platform::CPUPlace(), phi::DataType::FLOAT32,
      phi::DataLayout::NCHW, 0.0, true);
  VLOG(6) << "Make paddle::experimental::Tensor";
  egr_utils_api::RetainGradForTensor(tensor);
  VLOG(6) << "Retain Grad for Tensor";
  auto output_tensor = sigmoid_dygraph_function(tensor, {});
  VLOG(6) << "Run Backward";
  eager_test::CompareTensorWithValue<float>(output_tensor, 0.5);

  std::vector<paddle::experimental::Tensor> target_tensors = {output_tensor};
  VLOG(6) << "Runing Backward";
  Backward(target_tensors, {});

  VLOG(6) << "Finish Backward";
  eager_test::CompareGradTensorWithValue<float>(tensor, 0.25);
}

TEST(Generated, Matmul_v2) {
  // Prepare Device Contexts
  eager_test::InitEnv(paddle::platform::CPUPlace());

  auto tracer = std::make_shared<paddle::imperative::Tracer>();
  paddle::imperative::SetCurrentTracer(tracer);

  // 1. Prepare Input
  paddle::framework::DDim ddimX = phi::make_ddim({4, 16});
  paddle::experimental::Tensor X = egr_utils_api::CreateTensorWithValue(
      ddimX, paddle::platform::CPUPlace(), phi::DataType::FLOAT32,
      phi::DataLayout::NCHW, 3.0, true);
  egr_utils_api::RetainGradForTensor(X);

  paddle::framework::DDim ddimY = phi::make_ddim({16, 20});
  paddle::experimental::Tensor Y = egr_utils_api::CreateTensorWithValue(
      ddimY, paddle::platform::CPUPlace(), phi::DataType::FLOAT32,
      phi::DataLayout::NCHW, 2.0, true);
  egr_utils_api::RetainGradForTensor(Y);

  auto output_tensor = matmul_v2_dygraph_function(
      X, Y, {{"trans_x", false}, {"trans_y", false}});

  eager_test::CompareTensorWithValue<float>(output_tensor, 96);

  std::vector<paddle::experimental::Tensor> target_tensors = {output_tensor};
  Backward(target_tensors, {});

  eager_test::CompareGradTensorWithValue<float>(X, 2.0 * 20);
  eager_test::CompareGradTensorWithValue<float>(Y, 3.0 * 4);
}

TEST(Generated, ElementwiseAdd) {
  // Prepare Device Contexts
  eager_test::InitEnv(paddle::platform::CPUPlace());

  auto tracer = std::make_shared<paddle::imperative::Tracer>();
  paddle::imperative::SetCurrentTracer(tracer);

  // 1. Prepare Input
  paddle::framework::DDim ddimX = phi::make_ddim({4, 16});
  paddle::experimental::Tensor X = egr_utils_api::CreateTensorWithValue(
      ddimX, paddle::platform::CPUPlace(), phi::DataType::FLOAT32,
      phi::DataLayout::NCHW, 3.0, true);
  egr_utils_api::RetainGradForTensor(X);

  paddle::framework::DDim ddimY = phi::make_ddim({4, 16});
  paddle::experimental::Tensor Y = egr_utils_api::CreateTensorWithValue(
      ddimY, paddle::platform::CPUPlace(), phi::DataType::FLOAT32,
      phi::DataLayout::NCHW, 2.0, true);
  egr_utils_api::RetainGradForTensor(Y);

  auto output_tensor = elementwise_add_dygraph_function(X, Y, {});

  eager_test::CompareTensorWithValue<float>(output_tensor, 5);

  std::vector<paddle::experimental::Tensor> target_tensors = {output_tensor};
  Backward(target_tensors, {});

  eager_test::CompareGradTensorWithValue<float>(X, 1.0);
  eager_test::CompareGradTensorWithValue<float>(Y, 1.0);
}

}  // namespace egr

USE_OP_ITSELF(sigmoid);
USE_OP_ITSELF(elementwise_add);
USE_OP_ITSELF(matmul_v2);
