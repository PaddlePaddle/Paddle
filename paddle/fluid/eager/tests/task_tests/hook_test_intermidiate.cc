// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <sstream>

#include "gtest/gtest.h"

#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/tests/test_utils.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/phi/core/dense_tensor.h"

#include "paddle/fluid/eager/api/generated/fluid_generated/dygraph_forward_api.h"
#include "paddle/fluid/eager/hooks.h"
#include "paddle/phi/core/kernel_registry.h"

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(matmul, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(matmul_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sigmoid, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sigmoid_grad, CPU, ALL_LAYOUT);

namespace egr {

paddle::experimental::Tensor hook_function(
    const paddle::experimental::Tensor& t) {
  auto t_dense = std::dynamic_pointer_cast<phi::DenseTensor>(t.impl());

  auto ret_meta = phi::DenseTensorMeta(t_dense->dtype(), t_dense->dims(),
                                       t_dense->layout());
  auto place = t_dense->place();
  size_t bytes_size = phi::product(t_dense->dims()) * SizeOf(t_dense->dtype());
  auto ret_dense = std::make_shared<phi::DenseTensor>(
      phi::make_intrusive<paddle::experimental::SharedStorage>(
          paddle::memory::Alloc(place, bytes_size)),
      std::move(ret_meta));

  float* t_ptr = t_dense->mutable_data<float>(place);
  float* ret_ptr = ret_dense->mutable_data<float>(place);
  for (int i = 0; i < ret_dense->numel(); i++) {
    ret_ptr[i] = t_ptr[i] + 3.0;
  }

  auto ret_impl = std::dynamic_pointer_cast<phi::TensorBase>(ret_dense);
  paddle::experimental::Tensor ret = paddle::experimental::Tensor();
  ret.set_impl(ret_impl);

  return ret;
}

void test_sigmoid(bool is_remove_gradient_hook) {
  // Prepare Device Contexts
  VLOG(6) << "Init Env";
  eager_test::InitEnv(paddle::platform::CPUPlace());

  VLOG(6) << "Make Dim";
  paddle::framework::DDim ddim = phi::make_ddim({2, 4, 4, 4});

  VLOG(6) << "Make paddle::experimental::Tensor";
  paddle::experimental::Tensor tensor = egr_utils_api::CreateTensorWithValue(
      ddim, paddle::platform::CPUPlace(), phi::DataType::FLOAT32,
      phi::DataLayout::NCHW, 0.0, true);

  VLOG(6) << "Make ReduceHook function";
  auto reduce_hook = [&](void) -> void {
    auto* t_ptr = std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl())
                      ->data<float>();
    for (int i = 0; i < tensor.numel(); i++) {
      t_ptr[i] = 100.0;  // set to 100.0
    }
  };

  VLOG(6) << "Retain Grad for Tensor";
  egr_utils_api::RetainGradForTensor(tensor);

  VLOG(6) << "Register GradientHook for Tensor";
  int64_t hook_id = egr_utils_api::RegisterGradientHookForTensor(
      tensor, std::make_shared<CppTensorHook>(hook_function));

  VLOG(6) << "Register ReduceHook for Tensor";
  egr_utils_api::RegisterReduceHookForTensor(
      tensor, std::make_shared<CppTensorVoidHook>(reduce_hook));

  VLOG(6) << "Runing Forward";
  auto output_tensor = sigmoid_dygraph_function(tensor, {});
  VLOG(6) << "Finish Forward";

  eager_test::CompareTensorWithValue<float>(output_tensor, 0.5);

  std::vector<paddle::experimental::Tensor> target_tensors = {output_tensor};

  if (is_remove_gradient_hook) {
    std::shared_ptr<GradNodeBase> grad_node_tmp = EagerUtils::grad_node(tensor);
    grad_node_tmp->RemoveGradientHook(hook_id);
  }

  VLOG(6) << "Runing Backward";
  Backward(target_tensors, {});
  VLOG(6) << "Finish Backward";

  eager_test::CompareGradTensorWithValue<float>(
      tensor, is_remove_gradient_hook ? 0.25 : 0.25 + 3.0);

  VLOG(6) << "Checking ReduceHook results";
  for (int i = 0; i < tensor.numel(); i++) {
    CHECK_EQ(std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl())
                 ->data<float>()[i],
             static_cast<float>(100.0f));
  }
  VLOG(6) << "After Tests";
}

void test_elementwiseAdd(bool is_remove_gradient_hook) {
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

  auto reduce_hook = [&]() -> void {
    auto* t_ptr =
        std::dynamic_pointer_cast<phi::DenseTensor>(Y.impl())->data<float>();
    for (int i = 0; i < Y.numel(); i++) {
      t_ptr[i] = 100.0;  // set to 100.0
    }
  };

  egr_utils_api::RetainGradForTensor(Y);
  int64_t hook_id = egr_utils_api::RegisterGradientHookForTensor(
      Y, std::make_shared<CppTensorHook>(hook_function));
  egr_utils_api::RegisterReduceHookForTensor(
      Y, std::make_shared<CppTensorVoidHook>(reduce_hook));

  auto output_tensor = elementwise_add_dygraph_function(X, Y, {});

  eager_test::CompareTensorWithValue<float>(output_tensor, 5);
  std::vector<paddle::experimental::Tensor> target_tensors = {output_tensor};

  if (is_remove_gradient_hook) {
    std::shared_ptr<GradNodeBase> grad_node_tmp = EagerUtils::grad_node(Y);
    grad_node_tmp->RemoveGradientHook(hook_id);
  }

  Backward(target_tensors, {});

  eager_test::CompareGradTensorWithValue<float>(X, 1.0);
  eager_test::CompareGradTensorWithValue<float>(
      Y, is_remove_gradient_hook ? 1.0 : 1.0 + 3.0);

  // Checking ReduceHook results
  for (int i = 0; i < Y.numel(); i++) {
    CHECK_EQ(
        std::dynamic_pointer_cast<phi::DenseTensor>(Y.impl())->data<float>()[i],
        static_cast<float>(100.0f));
  }
}

void test_matmul(bool is_remove_gradient_hook) {
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

  auto reduce_hook = [&](void) -> void {
    auto* t_ptr =
        std::dynamic_pointer_cast<phi::DenseTensor>(Y.impl())->data<float>();
    for (int i = 0; i < Y.numel(); i++) {
      t_ptr[i] = 100.0;  // set to 100.0
    }
  };

  egr_utils_api::RetainGradForTensor(Y);
  int64_t hook_id = egr_utils_api::RegisterGradientHookForTensor(
      Y, std::make_shared<CppTensorHook>(hook_function));
  egr_utils_api::RegisterReduceHookForTensor(
      Y, std::make_shared<CppTensorVoidHook>(reduce_hook));

  auto output_tensor = matmul_v2_dygraph_function(
      X, Y, {{"trans_x", false}, {"trans_y", false}});

  eager_test::CompareTensorWithValue<float>(output_tensor, 96);
  std::vector<paddle::experimental::Tensor> target_tensors = {output_tensor};

  if (is_remove_gradient_hook) {
    std::shared_ptr<GradNodeBase> grad_node_tmp = EagerUtils::grad_node(Y);
    grad_node_tmp->RemoveGradientHook(hook_id);
  }

  Backward(target_tensors, {});

  eager_test::CompareGradTensorWithValue<float>(X, 2.0 * 20);
  eager_test::CompareGradTensorWithValue<float>(
      Y, is_remove_gradient_hook ? 3.0 * 4 : 3.0 * 4 + 3);

  // Checking ReduceHook results
  for (int i = 0; i < Y.numel(); i++) {
    CHECK_EQ(
        std::dynamic_pointer_cast<phi::DenseTensor>(Y.impl())->data<float>()[i],
        static_cast<float>(100.0f));
  }
}

TEST(Hook_intermidiate, Sigmoid) {
  // True or false represents whether to call RemoveGradientHook
  test_sigmoid(true);
  test_sigmoid(false);
}

TEST(Hook_intermidiate, ElementwiseAdd) {
  test_elementwiseAdd(true);
  test_elementwiseAdd(false);
}

TEST(Hook_intermidiate, Matmul_v2) {
  test_matmul(true);
  test_matmul(false);
}
}  // namespace egr

USE_OP_ITSELF(sigmoid);
USE_OP_ITSELF(elementwise_add);
USE_OP_ITSELF(matmul_v2);
