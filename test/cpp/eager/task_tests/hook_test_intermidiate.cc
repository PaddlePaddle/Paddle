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
#include "paddle/fluid/eager/api/generated/fluid_generated/dygraph_forward_api.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/hooks.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "test/cpp/eager/test_utils.h"

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(matmul, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(matmul_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sigmoid, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sigmoid_grad, CPU, ALL_LAYOUT);

namespace egr {

paddle::Tensor hook_function(const paddle::Tensor& t) {
  auto t_dense = std::dynamic_pointer_cast<phi::DenseTensor>(t.impl());

  auto ret_meta = phi::DenseTensorMeta(
      t_dense->dtype(), t_dense->dims(), t_dense->layout());
  auto place = t_dense->place();
  size_t bytes_size =
      common::product(t_dense->dims()) * SizeOf(t_dense->dtype());
  auto ret_dense = std::make_shared<phi::DenseTensor>(
      paddle::memory::Alloc(place, bytes_size), std::move(ret_meta));

  float* t_ptr = t_dense->mutable_data<float>(place);
  float* ret_ptr = ret_dense->mutable_data<float>(place);
  for (int i = 0; i < ret_dense->numel(); i++) {
    ret_ptr[i] = t_ptr[i] + 3.0f;
  }

  auto ret_impl = std::dynamic_pointer_cast<phi::TensorBase>(ret_dense);
  paddle::Tensor ret = paddle::Tensor();
  ret.set_impl(ret_impl);

  return ret;
}

void test_sigmoid(bool is_remove_gradient_hook) {
  // Prepare Device Contexts
  VLOG(6) << "Init Env";
  eager_test::InitEnv(phi::CPUPlace());

  VLOG(6) << "Make Dim";
  phi::DDim ddim = common::make_ddim({2, 4, 4, 4});

  VLOG(6) << "Make paddle::Tensor";
  paddle::Tensor tensor =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::CPUPlace(),
                                        phi::DataType::FLOAT32,
                                        phi::DataLayout::NCHW,
                                        0.0,
                                        true);

  VLOG(6) << "Make ReduceHook function";
  auto reduce_hook = [&]() -> void {
    auto* t_ptr = std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl())
                      ->data<float>();
    for (int i = 0; i < tensor.numel(); i++) {
      t_ptr[i] = 100.0;  // set to 100.0
    }
  };

  VLOG(6) << "Retain Grad for Tensor";
  egr_utils_api::RetainGradForTensor(tensor);

  VLOG(6) << "Register GradientHook for Tensor";
  int64_t hook_id =
      egr_utils_api::RegisterGradientHookForTensor(tensor, hook_function);

  VLOG(6) << "Register ReduceHook for Tensor";
  egr_utils_api::RegisterReduceHookForTensor(tensor, reduce_hook);

  VLOG(6) << "Running Forward";
  auto output_tensor = sigmoid_dygraph_function(tensor, {});
  VLOG(6) << "Finish Forward";

  eager_test::CompareTensorWithValue<float>(output_tensor, 0.5);

  std::vector<paddle::Tensor> target_tensors = {output_tensor};

  if (is_remove_gradient_hook) {
    std::shared_ptr<GradNodeBase> grad_node_tmp = EagerUtils::grad_node(tensor);
    grad_node_tmp->RemoveGradientHook(hook_id);
  }

  VLOG(6) << "Running Backward";
  Backward(target_tensors, {});
  VLOG(6) << "Finish Backward";

  eager_test::CompareGradTensorWithValue<float>(
      tensor, is_remove_gradient_hook ? 0.25 : 0.25 + 3.0);

  VLOG(6) << "Checking ReduceHook results";
  for (int i = 0; i < tensor.numel(); i++) {
    PADDLE_ENFORCE_EQ(
        std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl())
            ->data<float>()[i],
        static_cast<float>(100.0f),
        common::errors::InvalidArgument(
            "Required tensor.impl()->data[%d] should be equal to 100.0 . ", i));
  }
  VLOG(6) << "After Tests";
}

void test_elementwiseAdd(bool is_remove_gradient_hook) {
  // Prepare Device Contexts
  eager_test::InitEnv(phi::CPUPlace());

  auto tracer = std::make_shared<paddle::imperative::Tracer>();
  paddle::imperative::SetCurrentTracer(tracer);

  // 1. Prepare Input
  phi::DDim ddimX = common::make_ddim({4, 16});
  paddle::Tensor X = eager_test::CreateTensorWithValue(ddimX,
                                                       phi::CPUPlace(),
                                                       phi::DataType::FLOAT32,
                                                       phi::DataLayout::NCHW,
                                                       3.0,
                                                       true);
  egr_utils_api::RetainGradForTensor(X);

  phi::DDim ddimY = common::make_ddim({4, 16});
  paddle::Tensor Y = eager_test::CreateTensorWithValue(ddimY,
                                                       phi::CPUPlace(),
                                                       phi::DataType::FLOAT32,
                                                       phi::DataLayout::NCHW,
                                                       2.0,
                                                       true);

  auto reduce_hook = [&]() -> void {
    auto* t_ptr =
        std::dynamic_pointer_cast<phi::DenseTensor>(Y.impl())->data<float>();
    for (int i = 0; i < Y.numel(); i++) {
      t_ptr[i] = 100.0;  // set to 100.0
    }
  };

  egr_utils_api::RetainGradForTensor(Y);
  int64_t hook_id =
      egr_utils_api::RegisterGradientHookForTensor(Y, hook_function);
  egr_utils_api::RegisterReduceHookForTensor(Y, reduce_hook);

  auto output_tensor = elementwise_add_dygraph_function(X, Y, {});

  eager_test::CompareTensorWithValue<float>(output_tensor, 5);
  std::vector<paddle::Tensor> target_tensors = {output_tensor};

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
    PADDLE_ENFORCE_EQ(
        std::dynamic_pointer_cast<phi::DenseTensor>(Y.impl())->data<float>()[i],
        static_cast<float>(100.0f),
        common::errors::InvalidArgument(
            "Required Y.impl()->data[%d] should be equal to 100.0 . ", i));
  }
}

void test_matmul(bool is_remove_gradient_hook) {
  // Prepare Device Contexts
  eager_test::InitEnv(phi::CPUPlace());

  auto tracer = std::make_shared<paddle::imperative::Tracer>();
  paddle::imperative::SetCurrentTracer(tracer);

  // 1. Prepare Input
  phi::DDim ddimX = common::make_ddim({4, 16});
  paddle::Tensor X = eager_test::CreateTensorWithValue(ddimX,
                                                       phi::CPUPlace(),
                                                       phi::DataType::FLOAT32,
                                                       phi::DataLayout::NCHW,
                                                       3.0,
                                                       true);
  egr_utils_api::RetainGradForTensor(X);

  phi::DDim ddimY = common::make_ddim({16, 20});
  paddle::Tensor Y = eager_test::CreateTensorWithValue(ddimY,
                                                       phi::CPUPlace(),
                                                       phi::DataType::FLOAT32,
                                                       phi::DataLayout::NCHW,
                                                       2.0,
                                                       true);

  auto reduce_hook = [&]() -> void {
    auto* t_ptr =
        std::dynamic_pointer_cast<phi::DenseTensor>(Y.impl())->data<float>();
    for (int i = 0; i < Y.numel(); i++) {
      t_ptr[i] = 100.0;  // set to 100.0
    }
  };

  egr_utils_api::RetainGradForTensor(Y);
  int64_t hook_id =
      egr_utils_api::RegisterGradientHookForTensor(Y, hook_function);
  egr_utils_api::RegisterReduceHookForTensor(Y, reduce_hook);

  auto output_tensor = matmul_v2_dygraph_function(
      X, Y, {{"trans_x", false}, {"trans_y", false}});

  eager_test::CompareTensorWithValue<float>(output_tensor, 96);
  std::vector<paddle::Tensor> target_tensors = {output_tensor};

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
    PADDLE_ENFORCE_EQ(
        std::dynamic_pointer_cast<phi::DenseTensor>(Y.impl())->data<float>()[i],
        static_cast<float>(100.0f),
        common::errors::InvalidArgument(
            "Required Y.impl()->data[%d] should be equal to 100.0 . ", i));
  }
}

void test_backward_final_hooks() {
  // Prepare Device Contexts
  VLOG(6) << "Init Env";
  eager_test::InitEnv(phi::CPUPlace());

  VLOG(6) << "Make paddle::Tensor";
  phi::DDim ddimX = common::make_ddim({4, 16});
  paddle::Tensor X = eager_test::CreateTensorWithValue(ddimX,
                                                       phi::CPUPlace(),
                                                       phi::DataType::FLOAT32,
                                                       phi::DataLayout::NCHW,
                                                       3.0,
                                                       true);
  phi::DDim ddimY = common::make_ddim({16, 20});
  egr_utils_api::RetainGradForTensor(X);

  paddle::Tensor Y = eager_test::CreateTensorWithValue(ddimY,
                                                       phi::CPUPlace(),
                                                       phi::DataType::FLOAT32,
                                                       phi::DataLayout::NCHW,
                                                       2.0,
                                                       true);

  VLOG(6) << "Make ReduceHook function";
  auto backward_final_hook = [&]() -> void {
    auto* t_ptr =
        std::dynamic_pointer_cast<phi::DenseTensor>(X.impl())->data<float>();
    VLOG(6) << "Run Target Backward Hook";
    for (int i = 0; i < X.numel(); i++) {
      t_ptr[i] = 100.0;  // set to 100.0
    }
  };
  VLOG(6) << "Register Backward Final Hook";
  egr_utils_api::RegisterBackwardFinalHook(backward_final_hook);

  VLOG(6) << "Running Forward";
  auto output_tensor = matmul_v2_dygraph_function(
      X, Y, {{"trans_x", false}, {"trans_y", false}});
  auto res = sigmoid_dygraph_function(output_tensor, {});
  VLOG(6) << "Finish Forward";

  eager_test::CompareTensorWithValue<float>(X, 3.0);

  std::vector<paddle::Tensor> target_tensors = {output_tensor};

  VLOG(6) << "Running Backward";
  Backward(target_tensors, {});
  VLOG(6) << "Finish Backward";
  eager_test::CompareTensorWithValue<float>(X, 100.0);
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

TEST(Hook_intermidiate, BackwardFinal) { test_backward_final_hooks(); }
}  // namespace egr
