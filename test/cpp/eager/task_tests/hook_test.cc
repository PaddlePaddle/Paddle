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

#include <sstream>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/api/generated/eager_generated/backwards/scale_node.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/hooks.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "test/cpp/eager/test_utils.h"

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);

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

TEST(RetainGrad, HookBeforeRetainGrad) {
  eager_test::InitEnv(phi::CPUPlace());

  // Prepare Inputs
  std::vector<paddle::Tensor> target_tensors;
  phi::DDim ddim = common::make_ddim({4, 16, 16, 32});

  // Create Target Tensor
  paddle::Tensor tensor =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::CPUPlace(),
                                        phi::DataType::FLOAT32,
                                        phi::DataLayout::NCHW,
                                        1.0 /*value*/,
                                        false /*is_leaf*/);
  target_tensors.emplace_back(std::move(tensor));
  paddle::Tensor& target_tensor = target_tensors[0];

  // Create ScaleNode
  auto scale_node_ptr = std::make_shared<GradNodeScale>(1, 1);
  scale_node_ptr->SetAttributes_scale(5.0 /*scale*/);

  // Set grad in/out meta for node0
  scale_node_ptr->SetDefaultGradInOutMeta();

  // Connect Input Tensor and ScaleNode via AutoGradMeta
  // Apply RetainGrad
  {
    // ScaleNode Hook: +3

    auto auto_grad_meta = std::make_shared<AutogradMeta>();
    auto_grad_meta->SetGradNode(
        std::dynamic_pointer_cast<GradNodeBase>(scale_node_ptr));
    auto_grad_meta->SetSingleOutRankWithSlot(0, 0);
    auto_grad_meta->SetStopGradient(false);
    target_tensor.set_autograd_meta(
        std::dynamic_pointer_cast<paddle::AbstractAutogradMeta>(
            auto_grad_meta));

    egr_utils_api::RegisterGradientHookForTensor(target_tensor, hook_function);
    egr_utils_api::RetainGradForTensor(
        target_tensor);  // result: 1.0 + 3.0 = 4.0
    egr_utils_api::RetainGradForTensor(
        target_tensor);  // result: 1.0 + 3.0 = 4.0
  }

  // Retain Grad for leaf tensor1
  paddle::Tensor leaf_tensor = paddle::Tensor();
  {
    // AccumulationNode Hook: +3
    auto tmp_tensor0 = paddle::Tensor();
    auto auto_grad_meta = EagerUtils::autograd_meta(&tmp_tensor0);

    auto acc_node_ptr = std::make_shared<GradNodeAccumulation>(auto_grad_meta);

    auto_grad_meta->SetStopGradient(false);
    auto_grad_meta->SetGradNode(acc_node_ptr);
    auto_grad_meta->SetSingleOutRankWithSlot(0, 0);
    std::vector<egr::AutogradMeta*> res = {auto_grad_meta};
    scale_node_ptr->SetGradOutMeta(tmp_tensor0, 0);

    leaf_tensor.set_autograd_meta(
        std::dynamic_pointer_cast<paddle::AbstractAutogradMeta>(
            tmp_tensor0.mutable_autograd_meta()));

    egr_utils_api::RegisterGradientHookForTensor(leaf_tensor, hook_function);
    egr_utils_api::RetainGradForTensor(
        leaf_tensor);  // result: 4.0*5.0 + 3.0 = 23.0
  }

  Backward(target_tensors, {});

  eager_test::CompareGradTensorWithValue<float>(target_tensor, 4.0);
  eager_test::CompareGradTensorWithValue<float>(leaf_tensor, 23.0);
}

TEST(RetainGrad, HookAfterRetainGrad) {
  eager_test::InitEnv(phi::CPUPlace());

  // Prepare Inputs
  std::vector<paddle::Tensor> target_tensors;
  phi::DDim ddim = common::make_ddim({4, 16, 16, 32});

  // Create Target Tensor
  paddle::Tensor tensor =
      eager_test::CreateTensorWithValue(ddim,
                                        phi::CPUPlace(),
                                        phi::DataType::FLOAT32,
                                        phi::DataLayout::NCHW,
                                        1.0 /*value*/,
                                        false /*is_leaf*/);
  target_tensors.emplace_back(std::move(tensor));
  paddle::Tensor& target_tensor = target_tensors[0];

  // Create ScaleNode
  auto scale_node_ptr = std::make_shared<GradNodeScale>(1, 1);
  scale_node_ptr->SetAttributes_scale(5.0 /*scale*/);
  // Set grad in/out meta for node0
  scale_node_ptr->SetDefaultGradInOutMeta();

  // Connect Input Tensor and ScaleNode via AutoGradMeta
  // Apply RetainGrad
  {
    // ScaleNode Hook: +3

    auto auto_grad_meta = std::make_shared<AutogradMeta>();
    auto_grad_meta->SetGradNode(
        std::dynamic_pointer_cast<GradNodeBase>(scale_node_ptr));
    auto_grad_meta->SetSingleOutRankWithSlot(0, 0);
    auto_grad_meta->SetStopGradient(false);
    target_tensor.set_autograd_meta(
        std::dynamic_pointer_cast<paddle::AbstractAutogradMeta>(
            auto_grad_meta));

    egr_utils_api::RetainGradForTensor(target_tensor);  // result: 1.0
    egr_utils_api::RegisterGradientHookForTensor(target_tensor, hook_function);
  }

  // Retain Grad for leaf tensor1
  paddle::Tensor leaf_tensor = paddle::Tensor();
  {
    // AccumulationNode Hook: +3
    auto tmp_tensor0 = paddle::Tensor();
    auto auto_grad_meta = EagerUtils::autograd_meta(&tmp_tensor0);
    auto acc_node_ptr = std::make_shared<GradNodeAccumulation>(auto_grad_meta);
    auto_grad_meta->SetGradNode(acc_node_ptr);
    auto_grad_meta->SetStopGradient(false);
    scale_node_ptr->SetGradOutMeta(tmp_tensor0, 0);

    auto_grad_meta->SetSingleOutRankWithSlot(0, 0);
    leaf_tensor.set_autograd_meta(
        std::dynamic_pointer_cast<paddle::AbstractAutogradMeta>(
            tmp_tensor0.mutable_autograd_meta()));

    egr_utils_api::RegisterGradientHookForTensor(leaf_tensor, hook_function);
  }

  Backward(target_tensors, {});
  eager_test::CompareGradTensorWithValue<float>(target_tensor, 1.0);
  eager_test::CompareGradTensorWithValue<float>(leaf_tensor, 23.0);
}
}  // namespace egr
