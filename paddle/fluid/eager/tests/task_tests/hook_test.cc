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
#include "paddle/fluid/eager/api/generated/eager_generated/backwards/scale_node.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/grad_node_info.h"

#include "paddle/fluid/eager/api/all.h"

#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/tensor_meta.h"

#include "paddle/fluid/eager/tests/test_utils.h"

namespace egr {

egr::EagerTensor hook_function(const egr::EagerTensor& t) {
  auto t_dense = std::dynamic_pointer_cast<pten::DenseTensor>(t.impl());

  auto ret_meta = pten::DenseTensorMeta(t_dense->dtype(), t_dense->dims(),
                                        t_dense->layout());
  auto place = t_dense->place();
  size_t bytes_size =
      paddle::framework::product(t_dense->dims()) * SizeOf(t_dense->dtype());
  auto ret_dense = std::make_shared<pten::DenseTensor>(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          paddle::memory::Alloc(place, bytes_size), 0),
      std::move(ret_meta));

  float* t_ptr = t_dense->mutable_data<float>();
  float* ret_ptr = ret_dense->mutable_data<float>();
  for (int i = 0; i < ret_dense->numel(); i++) {
    ret_ptr[i] = t_ptr[i] + 3.0;
  }

  auto ret_impl = std::dynamic_pointer_cast<pten::TensorBase>(ret_dense);
  egr::EagerTensor ret = egr::EagerTensor();
  ret.set_impl(ret_impl);

  return ret;
}

TEST(RetainGrad, HookBeforeRetainGrad) {
  eager_test::InitEnv(paddle::platform::CPUPlace());

  // Prepare Inputs
  std::vector<egr::EagerTensor> target_tensors;
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});

  // Create Target Tensor
  egr::EagerTensor tensor = egr_utils_api::CreateTensorWithValue(
      ddim, paddle::platform::CPUPlace(), pten::DataType::FLOAT32,
      pten::DataLayout::NCHW, 1.0 /*value*/, false /*is_leaf*/);
  target_tensors.emplace_back(std::move(tensor));
  egr::EagerTensor& target_tensor = target_tensors[0];

  // Create ScaleNode
  auto scale_node_ptr = std::make_shared<GradNodeScale>(1, 1);
  scale_node_ptr->SetAttributes_scale(5.0 /*scale*/);

  // Set grad in/out meta for node0
  scale_node_ptr->SetDefaultGradInOutMeta();

  // Create AccumulationNode
  auto acc_node_ptr = std::make_shared<GradNodeAccumulation>();

  // Connect Input Tensor and ScaleNode via AutoGradMeta
  // Apply RetainGrad
  {
    // ScaleNode Hook: +3
    std::function<egr::EagerTensor(const egr::EagerTensor&)> hook =
        &hook_function;

    auto auto_grad_meta = std::make_shared<AutogradMeta>();
    auto_grad_meta->SetGradNode(
        std::dynamic_pointer_cast<GradNodeBase>(scale_node_ptr));
    auto_grad_meta->SetSingleOutRankWithSlot(0, 0);
    target_tensor.set_autograd_meta(
        std::dynamic_pointer_cast<paddle::experimental::AbstractAutogradMeta>(
            auto_grad_meta));

    egr_utils_api::RegisterGradientHookForTensor(target_tensor, hook);
    egr_utils_api::RetainGradForTensor(
        target_tensor);  // result: 1.0 + 3.0 = 4.0
  }

  // Connect ScaleNode -> AccumulationNode via Edge
  {
    auto meta = AutogradMeta();
    meta.SetSingleOutRankWithSlot(0, 0);
    meta.SetGradNode(acc_node_ptr);
    scale_node_ptr->AddEdges({&meta}, 0);
  }

  // Retain Grad for leaf tensor1
  egr::EagerTensor leaf_tensor = egr::EagerTensor();
  {
    // AccumulationNode Hook: +3
    std::function<egr::EagerTensor(const egr::EagerTensor&)> hook =
        &hook_function;

    auto auto_grad_meta = std::make_shared<AutogradMeta>();
    auto_grad_meta->SetGradNode(
        std::dynamic_pointer_cast<GradNodeBase>(acc_node_ptr));
    auto_grad_meta->SetSingleOutRankWithSlot(0, 0);
    leaf_tensor.set_autograd_meta(
        std::dynamic_pointer_cast<paddle::experimental::AbstractAutogradMeta>(
            auto_grad_meta));

    egr_utils_api::RegisterGradientHookForTensor(leaf_tensor, hook);
    egr_utils_api::RetainGradForTensor(
        leaf_tensor);  // result: 4.0*5.0 + 3.0 = 23.0
  }

  RunBackward(target_tensors, {});

  eager_test::CompareGradTensorWithValue<float>(target_tensor, 4.0);
  eager_test::CompareGradTensorWithValue<float>(leaf_tensor, 23.0);
}

TEST(RetainGrad, HookAfterRetainGrad) {
  eager_test::InitEnv(paddle::platform::CPUPlace());

  // Prepare Inputs
  std::vector<egr::EagerTensor> target_tensors;
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});

  // Create Target Tensor
  egr::EagerTensor tensor = egr_utils_api::CreateTensorWithValue(
      ddim, paddle::platform::CPUPlace(), pten::DataType::FLOAT32,
      pten::DataLayout::NCHW, 1.0 /*value*/, false /*is_leaf*/);
  target_tensors.emplace_back(std::move(tensor));
  egr::EagerTensor& target_tensor = target_tensors[0];

  // Create ScaleNode
  auto scale_node_ptr = std::make_shared<GradNodeScale>(1, 1);
  scale_node_ptr->SetAttributes_scale(5.0 /*scale*/);
  // Set grad in/out meta for node0
  scale_node_ptr->SetDefaultGradInOutMeta();
  // Create AccumulationNode
  auto acc_node_ptr = std::make_shared<GradNodeAccumulation>();

  // Connect Input Tensor and ScaleNode via AutoGradMeta
  // Apply RetainGrad
  {
    // ScaleNode Hook: +3
    std::function<egr::EagerTensor(const egr::EagerTensor&)> hook =
        &hook_function;

    auto auto_grad_meta = std::make_shared<AutogradMeta>();
    auto_grad_meta->SetGradNode(
        std::dynamic_pointer_cast<GradNodeBase>(scale_node_ptr));
    auto_grad_meta->SetSingleOutRankWithSlot(0, 0);
    target_tensor.set_autograd_meta(
        std::dynamic_pointer_cast<paddle::experimental::AbstractAutogradMeta>(
            auto_grad_meta));

    egr_utils_api::RetainGradForTensor(target_tensor);  // result: 1.0
    egr_utils_api::RegisterGradientHookForTensor(target_tensor, hook);
  }

  // Connect ScaleNode -> AccumulationNode via Edge
  {
    auto meta = AutogradMeta();
    meta.SetSingleOutRankWithSlot(0, 0);
    meta.SetGradNode(acc_node_ptr);
    scale_node_ptr->AddEdges({&meta}, 0);
  }

  // Retain Grad for leaf tensor1
  egr::EagerTensor leaf_tensor = egr::EagerTensor();
  {
    // AccumulationNode Hook: +3
    std::function<egr::EagerTensor(const egr::EagerTensor&)> hook =
        &hook_function;

    auto auto_grad_meta = std::make_shared<AutogradMeta>();
    auto_grad_meta->SetGradNode(
        std::dynamic_pointer_cast<GradNodeBase>(acc_node_ptr));
    auto_grad_meta->SetSingleOutRankWithSlot(0, 0);
    leaf_tensor.set_autograd_meta(
        std::dynamic_pointer_cast<paddle::experimental::AbstractAutogradMeta>(
            auto_grad_meta));

    egr_utils_api::RetainGradForTensor(
        leaf_tensor);  // RetainGrad for leaf tensor gets
                       // postponed, result: 4.0*5.0 + 3.0 =
                       // 23.0
    egr_utils_api::RegisterGradientHookForTensor(leaf_tensor, hook);
  }

  RunBackward(target_tensors, {});
  eager_test::CompareGradTensorWithValue<float>(target_tensor, 1.0);
  eager_test::CompareGradTensorWithValue<float>(leaf_tensor, 23.0);
}
}  // namespace egr
