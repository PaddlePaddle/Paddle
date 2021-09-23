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

#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/nodes/accumulation_node.h"
#include "paddle/fluid/eager/nodes/scale_node.h"

#include "paddle/fluid/eager/api/api.h"

#include "paddle/tcmpt/core/dense_tensor.h"
#include "paddle/tcmpt/core/tensor_meta.h"

#include "paddle/fluid/eager/tests/test_utils.h"

// TODO(jiabin): remove nolint here!!!
using namespace egr;  // NOLINT

pt::Tensor hook_function(const pt::Tensor& t) {
  auto t_dense = std::dynamic_pointer_cast<pt::DenseTensor>(t.impl());

  auto ret_meta = pt::TensorMeta(t_dense->dims(), t_dense->backend(),
                                 t_dense->type(), t_dense->layout());
  auto ret_dense = std::make_shared<pt::DenseTensor>(std::move(ret_meta),
                                                     pt::TensorStatus());

  float* t_ptr = t_dense->mutable_data<float>();
  float* ret_ptr = ret_dense->mutable_data<float>();
  for (int i = 0; i < ret_dense->numel(); i++) {
    ret_ptr[i] = t_ptr[i] + 3.0;
  }

  auto ret_impl = std::dynamic_pointer_cast<pt::TensorInterface>(ret_dense);
  pt::Tensor ret = pt::Tensor();
  ret.SetImpl(ret_impl);

  return ret;
}

/*
AccumulationNode
  |
  |retain_grad
  |hook
  |
ScaleNode
  |
  |retain_grad
  |hook
  |
 inp0
*/
TEST(RetainGrad, HookBeforeRetainGrad) {
  InitEnv(paddle::platform::CPUPlace());

  // Prepare Inputs
  std::vector<pt::Tensor> target_tensors;
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});

  // Create Target Tensor
  pt::Tensor tensor = EagerUtils::CreateTensorWithValue(
      ddim, pt::Backend::kCPU, pt::DataType::kFLOAT32, pt::DataLayout::kNCHW,
      1.0 /*value*/, false /*is_leaf*/);
  target_tensors.emplace_back(std::move(tensor));
  pt::Tensor& target_tensor = target_tensors[0];

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
    std::function<pt::Tensor(const pt::Tensor&)> hook = &hook_function;

    auto auto_grad_meta = std::make_shared<AutogradMeta>();
    auto_grad_meta->SetGradNode(
        std::dynamic_pointer_cast<GradNodeBase>(scale_node_ptr));
    auto_grad_meta->SetSingleOutRankWithSlot(0, 0);
    target_tensor.set_autograd_meta(
        std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));

    RegisterGradientHookForTensor(target_tensor, hook);
    RetainGradForTensor(target_tensor);  // result: 1.0 + 3.0 = 4.0
  }

  // Connect ScaleNode -> AccumulationNode via Edge
  {
    auto meta = AutogradMeta();
    meta.SetSingleOutRankWithSlot(0, 0);
    meta.SetGradNode(acc_node_ptr);
    scale_node_ptr->AddEdges({&meta}, 0);
  }

  // Retain Grad for leaf tensor1
  pt::Tensor leaf_tensor = pt::Tensor();
  {
    // AccumulationNode Hook: +3
    std::function<pt::Tensor(const pt::Tensor&)> hook = &hook_function;

    auto auto_grad_meta = std::make_shared<AutogradMeta>();
    auto_grad_meta->SetGradNode(
        std::dynamic_pointer_cast<GradNodeBase>(acc_node_ptr));
    auto_grad_meta->SetSingleOutRankWithSlot(0, 0);
    leaf_tensor.set_autograd_meta(
        std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));

    RegisterGradientHookForTensor(leaf_tensor, hook);
    RetainGradForTensor(leaf_tensor);  // result: 4.0*5.0 + 3.0 = 23.0
  }

  // Use Empty Grad Tensor
  RunBackward(target_tensors, {});

  // Print target tensor grad
  PADDLE_ENFORCE(
      CompareGradTensorWithValue<float>(target_tensor, 4.0) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 4.0));

  // Print leaf tensor grad
  PADDLE_ENFORCE(
      CompareGradTensorWithValue<float>(leaf_tensor, 23.0) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 23.0));
}

/*
AccumulationNode
  |
  |hook
  |retain_grad
  |
ScaleNode
  |
  |hook
  |retain_grad
  |
 inp0
*/
TEST(RetainGrad, HookAfterRetainGrad) {
  InitEnv(paddle::platform::CPUPlace());

  // Prepare Inputs
  std::vector<pt::Tensor> target_tensors;
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});

  // Create Target Tensor
  pt::Tensor tensor = EagerUtils::CreateTensorWithValue(
      ddim, pt::Backend::kCPU, pt::DataType::kFLOAT32, pt::DataLayout::kNCHW,
      1.0 /*value*/, false /*is_leaf*/);
  target_tensors.emplace_back(std::move(tensor));
  pt::Tensor& target_tensor = target_tensors[0];

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
    std::function<pt::Tensor(const pt::Tensor&)> hook = &hook_function;

    auto auto_grad_meta = std::make_shared<AutogradMeta>();
    auto_grad_meta->SetGradNode(
        std::dynamic_pointer_cast<GradNodeBase>(scale_node_ptr));
    auto_grad_meta->SetSingleOutRankWithSlot(0, 0);
    target_tensor.set_autograd_meta(
        std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));

    RetainGradForTensor(target_tensor);  // result: 1.0
    RegisterGradientHookForTensor(target_tensor, hook);
  }

  // Connect ScaleNode -> AccumulationNode via Edge
  {
    auto meta = AutogradMeta();
    meta.SetSingleOutRankWithSlot(0, 0);
    meta.SetGradNode(acc_node_ptr);
    scale_node_ptr->AddEdges({&meta}, 0);
  }

  // Retain Grad for leaf tensor1
  pt::Tensor leaf_tensor = pt::Tensor();
  {
    // AccumulationNode Hook: +3
    std::function<pt::Tensor(const pt::Tensor&)> hook = &hook_function;

    auto auto_grad_meta = std::make_shared<AutogradMeta>();
    auto_grad_meta->SetGradNode(
        std::dynamic_pointer_cast<GradNodeBase>(acc_node_ptr));
    auto_grad_meta->SetSingleOutRankWithSlot(0, 0);
    leaf_tensor.set_autograd_meta(
        std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));

    RetainGradForTensor(leaf_tensor);  // RetainGrad for leaf tensor gets
                                       // postponed, result: 4.0*5.0 + 3.0 =
                                       // 23.0
    RegisterGradientHookForTensor(leaf_tensor, hook);
  }

  // Use Empty Grad Tensor
  RunBackward(target_tensors, {});

  // Print target tensor grad
  PADDLE_ENFORCE(
      CompareGradTensorWithValue<float>(target_tensor, 1.0) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 1.0));

  // Print leaf tensor grad
  PADDLE_ENFORCE(
      CompareGradTensorWithValue<float>(leaf_tensor, 23.0) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 23.0));
}
