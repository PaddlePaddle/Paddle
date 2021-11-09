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

#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/tensor_meta.h"

#include "paddle/fluid/eager/tests/test_utils.h"

using namespace egr;  // NOLINT

/*
AccumulationNode
  |
ScaleNode
  |
 inp0
*/
TEST(CrossBatchAccumulation, SingleScaleNode) {
  // Prepare Device Contexts
  InitEnv(paddle::platform::CPUPlace());

  // Prepare Inputs
  std::vector<egr::EagerTensor> target_tensors;
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});

  // Create Target Tensor
  egr::EagerTensor tensor = EagerUtils::CreateTensorWithValue(
      ddim, paddle::platform::CPUPlace(), pten::DataType::FLOAT32,
      pten::DataLayout::NCHW, 1.0 /*value*/, false /*is_leaf*/);
  target_tensors.emplace_back(std::move(tensor));
  egr::EagerTensor& target_tensor = target_tensors[0];

  egr::EagerTensor leaf_tensor = egr::EagerTensor();
  {
    // Create ScaleNode
    auto scale_node_ptr = std::make_shared<GradNodeScale>(1, 1);
    scale_node_ptr->SetAttributes_scale(5.0 /*scale*/);

    // Set grad in/out meta for node0
    scale_node_ptr->SetDefaultGradInOutMeta();

    // Create AccumulationNode
    auto acc_node_ptr = std::make_shared<GradNodeAccumulation>();

    // Connect Input Tensor and ScaleNode via AutoGradMeta
    // Apply RetainGrad
    AutogradMeta* auto_grad_meta = EagerUtils::autograd_meta(&target_tensor);
    auto_grad_meta->SetGradNode(
        std::dynamic_pointer_cast<GradNodeBase>(scale_node_ptr));
    auto_grad_meta->SetSingleOutRankWithSlot(0, 0);
    RetainGradForTensor(target_tensor);  // result: 1.0

    // Connect ScaleNode -> AccumulationNode via Edge
    auto meta = AutogradMeta();
    meta.SetSingleOutRankWithSlot(0, 0);
    meta.SetGradNode(acc_node_ptr);
    scale_node_ptr->AddEdges({&meta}, 0);

    AutogradMeta* auto_grad_meta1 = EagerUtils::autograd_meta(&leaf_tensor);
    auto_grad_meta1->SetGradNode(
        std::dynamic_pointer_cast<GradNodeBase>(acc_node_ptr));
    auto_grad_meta1->SetSingleOutRankWithSlot(0, 0);
    RetainGradForTensor(leaf_tensor);
  }

  // Use Empty Grad Tensor
  RunBackward(target_tensors, {});

  // target tensor's grad should remain the same
  PADDLE_ENFORCE(
      CompareGradTensorWithValue<float>(target_tensor, 1.0) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 1.0));

  // Leaf tensor should keep accumulated grad
  PADDLE_ENFORCE(
      CompareGradTensorWithValue<float>(leaf_tensor, 5.0) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 5.0));

  // Cross-Batch Accumulation
  RunBackward(target_tensors, {});

  // target tensor's grad should remain the same
  PADDLE_ENFORCE(
      CompareGradTensorWithValue<float>(target_tensor, 1.0) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 1.0));

  // Leaf tensor should keep accumulated grad
  PADDLE_ENFORCE(
      CompareGradTensorWithValue<float>(leaf_tensor, 10.0) == true,
      paddle::platform::errors::Fatal("Numerical Error, Expected %f", 10.0));
}
